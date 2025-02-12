"""
这个里面，make_dataset 是核心入口。pretrain_gpt.py 也会调用到这个，它是来read。

preprocess_data 同样会调用这里，他主要是write数据。

"""

# =============================================================================
# 模块说明：
# 1. 本模块主要用于构造和加载经过预处理的索引数据集（Indexed Dataset），支持多种实现方式：
#    - lazy：懒加载
#    - cached：预缓存
#    - mmap：内存映射
# 2. 代码依赖的外部库有：numpy、torch、struct、os、shutil、functools、itertools等，
#    其中 torch 用于和 PyTorch 框架对接，numpy 用于高效数组操作。
# 3. 此实现借鉴了 Facebook 的 Fairseq 和 Megatron-DeepSpeed 中的相关思路。
# =============================================================================

from functools import lru_cache  # 用于对部分方法进行缓存，提高重复调用效率
import os                        # 文件和路径操作
import shutil                    # 文件复制操作
import struct                    # 用于处理二进制数据的打包与解包
from itertools import accumulate  # 用于计算累加和

import numpy as np               # 高效数组操作库
import torch                     # PyTorch 框架
from megatron import print_rank_0  # 用于打印信息（仅在 rank 0 时输出，多用于分布式环境下）

# =============================================================================
# 【函数功能】选择合适的数据类型
# 根据词汇表大小 vocab_size（若提供）来确定最适合的 numpy 数据类型
# 例如：若词汇表小于 65500，则使用 np.uint16，否则使用 np.int32
# -----------------------------------------------------------------------------
def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


# =============================================================================
# 【函数功能】返回支持的数据集实现列表
# 返回可选的字符串列表，表示支持的实现方式：'lazy', 'cached', 'mmap'
# -----------------------------------------------------------------------------
def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


# =============================================================================
# 【函数功能】自动推断数据集的实现方式
# 读取索引文件的前几个字节（magic header），判断使用 cached 还是 mmap 实现
# -----------------------------------------------------------------------------
def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            # 判断 magic 值与各自实现的 magic 值是否匹配
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


# =============================================================================
# 【函数功能】构造数据集构造器（builder）
# 根据传入的实现方式 impl 和词汇表大小 vocab_size，返回相应 builder 实例
# 目的：支持不同实现方式下的数据集构建方法（例如 mmap 方式需要指定数据类型）
# -----------------------------------------------------------------------------
def make_builder(out_file, impl, vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    else:
        return IndexedDatasetBuilder(out_file)


# =============================================================================
# 【函数功能】构造数据集实例
# 根据指定的实现方式（impl）返回对应的数据集对象：
#   - lazy：返回 IndexedDataset（懒加载方式）
#   - cached：返回 IndexedCachedDataset（支持预缓存数据）
#   - mmap：返回 MMapIndexedDataset（内存映射方式）
#
# 参数 skip_warmup 用于控制 mmap 实现是否跳过预热（warmup）步骤
# -----------------------------------------------------------------------------
def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)  # 通过数据集开头的魔法字符串来推断，我们用的都是mmap的，默认的也是这个。
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


# =============================================================================
# 【函数功能】判断数据集文件是否存在
# 根据不同实现方式，检查索引文件和数据文件是否存在
# -----------------------------------------------------------------------------
def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


# =============================================================================
# 【函数功能】读取 n 个 int64 数据
# 从文件对象 f 中读取 n 个 int64 数字，并返回 numpy 数组，数组元素类型为 np.int64
#
# 注意：readinto 方法要求目标数组必须预先分配好空间
# -----------------------------------------------------------------------------
def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


# =============================================================================
# 【函数功能】将一个 numpy 数组写入文件
# 写入数组 a（数据类型为 np.int64）到文件对象 f 中
# -----------------------------------------------------------------------------
def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


# =============================================================================
# 【变量说明】dtypes 字典
# 该字典将整数编码（1~8）映射到相应的 numpy 数据类型：
#   1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64,
#   6: np.float64, 7: np.float32, 8: np.uint16
#
# 用于在写入索引文件时标识数据类型，并在读取时恢复数据格式。
# -----------------------------------------------------------------------------
dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


# =============================================================================
# 【函数功能】根据 numpy dtype 获取对应的编码数字
# 遍历 dtypes 字典，若找到匹配则返回对应的 key，否则抛出 ValueError 异常
#
# 注意：用于在构造索引文件时写入数据类型编码
# -----------------------------------------------------------------------------
def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


# =============================================================================
# 【函数功能】构造索引文件路径
# 根据前缀路径（basename），添加 .idx 后缀构成完整的索引文件路径
# -----------------------------------------------------------------------------
def index_file_path(prefix_path):
    return prefix_path + '.idx'


# =============================================================================
# 【函数功能】构造数据文件路径
# 根据前缀路径（basename），添加 .bin 后缀构成完整的数据文件路径
# -----------------------------------------------------------------------------
def data_file_path(prefix_path):
    return prefix_path + '.bin'


# =============================================================================
# 【函数功能】创建文档索引（document index）
# 参数 sizes 为句子或样本的大小列表，当遇到 size==0（空句子）时认为是文档分隔符，
# 因此每个空句子后面的第一个句子被视为新文档的开始位置。
#
# 返回一个列表，其中第一个元素为 0，后续元素为各文档的起始句子索引
# -----------------------------------------------------------------------------
def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


# =============================================================================
# 【类说明】IndexedDataset
# 继承自 torch.utils.data.Dataset，用于加载存储在二进制文件中的索引数据集
#
# 核心思路：
#   1. 索引文件（.idx）中存储了元数据，包括数据个数（_len）、每个样本的大小（sizes）、偏移量等信息，
#      以及文档索引（doc_idx）。
#   2. 数据文件（.bin）中存储了具体数据，通过索引中的 data_offsets 定位数据的位置。
#
# 注意：__getitem__ 方法支持两种索引方式：
#   - 整数索引：返回一个 numpy 数组
#   - 切片索引：返回一个列表，其中每个元素对应一个样本
# -----------------------------------------------------------------------------
class IndexedDataset(torch.utils.data.Dataset):
    # 索引文件的魔数，确保文件格式正确
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    # -------------------------------------------------------------------------
    # 【构造函数】初始化 IndexedDataset
    # 参数 path 为数据集的前缀路径（basename），构造器会读取索引文件并初始化相关变量
    # 变量说明：
    #   - self.path: 数据集路径（字符串）
    #   - self.data_file: 数据文件句柄（初始为 None，在第一次调用 __getitem__ 时打开）
    #   - self._len: 样本个数（整数）
    #   - self.s: 存储所有维度信息的总数（整数）
    #   - self.dtype: 数据存储时的 numpy 数据类型
    #   - self.element_size: 数据元素的字节数
    #   - self.dim_offsets: 每个样本维度描述的起始位置（np.int64 数组，长度为 _len+1）
    #   - self.data_offsets: 数据在二进制文件中的偏移量（np.int64 数组，长度为 _len+1）
    #   - self.sizes: 每个维度的大小信息（np.int64 数组，长度为 s）
    #   - self.doc_idx: 文档索引信息（np.int64 数组）
    # -------------------------------------------------------------------------
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None  # 数据文件句柄，延迟加载
        self.read_index(path)  # 立即读取索引文件

    # -------------------------------------------------------------------------
    # 【方法】read_index
    # 从索引文件中读取各项元数据，按顺序读取：
    #   1. 魔数（magic）：确保文件格式正确
    #   2. 版本号（version）：目前只支持版本 1
    #   3. 数据类型编码（code）及每个元素字节数（element_size）
    #   4. 样本个数（_len）和维度数总数（s）
    #   5. 文档数量（doc_count）
    #   6. 各样本的维度偏移量（dim_offsets）
    #   7. 各样本数据在二进制文件中的偏移量（data_offsets）
    #   8. 各维度的大小（sizes）
    #   9. 文档索引（doc_idx）
    #
    # 注意：文件读取均采用二进制模式，部分数据使用 struct.unpack 解包
    # -------------------------------------------------------------------------
    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            # 解包为无符号长整数（<Q 表示小端格式，8 字节无符号整数）
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            # dim_offsets 和 data_offsets 数组的长度均为 _len + 1，用于快速查找样本边界
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    # -------------------------------------------------------------------------
    # 【方法】read_data
    # 打开数据文件 (.bin) 用于读取数据
    # -----------------------------------------------------------------------------
    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    # -------------------------------------------------------------------------
    # 【方法】check_index
    # 检查索引 i 是否越界；若 i 小于 0 或大于等于 _len，则抛出 IndexError 异常
    # -----------------------------------------------------------------------------
    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    # -------------------------------------------------------------------------
    # 【析构函数】__del__
    # 释放数据文件资源，确保文件关闭（防止内存泄漏）
    # -----------------------------------------------------------------------------
    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # -------------------------------------------------------------------------
    # 【方法】__getitem__
    # 支持两种索引方式：整数索引和切片索引
    #
    # 【阅读提示】先阅读整数索引部分，再阅读切片索引部分。
    #
    # 整数索引：
    #   1. 若数据文件未打开则调用 read_data 打开数据文件。
    #   2. 调用 check_index 检查 i 是否在合法范围内。
    #   3. 根据 dim_offsets 数组确定该样本的维度信息（tensor_size）。
    #   4. 分配一个空 numpy 数组 a，数据类型为 self.dtype，尺寸为 tensor_size。
    #   5. 根据 data_offsets 定位文件位置，并读取对应的字节到 a 中。
    #   6. 返回 numpy 数组 a。
    #
    # 切片索引：
    #   1. 切片必须连续（step 必须为 1）。
    #   2. 根据 dim_offsets 获取多个样本的尺寸信息，将所有尺寸加和得到总大小。
    #   3. 读取一大块数据到 numpy 数组 a，然后根据累加和（accumulate）将 a 分割成多个样本。
    # -----------------------------------------------------------------------------
    # 变量说明：
    #   - idx: 索引值（整数或 slice）
    #   - tensor_size: 当前样本的尺寸列表（np.int64 数组的切片）
    # -----------------------------------------------------------------------------
    # 性能分析：
    #   - 每次调用 __getitem__ 都会进行文件 seek 和 readinto 操作，适合大规模数据的随机访问
    #   - 对于连续切片访问建议预先缓存或使用 mmap 实现
    # -----------------------------------------------------------------------------
    # 错误处理：若索引越界，则抛出 IndexError；若切片的步长不为 1，则抛出 ValueError
    # -----------------------------------------------------------------------------
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            # 通过 dim_offsets 获取当前样本在 sizes 数组中的范围，即 tensor 的每个维度大小
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            # 根据 tensor_size 分配内存（numpy 数组）
            a = np.empty(tensor_size, dtype=self.dtype)
            # 定位到数据文件中的起始位置（偏移量乘以每个元素的字节数）
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            # 读取数据到 a 中
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            # 获取所有样本的尺寸
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)  # 总数据量
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            # 根据累加和计算分割位置，将一大块数据分割成各个样本
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    # -----------------------------------------------------------------------------
    # 【方法】__len__
    # 返回数据集中样本的总数（_len）
    # -----------------------------------------------------------------------------
    def __len__(self):
        return self._len

    # -----------------------------------------------------------------------------
    # 【方法】num_tokens / size
    # 返回指定索引样本的 token 数或尺寸信息
    # -----------------------------------------------------------------------------
    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    # -----------------------------------------------------------------------------
    # 【静态方法】exists
    # 判断数据集是否存在（索引文件和数据文件均存在）
    # -----------------------------------------------------------------------------
    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    # -----------------------------------------------------------------------------
    # 【属性】supports_prefetch
    # 用于指示该实现是否支持预取（prefetch）功能，返回 False 表示不支持
    # -----------------------------------------------------------------------------
    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


# =============================================================================
# 【类说明】IndexedCachedDataset
# 继承自 IndexedDataset，增加了缓存功能，用于在预取数据后直接从内存中复制数据，
# 以减少重复 I/O 操作，提高数据访问速度。
#
# 主要新增：
#   - self.cache: 用于存放预取的数据（numpy 数组）
#   - self.cache_index: 用于记录每个样本在 cache 中的起始位置（字节偏移）
#
# -----------------------------------------------------------------------------
class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = None          # 缓存数组（numpy 数组）
        self.cache_index = {}      # 缓存索引：样本索引 -> 在 cache 中的起始位置

    @property
    def supports_prefetch(self):
        return True

    # -----------------------------------------------------------------------------
    # 【方法】prefetch
    # 预取给定 indices 的数据，将数据存入 self.cache，并建立 cache_index 索引。
    #
    # 1. 如果所有 requested indices 均已经缓存，则直接返回。
    # 2. 计算所有待缓存数据的总大小，并分配一个大数组存放所有数据。
    # 3. 按索引顺序读取数据到缓存中，并更新 cache_index。
    # 4. 读取完毕后关闭原数据文件，以便支持 pickle 化（序列化）。
    # -----------------------------------------------------------------------------
    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # 预取完成后关闭数据文件，以便对象可序列化
            self.data_file.close()
            self.data_file = None

    # -----------------------------------------------------------------------------
    # 【方法】__getitem__
    # 重写 __getitem__ 方法，直接从缓存中读取数据，而非每次进行文件 I/O。
    #
    # 注意：同样支持整数索引和切片索引，但切片索引目前采用逐个样本读取的“hack”方式，
    #       后续可考虑进一步优化。
    # -----------------------------------------------------------------------------
    # 使用 lru_cache 注释被注释掉，如有必要可启用缓存优化（目前缓存命中率已在 prefetch 中控制）
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx: ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # 切片处理：依次调用 __getitem__（可能存在性能瓶颈，建议预先缓存需要的切片数据）
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


# =============================================================================
# 【类说明】IndexedDatasetBuilder
# 用于构造数据集，将 PyTorch tensor 数据写入二进制数据文件 (.bin)，
# 同时记录相关索引信息，并最终生成索引文件 (.idx)
#
# 主要属性：
#   - self.out_file: 数据输出文件句柄（以二进制写模式打开）
#   - self.dtype: 数据存储时的 numpy 数据类型
#   - self.data_offsets: 记录每个样本在数据文件中的起始偏移（单位：元素数），初始值 [0]
#   - self.dim_offsets: 记录每个样本的维度信息在 sizes 数组中的起始索引，初始值 [0]
#   - self.sizes: 记录所有样本每个维度的大小（例如 shape 信息）的列表
#   - self.element_size: 单个元素所占字节数，根据 dtype 查表获得
#   - self.doc_idx: 文档索引，用于标识文档起始位置，初始值 [0]
#
# 主要方法：
#   - add_item: 添加一个样本，将 tensor 数据写入文件，同时更新 offsets 和 sizes 信息
#   - end_document: 标记当前已写入的数据作为一个文档结束，更新 doc_idx
#   - merge_file_: 将另一个文件的数据合并到当前 builder 中（支持多文件拼接）
#   - finalize: 结束构造，关闭数据文件，并生成索引文件
#
# -----------------------------------------------------------------------------
class IndexedDatasetBuilder(object):
    # 每种 numpy 数据类型对应的字节数（用于写入时计算偏移量）
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float32: 4,
        np.float64: 8,
    }

    # -----------------------------------------------------------------------------
    # 【构造函数】初始化 builder
    # 参数 out_file 为数据文件输出路径，dtype 为数据存储时的 numpy 数据类型（默认 np.int32）
    # -----------------------------------------------------------------------------
    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    # -----------------------------------------------------------------------------
    # 【方法】add_item
    # 参数 tensor 为待写入的 PyTorch tensor（例如 torch.Tensor 对象）
    #
    # 步骤：
    #   1. 将 tensor 转换为 numpy 数组，并确保数据类型与 builder 中设置的一致。
    #   2. 写入二进制文件，返回写入的字节数（bytes）。
    #   3. 更新 data_offsets：新偏移 = 上一个偏移 + 写入元素数（字节数除以每个元素字节数）。
    #   4. 遍历 tensor 的每个维度大小，添加到 sizes 列表中。
    #   5. 更新 dim_offsets：新偏移 = 上一个 dim_offsets 加上 tensor 的维度数。
    # -----------------------------------------------------------------------------
    def add_item(self, tensor):
        # 注意：tensor.numpy() 将 tensor 转换为 numpy 数组，假设数据在 CPU 上
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    # -----------------------------------------------------------------------------
    # 【方法】end_document
    # 表示当前文档结束，在 doc_idx 中添加当前 sizes 的长度，标记一个文档边界
    # -----------------------------------------------------------------------------
    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    # -----------------------------------------------------------------------------
    # 【方法】merge_file_
    # 将另一个数据集文件（another_file）合并到当前 builder 中
    #
    # 步骤：
    #   1. 读取另一个文件的索引信息，并验证数据类型一致。
    #   2. 计算合并前后的偏移量，并更新 data_offsets、dim_offsets、sizes 和 doc_idx。
    #   3. 将另一个文件的二进制数据追加写入当前数据文件中。
    #
    # 注意：此方法可用于数据集文件的分布式拼接。
    # -----------------------------------------------------------------------------
    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        doc_offset = len(self.sizes)

        begin = self.data_offsets[-1]
        for data_offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + data_offset)
        self.sizes.extend(index.sizes)

        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        # 注意：这里利用 numpy 的广播特性，将 index.doc_idx 中的值偏移后追加
        self.doc_idx.extend((doc_offset + index.doc_idx)[1:])

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    # -----------------------------------------------------------------------------
    # 【方法】finalize
    # 结束数据集构造，关闭数据文件，并生成索引文件
    #
    # 具体步骤：
    #   1. 关闭数据文件句柄。
    #   2. 打开索引文件（以二进制写模式）。
    #   3. 写入魔数、版本号、数据类型编码和元素大小等信息。
    #   4. 写入样本数、sizes 数量、文档数量。
    #   5. 写入 dim_offsets、data_offsets、sizes、doc_idx 四个长整型数组。
    #   6. 关闭索引文件。
    #
    # 设计模式：此处采用 Builder 模式将数据和索引分离写入，便于后续随机访问。
    # -----------------------------------------------------------------------------
    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


# =============================================================================
# 【函数功能】预热 mmap 文件
# 读取文件，直到文件末尾，用于将数据提前加载到操作系统缓存中（减少后续访问延迟）
#
# 参数 path 为待预热文件的路径
# -----------------------------------------------------------------------------
def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        # 每次读取 100MB，直至文件末尾
        while stream.read(100 * 1024 * 1024):
            pass


# =============================================================================
# 【函数功能】将包含累加和的数组转换为排它扫描（exclusive scan）
# 给定一个包含累计和（inclusive scan）的数组，将其向右平移一位：
# 例如：[10, 30, 35, 50] 变为 [0, 10, 30, 35]
#
# 说明：这是一种常见的前缀和算法，用于计算偏移量等问题
# -----------------------------------------------------------------------------
def exscan_from_cumsum_(arr):
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


# =============================================================================
# 【函数功能】计算数据指针和总字节数
# 根据 sizes 数组（每个元素表示个样本所包含的元素个数），
# 先乘以 elemsize 得到每个样本的字节数，再计算累计和得到各样本在数据文件中的起始字节偏移量，
# 同时返回所有数据的总字节数。
#
# 参数：
#   - sizes: 样本中元素个数的列表或数组
#   - elemsize: 单个元素的字节数（例如 4 或 8）
#   - dtype: numpy 数据类型，用于创建数组
#
# 返回：
#   - pointers: 包含各样本起始偏移（单位：字节）的 numpy 数组（经过 exclusive scan）
#   - bytes_last: 总字节数（最后一个元素的累计和）
#
# 相关知识：前缀和（prefix sum）和排它扫描在并行计算中常见
# -----------------------------------------------------------------------------
def get_pointers_with_total(sizes, elemsize, dtype):
    # 将 sizes 数组转换为指定数据类型的 numpy 数组
    pointers = np.array(sizes, dtype=dtype)
    # 每个元素乘以 elemsize 得到对应的字节数
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)
    # 总字节数：sizes 数组的累计和最后一项
    bytes_last = pointers[-1] if len(sizes) > 0 else 0
    # 转换为 exclusive scan（排它扫描）
    exscan_from_cumsum_(pointers)
    return pointers, bytes_last


# =============================================================================
# 【类说明】MMapIndexedDataset
# 基于内存映射（mmap）技术实现的数据集加载，继承自 torch.utils.data.Dataset，
# 能够在不加载全部数据到内存的情况下实现快速随机访问。
#
# 主要设计思路：
#   1. 利用内存映射将数据文件映射到内存，减少 I/O 开销。
#   2. 内部类 Index 用于解析索引文件，读取 sizes、pointers（偏移量）和 doc_idx。
#   3. __getitem__ 方法支持整数索引和切片索引，返回 numpy 数组或样本列表。
#
# -----------------------------------------------------------------------------
class MMapIndexedDataset(torch.utils.data.Dataset):
    # -------------------------------------------------------------------------
    # 【内部类】Index
    # 用于读取和解析内存映射索引文件 (.idx)
    # -----------------------------------------------------------------------------
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        # ---------------------------------------------------------------------
        # 【类方法】writer
        # 返回一个上下文管理器，用于写入索引文件。使用 with 语句自动管理文件打开和关闭。
        #
        # 内部类 _Writer 的主要功能：
        #   - __enter__: 打开文件，并写入魔数和版本号
        #   - write: 根据传入的 sizes 和 doc_idx 写入索引信息，包含：
        #         样本数量、文档数量、sizes 数组、pointers 数组（由 sizes 计算得到）、doc_idx 数组
        #   - __exit__: 关闭文件句柄
        # -----------------------------------------------------------------------------
        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes, npdtype):
                    # 利用 get_pointers_with_total 函数计算指针（偏移量）
                    pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                    return pointers

                def write(self, sizes, doc_idx):
                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))
                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order='C'))
                    del sizes32
                    pointers = self._get_pointers(sizes, np.int64)
                    del sizes
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        # ---------------------------------------------------------------------
        # 【构造函数】Index.__init__
        # 解析索引文件，读取 header 信息、版本、数据类型编码、样本数、文档数，
        # 以及 sizes、pointers 和 doc_idx 数组
        #
        # 参数 skip_warmup 控制是否跳过预热（warmup）内存映射文件操作
        # -----------------------------------------------------------------------------
        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            # 利用 numpy 的 memmap 加载整个索引文件到内存（只读模式）
            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            # 使用 memoryview 避免数据拷贝
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        # -----------------------------------------------------------------------------
        # 【析构函数】__del__
        # 关闭 memmap 文件句柄，释放内存
        # -----------------------------------------------------------------------------
        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        # -----------------------------------------------------------------------------
        # 【方法】__getitem__
        # 利用 lru_cache 缓存最近访问的 8 个索引，返回一个二元组 (pointer, size)
        # pointer 表示数据在二进制数据文件中的字节偏移量，size 为该样本包含的元素个数
        # -----------------------------------------------------------------------------
        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    # -----------------------------------------------------------------------------
    # 【构造函数】MMapIndexedDataset.__init__
    # 初始化时调用 _do_init 完成数据的内存映射和索引加载
    # -----------------------------------------------------------------------------
    def __init__(self, path, skip_warmup=False):
        super().__init__()
        self._path = None
        self._index = None
        self._bin_buffer = None
        self._do_init(path, skip_warmup)

    # -----------------------------------------------------------------------------
    # 【方法】__getstate__ 与 __setstate__
    # 用于支持对象序列化（pickle），只保存路径信息，重构时重新调用 _do_init
    # -----------------------------------------------------------------------------
    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    # -----------------------------------------------------------------------------
    # 【方法】_do_init
    # 核心初始化流程：
    #   1. 保存路径 self._path
    #   2. 读取索引文件（调用内部 Index 类构造函数）
    #   3. 对数据文件进行预热（若不跳过）
    #   4. 使用 numpy.memmap 将数据文件映射到内存，并创建 memoryview 以减少拷贝
    # -----------------------------------------------------------------------------
    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    # -----------------------------------------------------------------------------
    # 【析构函数】__del__
    # 关闭数据文件映射，释放内存
    # -----------------------------------------------------------------------------
    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    # -----------------------------------------------------------------------------
    # 【方法】__len__
    # 返回数据集中样本的总数，由索引文件中获取
    # -----------------------------------------------------------------------------
    def __len__(self):
        return len(self._index)

    # -----------------------------------------------------------------------------
    # 【方法】__getitem__
    # 支持两种索引方式：
    #   - 整数索引：通过索引文件中记录的偏移量和尺寸，从内存映射的二进制数据中提取数据，
    #                 返回 numpy 数组。
    #   - 切片索引：要求连续，返回样本列表
    #
    # 注意：切片索引中使用了 accumulate 计算累加和来分割数据块。
    #
    # 错误处理：若索引类型非整数或 slice，则抛出 TypeError
    # -----------------------------------------------------------------------------
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    # -----------------------------------------------------------------------------
    # 【方法】get
    # 获取单个样本，可指定偏移量 offset 和读取长度 length（部分读取）
    # 与 __getitem__ 不同的是，不支持切片，仅支持整数索引
    # -----------------------------------------------------------------------------
    def get(self, idx, offset=0, length=None):
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    def size(self, index):
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def dtype(self):
        return self._index.dtype


# =============================================================================
# 【类说明】MMapIndexedDatasetBuilder
# 用于构造基于内存映射实现的数据集。与 IndexedDatasetBuilder 类似，
# 但在构造索引时调用 MMapIndexedDataset.Index.writer 生成对应的索引文件。
#
# 主要属性：
#   - self._data_file: 数据文件写入句柄（以二进制写模式打开）
#   - self._dtype: 数据存储时的 numpy 数据类型（默认 np.int64）
#   - self._sizes: 用于存储各样本数据的元素个数列表
#   - self._doc_idx: 文档索引，初始为 [0]
# -----------------------------------------------------------------------------
class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    # -----------------------------------------------------------------------------
    # 【方法】add_item
    # 添加单个样本，将 tensor 数据转换为 numpy 数组后写入数据文件，并记录该样本的元素个数
    # -----------------------------------------------------------------------------
    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    # -----------------------------------------------------------------------------
    # 【方法】add_doc
    # 添加整篇文档的数据：传入一个 tensor（可以是多样本拼接）及对应的 sizes 列表，
    # 写入数据后更新 _sizes 列表，并在 _doc_idx 中记录文档边界
    # -----------------------------------------------------------------------------
    def add_doc(self, tensor, sizes):
        np_array = np.array(tensor, dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.extend(sizes)
        self._doc_idx.append(len(self._sizes))

    # -----------------------------------------------------------------------------
    # 【方法】end_document
    # 标记当前文档结束：将当前 _sizes 长度加入 _doc_idx 中
    # -----------------------------------------------------------------------------
    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    # -----------------------------------------------------------------------------
    # 【方法】merge_file_
    # 合并另一个文件的数据到当前数据集中，更新 _sizes、_doc_idx 并复制二进制数据
    # -----------------------------------------------------------------------------
    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        # Concatenate data：利用 shutil.copyfileobj 复制二进制数据
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    # -----------------------------------------------------------------------------
    # 【方法】finalize
    # 完成数据集构造，关闭数据文件句柄，并利用 MMapIndexedDataset.Index.writer 写入索引文件
    # -----------------------------------------------------------------------------
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
