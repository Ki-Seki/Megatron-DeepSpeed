# =============================================================================
# Megatron Tokenizers 模块
# =============================================================================
# 本模块实现了多种分词器（tokenizer）的封装，支持 BERT、GPT2、SentencePiece、
# HuggingFace 等多种分词方案。主要用于将输入文本转换为对应的 token id 序列，
# 并提供反向转换接口。代码中针对不同的模型需求进行了特殊 token（如 CLS、SEP、
# PAD、MASK、BOS、EOS 等）的处理，同时考虑了分布式环境下 vocab size 对齐的需求。
#
# 依赖与环境：
# - Python 3.x
# - transformers 库（用于 HF 分词器）：pip install transformers
# - sentencepiece 库（用于 SentencePiece 分词器）：pip install sentencepiece
# - 自定义的 bert_tokenization 和 gpt2_tokenization 模块
#
# 编码风格与设计模式：
# - 采用面向对象设计，使用抽象基类(AbstractTokenizer)来约束各分词器接口，
#   符合策略模式的思想。
# - 使用 assert 检查必要参数，保证调用者传入正确的参数。
#
# 阅读顺序提示：
# 1. 从 build_tokenizer() 函数入口开始，了解如何根据参数选择不同的分词器。
# 2. 关注 _vocab_size_with_padding() 函数，理解 vocab 大小的填充策略。
# 3. 阅读 AbstractTokenizer 抽象类，了解各分词器需实现的接口。
# 4. 分别查看各具体分词器的实现：_BertWordPieceTokenizer、_GPT2BPETokenizer、
#    _SentencePieceTokenizer、_GPTSentencePieceTokenizer、_NullTokenizer、_HFTokenizer。
# =============================================================================

# -----------------------------------------------------------------------------
# 模块引用与依赖导入
# -----------------------------------------------------------------------------
from abc import ABC
from abc import abstractmethod

from transformers import AutoTokenizer  # HuggingFace 分词器依赖
from .bert_tokenization import FullTokenizer as FullBertTokenizer  # 自定义 BERT 分词器实现
from .gpt2_tokenization import GPT2Tokenizer  # 自定义 GPT2 分词器实现

# -----------------------------------------------------------------------------
# 构造函数：根据传入参数构建对应的分词器实例
# -----------------------------------------------------------------------------
def build_tokenizer(args):
    """
    初始化分词器，根据 args 中的 tokenizer_type 参数选择不同的分词器实现。

    参数:
      - args: 包含各项配置参数的对象，主要字段包括：
          tokenizer_type (str): 指定分词器类型，如 'BertWordPieceLowerCase', 'GPT2BPETokenizer' 等；
          vocab_file (str): 词汇表文件路径（部分分词器需要）；
          merge_file (str): GPT2 BPE 分词器需要的 merge 文件；
          tokenizer_model (str): SentencePiece/HF 分词器的模型文件或名称；
          vocab_extra_ids (int): 额外特殊 token 数量（如 T5 模型中的 <extra_id_i>）；
          vocab_size (int): 对于 NullTokenizer，需要指定词汇表大小；
          seq_length (int): 序列最大长度，HF 分词器使用；
          trust_remote_code (bool): HF 分词器参数，允许下载远程代码；
          rank (int): 分布式训练中的进程编号，0 表示主进程；
          make_vocab_size_divisible_by (int): vocab 大小需被此数整除的基数；
          tensor_model_parallel_size (int): 模型并行的张量大小。
          
    返回:
      分词器实例对象，对象类型为对应的具体分词器子类。

    阅读提示：首先检查 rank==0 输出构建信息，然后根据 tokenizer_type 分支，
    最后调用 _vocab_size_with_padding 对词汇表大小进行填充处理。
    """
    # 如果当前进程为主进程，则输出构建信息
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # 根据 tokenizer_type 选择具体的分词器构造方式
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        # BERT 分词器（小写模式）：要求 vocab_file 非空
        assert args.vocab_file is not None, "vocab_file 参数不能为空"
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'BertWordPieceCase':
        # BERT 分词器（区分大小写）：要求 vocab_file 非空
        assert args.vocab_file is not None, "vocab_file 参数不能为空"
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        # GPT2 BPE 分词器：要求 vocab_file 与 merge_file 均非空
        assert args.vocab_file is not None, "vocab_file 参数不能为空"
        assert args.merge_file is not None, "merge_file 参数不能为空"
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        # SentencePiece 分词器：要求 tokenizer_model 非空
        assert args.tokenizer_model is not None, "tokenizer_model 参数不能为空"
        tokenizer = _SentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPTSentencePieceTokenizer':
        # GPT 专用的 SentencePiece 分词器：要求 tokenizer_model 非空
        assert args.tokenizer_model is not None, "tokenizer_model 参数不能为空"
        tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'NullTokenizer':
        # 空分词器：简单按空格分词，需要指定 vocab_size（通常用于调试）
        assert args.vocab_size is not None, "vocab_size 参数不能为空"
        tokenizer = _NullTokenizer(args.vocab_size)
    elif args.tokenizer_type == 'HFTokenizer':
        # HuggingFace 分词器：要求 tokenizer_model 非空
        assert args.tokenizer_model is not None, "tokenizer_model 参数不能为空"
        tokenizer = _HFTokenizer(args.tokenizer_model,
                                 args.seq_length,
                                 args.trust_remote_code)
    else:
        # 未实现的分词器类型，抛出异常提示
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))
    
    # -----------------------------------------------------------------------------
    # 对词汇表大小进行填充处理，确保其对齐模型并行大小及更适合 GPU 处理
    # -----------------------------------------------------------------------------
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer

# -----------------------------------------------------------------------------
# 函数：_vocab_size_with_padding
# 作用：对原始 vocab 大小进行向上填充，使其能被指定的 multiple 整除
# -----------------------------------------------------------------------------
def _vocab_size_with_padding(orig_vocab_size, args):
    """
    将词汇表大小填充至能够被模型并行参数整除，并且填充后的大小更适合 GPU 处理。

    参数:
      - orig_vocab_size (int): 原始词汇表大小
      - args: 包含如下字段：
            make_vocab_size_divisible_by (int): 使 vocab 大小能被其整除的数字
            tensor_model_parallel_size (int): 模型并行的张量大小
            rank (int): 当前进程编号（用于日志输出）

    返回:
      after (int): 填充后的词汇表大小

    算法思路:
      - 计算 multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
      - 当当前词汇大小 after 不能被 multiple 整除时，递增 after 直到满足条件
      - 在主进程(rank==0)上打印填充信息
    """
    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    # 迭代增加词汇表大小直到满足对齐条件，注意该过程时间复杂度 O(k)，其中 k 为增加的 token 数
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

# 解释1
# 1. **词表填充的目的：**  
#    为了满足分布式训练或模型并行的要求，代码中通过 `_vocab_size_with_padding` 函数计算了一个补齐后的词表大小（例如原始词表大小为 V，经过填充后变为 V_pad，使得 V_pad 能被特定数字整除）。这保证了模型中嵌入矩阵的行数（和最终投影矩阵的输出维度）是对齐且更易于并行分布的。

# 2. **实际的嵌入与输出矩阵：**  
#    在模型构造阶段，通常会创建两个主要矩阵：
#    - **词嵌入矩阵**：大小为 `[V_pad, hidden_size]`  
#    - **输出投影矩阵（或称 softmax 矩阵）**：大小为 `[hidden_size, V_pad]`  
   
#    这样，当模型对最后一层 hidden states 进行矩阵乘法时，得到的 logits 张量的维度就是 `[batch_size, sequence_length, V_pad]`。

# 3. **训练中的处理：**  
#    - **训练标签：** 实际的训练数据只包含原始词表中的 token（即有效 token 对应的 id 范围通常是 0 到 V-1）。  
#    - **损失计算：** 当使用交叉熵损失时，只有前 V 个位置是真正用于计算损失的；对于那些 dummy token 对应的额外维度（从 V 到 V_pad-1），通常不会出现在训练目标中，因此模型不会因为这些位置而受到损失计算的影响。

# 4. **总结：**  
#    虽然你在词汇表中没有显示地加入额外的 dummy token，但模型内部构造的嵌入矩阵和输出矩阵都是按照填充后的词表大小（V_pad）构建的。因此，最终矩阵乘完得到的 logits 的维度是 **padded vocab size** 的维度。  
   
#    在实际使用时，损失函数和评估过程只关注有效的词表部分（前 V 个 token），而额外的 dummy token 只是用于满足对齐和模型并行的技术要求。

# 解释2
# PyTorch 的 CrossEntropyLoss 并不是直接比较 input 和 target 的“形状是否完全一致”，而是要求：

# Input（模型输出 logits）：
# 一个形状为 (N, C)（或更高维度，但最后一维为类别数）的张量，其中 C 表示类别数。在你的场景中，C 就是 padded vocab size。

# Target（标签）：
# 一个形状为 (N)（或与 input 除了类别维度外其他维度一致）的张量，其中的每个值是一个整数类别索引，其取值范围必须在 [0, C-1] 内。

# 解释3
# 关于 qwen 词表大小：https://github.com/QwenLM/Qwen2.5/issues/147



# =============================================================================
# 抽象基类：AbstractTokenizer
# 作用：定义分词器需要实现的接口，包括属性（vocab、inv_vocab、vocab_size）和方法
# =============================================================================
class AbstractTokenizer(ABC):
    """分词器抽象类，规定了所有分词器必须提供的基本接口。"""

    def __init__(self, name):
        # name: 分词器名称（字符串）
        self.name = name
        super().__init__()

    # -------------------------------------------------------------------------
    # 抽象属性：vocab_size
    # 返回词汇表大小（整数）
    # -------------------------------------------------------------------------
    @property
    @abstractmethod
    def vocab_size(self):
        pass

    # -------------------------------------------------------------------------
    # 抽象属性：vocab
    # 返回词汇表，格式为 {token_str: token_id} 的字典
    # -------------------------------------------------------------------------
    @property
    @abstractmethod
    def vocab(self):
        pass

    # -------------------------------------------------------------------------
    # 抽象属性：inv_vocab
    # 返回反向词汇表，格式为 {token_id: token_str} 的字典
    # -------------------------------------------------------------------------
    @property
    @abstractmethod
    def inv_vocab(self):
        pass

    # -------------------------------------------------------------------------
    # 抽象方法：tokenize
    # 将输入文本（字符串）转换为 token id 列表
    # -------------------------------------------------------------------------
    @abstractmethod
    def tokenize(self, text):
        pass

    # -------------------------------------------------------------------------
    # 方法：detokenize
    # 将 token id 列表转换回文本，默认未实现，需子类重写
    # -------------------------------------------------------------------------
    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    # 以下属性为特殊 token（如 CLS、SEP、PAD、EOD、MASK），默认未实现，子类需根据具体情况提供
    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))

# =============================================================================
# 具体分词器实现：_BertWordPieceTokenizer
# 作用：实现原始 BERT WordPiece 分词器逻辑，并处理额外的特殊 token（如 BOS/EOS 和 T5 特殊 token）
# =============================================================================
class _BertWordPieceTokenizer(AbstractTokenizer):
    """原始 BERT WordPiece 分词器"""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        """
        参数:
          - vocab_file (str): 词汇表文件路径
          - lower_case (bool): 是否将输入文本转换为小写（True 表示小写）
          - vocab_extra_ids (int): 额外特殊 token 数量（例如 T5 的 <extra_id_i>）
        """
        # 根据大小写设置分词器名称
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        # 实例化实际的 FullBertTokenizer 分词器对象
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        # 存储特殊 token 的 id（均为 int），从词汇表中获取
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']
        # 用于记录额外特殊 token 的列表（例如用于 T5 模型的 <extra_id_i>）
        self._additional_special_tokens = []

        # -----------------------------------------------------------------------------
        # 添加 BOS 和 EOS token（句子起始和结束标志）
        # -----------------------------------------------------------------------------
        SPECIAL_TOKENS = {'eos_token': '[EOS]',
                          'bos_token': '[BOS]'}
        self._bos_token = '[BOS]'
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = '[EOS]'
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # -----------------------------------------------------------------------------
        # 添加额外特殊 token：用于 T5 模型中作为 sentinel tokens
        # -----------------------------------------------------------------------------
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)])
        self.add_additional_special_tokens(additional_special_tokens)

    # -------------------------------------------------------------------------
    # 方法：add_token
    # 功能：向词汇表中添加 token（如果不存在）
    # 参数:
    #    token (str)
    # 数据类型：
    #    vocab: dict {token: id}；inv_vocab: dict {id: token}
    # -------------------------------------------------------------------------
    def add_token(self, token):
        if token not in self.vocab:
            # 注意：self.vocab_size 是通过调用属性方法计算，随着新 token 的添加会动态更新
            self.inv_vocab[self.vocab_size] = token
            self.vocab[token] = self.vocab_size

    # -------------------------------------------------------------------------
    # 方法：add_additional_special_tokens
    # 功能：批量添加额外特殊 token，并保存到 _additional_special_tokens 列表中
    # -------------------------------------------------------------------------
    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    # -------------------------------------------------------------------------
    # 属性：vocab_size
    # 返回实际词汇表大小（调用内部 FullBertTokenizer 的 vocab_size 方法）
    # -------------------------------------------------------------------------
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    # -------------------------------------------------------------------------
    # 属性：vocab
    # 返回词汇表字典，格式 {token: id}
    # -------------------------------------------------------------------------
    @property
    def vocab(self):
        return self.tokenizer.vocab

    # -------------------------------------------------------------------------
    # 属性：inv_vocab
    # 返回反向词汇表，格式 {id: token}
    # -------------------------------------------------------------------------
    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    # -------------------------------------------------------------------------
    # 方法：tokenize
    # 功能：将文本分割为 token，并转换为对应的 token id 列表
    # 数据流：输入 text (str) -> 内部 tokenizer.tokenize() 得到 token 列表 ->
    #         转换为 token id 列表
    # -------------------------------------------------------------------------
    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    # -------------------------------------------------------------------------
    # 方法：decode
    # 功能：将 token id 列表转换为可读字符串（对 id 到 token 的映射）
    # -------------------------------------------------------------------------
    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    # -------------------------------------------------------------------------
    # 方法：decode_token_ids
    # 功能：将 token id 列表转换为字符串，排除掉 [PAD] 和 [CLS] 等特殊 token
    # 算法逻辑：
    #    - 遍历 token 列表，遇到以 "##" 开头的 token 表示与前一个 token 连续
    #    - 否则前置空格
    # -------------------------------------------------------------------------
    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    # -------------------------------------------------------------------------
    # 以下属性分别返回各特殊 token 的 id，均为 int 类型
    # -------------------------------------------------------------------------
    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos_token(self):
        """句子开始标记（BOS）的 token 字符串"""
        return self._bos_token

    @property
    def eos_token(self):
        """句子结束标记（EOS）的 token 字符串"""
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """额外特殊 token 列表（list of strings）"""
        return self._additional_special_tokens

    @property
    def bos_token_id(self):
        """BOS token 在词汇表中的 id (int)"""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """EOS token 在词汇表中的 id (int)"""
        return self._eos_token_id

    @property
    def additional_special_tokens_ids(self):
        """额外特殊 token 的 id 列表 (list of int)"""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

# =============================================================================
# 具体分词器实现：_GPT2BPETokenizer
# 作用：实现 GPT2 的 BPE 分词器，通过自定义 GPT2Tokenizer 完成分词与反分词
# =============================================================================
class _GPT2BPETokenizer(AbstractTokenizer):
    """原始 GPT2 BPE 分词器"""

    def __init__(self, vocab_file, merge_file):
        """
        参数:
          - vocab_file (str): GPT2 的词汇表文件路径
          - merge_file (str): GPT2 的 BPE merge 文件路径
        """
        name = 'GPT2 BPE'
        super().__init__(name)
        # 初始化自定义的 GPT2Tokenizer，errors 参数设为 'replace' 确保遇到未知字符时替换
        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=[], max_len=None)
        # 记录文本结束标记（<|endoftext|>）的 id
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    # -------------------------------------------------------------------------
    # 方法：tokenize
    # 功能：将文本转换为 token id 列表
    # -------------------------------------------------------------------------
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    # -------------------------------------------------------------------------
    # 方法：detokenize
    # 功能：将 token id 列表转换回文本
    # -------------------------------------------------------------------------
    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

# =============================================================================
# 具体分词器实现：_SentencePieceTokenizer
# 作用：基于 SentencePieceProcessor 封装分词器，增加了特殊 token 处理和 T5 特殊 token
# =============================================================================
class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron 包装器"""

    def __init__(self, model_file, vocab_extra_ids=0):
        """
        参数:
          - model_file (str): SentencePiece 模型文件路径
          - vocab_extra_ids (int): 额外特殊 token 数量，用于 T5 模型的 <extra_id_i>
        """
        name = 'SentencePieceTokenizer'
        super().__init__(name)
        # 动态导入 sentencepiece 模块
        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    # -------------------------------------------------------------------------
    # 方法：_populate_vocab
    # 功能：遍历 SentencePiece 模型中的所有 token，构造 vocab 与反向 vocab
    # 数据类型:
    #    - self._vocab: dict {token_str: id}
    #    - self._inv_vocab: dict {id: token_str}
    # -------------------------------------------------------------------------
    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    # -------------------------------------------------------------------------
    # 方法：_initalize
    # 功能：初始化特殊 token，包括 <CLS>、<SEP>、<EOD>、<MASK>、<PAD>、<BOS>、<EOS> 以及 T5 特殊 token
    # 算法思路：
    #    - 首先构造基本的 vocab 与 inv_vocab
    #    - 定义内部函数 _add_special_token，用于添加 token 到词汇表中（若不存在）
    #    - 依次添加各个特殊 token，并记录其 id
    # -------------------------------------------------------------------------
    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []  # 用于记录 T5 模型的 <extra_id_i>

        def _add_special_token(t):
            # 如果 token 不在当前 vocab 中，则分配新 id
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            # 记录特殊 token 对应的 id
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        # 添加基本特殊 token
        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        # 添加 PAD token
        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        # 添加 BOS token
        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        # 添加 EOS token
        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        # 添加额外的 T5 特殊 token：<extra_id_0>, <extra_id_1>, ... <extra_id_{vocab_extra_ids-1}>
        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    # -------------------------------------------------------------------------
    # 方法：tokenize
    # 功能：分词，支持将文本中预先定义的特殊 token 单独处理
    # 算法逻辑：
    #    - 使用 while 循环扫描文本，查找所有特殊 token 出现的位置
    #    - 对特殊 token 之前的文本部分调用 sentencepiece 的编码
    #    - 将特殊 token 的 id 插入结果序列中
    #    - 循环处理直到文本全部编码
    # 注意：该逻辑保证特殊 token 不被拆分，并保留其在原文本中的顺序
    # -------------------------------------------------------------------------
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            # 搜索每个特殊 token 在当前子字符串中的位置（返回相对索引）
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            # 若没有找到特殊 token，则退出循环
            if len(indices) == 0:
                break

            # 找出下一个最靠前的特殊 token
            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            # 对特殊 token 前的文本部分进行编码
            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            # 将特殊 token 的 id 添加到结果中
            ids.append(self._special_tokens[next_token])
            # 更新索引，跳过当前特殊 token 的长度
            idx = next_idx + len(next_token)

        # 编码剩余文本
        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # -------------------------------------------------------------------------
    # 方法：detokenize
    # 功能：将 token id 序列转换为文本，遇到特殊 token 时插入其字符串表示
    # 算法逻辑：
    #    - 遍历 id 序列，当遇到特殊 token id 时，先解码前面连续的 id，
    #      然后添加特殊 token 字符串，再继续处理后续 id
    # -------------------------------------------------------------------------
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    # -------------------------------------------------------------------------
    # 以下属性返回特殊 token 的 id，均为 int 类型
    # -------------------------------------------------------------------------
    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]

# =============================================================================
# 具体分词器实现：_GPTSentencePieceTokenizer
# 作用：继承 _SentencePieceTokenizer，针对 GPT 模型进行简化（不提供 CLS/SEP/MASK）
# =============================================================================
class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron 包装器，用于 GPT 模型"""

    def __init__(self, model_file,):
        # 对于 GPT 模型，不需要额外的 vocab extra tokens，因此 vocab_extra_ids=0
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        # 仅构造基础 vocab，不添加额外特殊 token
        self._populate_vocab()
        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        return self.tokenizer.encode_as_ids(text)

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    # 对于 GPT 模型，未定义 CLS、SEP、MASK token，因此返回 -1
    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None

# =============================================================================
# 具体分词器实现：_NullTokenizer
# 作用：空分词器，仅将文本按空格分割并转换为整数序列，常用于测试或调试
# =============================================================================
class _NullTokenizer:
    def __init__(self, vocab_size):
        # 参数 vocab_size (int) 指定初始词汇表大小
        vocab_size = int(vocab_size)
        # 将 vocab_size 作为文本结束 token 的 id
        self._eos_id = vocab_size
        # 实际词汇表大小为传入值加 1（预留 eos token）
        self.vocab_size = vocab_size+1

    def tokenize(self, text):
        # 将文本按空格分割后转换为整数列表
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        # 将整数列表转换为字符串，每个整数之间空格分隔
        text = [str(x) for x in ids]
        return ' '.join(text)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None

# =============================================================================
# 具体分词器实现：_HFTokenizer
# 作用：基于 HuggingFace 的 AutoTokenizer，支持自动加载预训练模型，
#      并补充缺失的特殊 token 配置
# =============================================================================
class _HFTokenizer(AbstractTokenizer):
    """HuggingFace 分词器封装"""

    def __init__(self, tokenizer_name_or_path, max_seq_len, trust_remote_code):
        """
        参数:
          - tokenizer_name_or_path (str): 模型名称或路径，传递给 AutoTokenizer.from_pretrained
          - max_seq_len (int): 模型支持的最大序列长度
          - trust_remote_code (bool): 是否信任远程代码，允许下载和执行模型仓库中的自定义代码
        """
        name = tokenizer_name_or_path
        super().__init__(name)
        # 调用 HF 的 AutoTokenizer 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                                       padding_side="right",
                                                       # 右 padding 和人类习惯对齐
                                                       # pos emb 位置 0 和真实文本位置 0 一致
                                                       # 自回归的时候不用像左 padding 一样还要预测多个 pad
                                                       trust_remote_code=trust_remote_code,
                                                       use_fast=False)
        
        # 设置默认的特殊 token
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        special_tokens_dict = dict()
        # 若模型中没有预定义对应特殊 token，则添加默认 token
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        self.tokenizer.add_special_tokens(special_tokens_dict)
        # 设置模型最大序列长度
        self.tokenizer.model_max_length = max_seq_len
        # 构建 encoder 与 decoder（词汇表及其反向映射），数据类型均为 dict
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.encoder

    @property
    def inv_vocab(self):
        return self.decoder

    # -------------------------------------------------------------------------
    # 方法：tokenize
    # 功能：将输入文本转换为 token id 序列（调用 HF 内部的 encode 方法）
    # -------------------------------------------------------------------------
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    # -------------------------------------------------------------------------
    # 方法：detokenize
    # 功能：将 token id 序列转换回文本（调用 HF 内部的 decode 方法）
    # -------------------------------------------------------------------------
    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    # -------------------------------------------------------------------------
    # 以下属性均返回对应特殊 token 的 id，若 token 不存在则通过 _check_token_candidate
    # 方法抛出异常，提示调用者该特殊 token 未定义。
    # -------------------------------------------------------------------------
    @property
    def bos(self):
        return self.bos_token_id

    @property
    def bos_token_id(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def cls(self):
        candidate = self.tokenizer.cls_token_id
        return self._check_token_candidate(candidate)

    @property
    def sep(self):
        candidate = self.tokenizer.sep_token_id
        return self._check_token_candidate(candidate)

    @property
    def pad(self):
        candidate = self.tokenizer.pad_token_id
        return self._check_token_candidate(candidate)

    @property
    def eod(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def eos(self):
        return self.eos_token_id

    @property
    def eos_token_id(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def mask(self):
        candidate = self.tokenizer.mask_token_id
        return self._check_token_candidate(candidate)

    @property
    def additional_special_tokens_ids(self):
        return self.tokenizer.additional_special_tokens_ids

    # -------------------------------------------------------------------------
    # 静态方法：_check_token_candidate
    # 功能：检查特殊 token id 是否为 None，如果是则抛出异常，保证使用时 token 已定义
    # -------------------------------------------------------------------------
    @staticmethod
    def _check_token_candidate(candidate):
        if candidate is None:
            raise AttributeError("Requested token doesn't exist in current tokenizer")
        return candidate


# 特殊token：
# BOS（Begin of Sentence）： 用于标记序列开始，但在本实现中直接复用了 eos_token_id。
# CLS（Classification）： 用于分类任务，通常放在句首。
# SEP（Separator）： 用于分隔句子或段落。
# PAD（Padding）： 用于填充序列至相同长度，方便 batch 处理。
# EOS/EOD（End Of Sequence/Document）： 标记序列或文档结束，生成任务中至关重要。
# MASK： 用于掩码语言模型任务中对输入 token 的掩盖。