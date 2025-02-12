# =============================================================================
# 版权声明及模块说明
# -----------------------------------------------------------------------------
# 本代码用于预训练 GPT 模型，基于 Megatron-LM 框架并集成了 DeepSpeed 优化。
# 主要功能包括：模型构造、数据批次生成、损失计算、前向传播、数据后处理以及数据集构造，
# 同时集成了知识蒸馏（KD）和混合专家（MoE）的支持。
#
# 依赖：
#   - torch：PyTorch 框架，用于张量计算和深度学习模型构造
#   - megatron：Megatron-LM 框架，提供模型并行、数据并行以及 Transformer 配置工具
#   - deepspeed：DeepSpeed 库，用于高性能训练优化、内存优化以及零冗余优化
#   - 其他 Python 标准库如 math、os、subprocess 等
#
# 环境要求：
#   - Python 3.6+
#   - 对应版本的 PyTorch、Megatron-LM 和 DeepSpeed（建议参照项目文档）
#
# 设计模式：
#   - 模块化设计：各功能模块（模型构建、数据批处理、损失计算等）分离，便于单独测试与维护
#   - 依赖注入：通过 get_args() 获取全局参数，使各函数对参数依赖更加清晰
#
# 作者：NVIDIA CORPORATION
# 日期：2023
# =============================================================================

"""Pretrain GPT"""

import torch
import math
from functools import partial

# Megatron-LM 框架相关模块，用于获取训练参数、日志打印、计时器、分布式训练以及模型并行等
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group, update_rotary_pos_emb
from megatron.arguments import core_transformer_config_from_args

# DeepSpeed 相关模块
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
from deepspeed.sequence.fpdt_layer import FPDT_InputConstruct

# 系统相关模块
import os
import subprocess

# PyTorch 中的 nn 模块和功能函数
from torch import nn
import torch.nn.functional as F


# =============================================================================
# 模型构造模块
# -----------------------------------------------------------------------------
# 函数：model_provider
# 作用：根据全局参数构建 GPT 模型（支持 pipeline 并行、零冗余优化等）
# 说明：
#   - 读取全局参数 args
#   - 根据参数配置生成 Transformer 配置 config
#   - 判断是否采用 DeepSpeed 零冗余初始化（Zero Stage 3）
#   - 支持 pipeline 并行与普通并行模式的区分
#   - 预计算固定不变的注意力 mask（节省训练中重复计算开销）
#   - 若使用旋转位置嵌入（rotary position embeddings），则提前更新缓存
# -----------------------------------------------------------------------------
def model_provider(pre_process=True, post_process=True):
    """【步骤1】构建 GPT 模型

    参数:
      pre_process (bool): 是否在模型前处理输入（默认 True）
      post_process (bool): 是否在模型后处理输出（默认 True）

    返回:
      model: 构造好的 GPT 模型实例

    变量说明:
      args: 全局训练参数（包含超参数、数据路径、模型配置等），数据类型：Namespace
      config: Transformer 模型配置对象，根据 args 构建，具体字段依赖 Megatron 配置
      dpg: 数据并行组（Data Parallel Group），用于分布式零冗余初始化时跨进程同步
    """
    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    # 【阅读提示】首先获取全局参数和 Transformer 配置
    args = get_args()
    config = core_transformer_config_from_args(args)

    # 判断数据并行组的获取方式（兼容不同版本的 mpu 实现）
    if hasattr(mpu, 'get_sequence_data_parallel_group'):
        dpg = mpu.get_sequence_data_parallel_group()
    elif hasattr(mpu, 'get_data_parallel_group'):
        dpg = mpu.get_data_parallel_group()
    else:
        dpg = None

    # 使用 deepspeed.zero.Init 对模型参数进行零冗余初始化（Zero Stage 3 支持）
    with deepspeed.zero.Init(data_parallel_group=dpg,
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config_dict,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):

        # 如果启用 DeepSpeed 且不禁用 pipeline 并行，则构造 pipeline 版本模型
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                config=config,
                num_tokentypes=0,  # GPT 一般不使用 token type embeddings
                parallel_output=True
            )
            # 【阅读提示】为了在 training.py 中调用 get_batch_pipe，将其赋值到模型内部
            model._megatron_batch_fn = get_batch_pipe

            # 预计算注意力 mask：构造下三角矩阵（tril）确保自回归训练中前向依赖
            # 变量说明：
            #   attention_mask: shape = [1, 1, seq_length, seq_length]，布尔型，True 表示被 mask 掉
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length),
                device=get_accelerator().current_device_name())).view(
                    1, 1, args.seq_length, args.seq_length)

            # 将 mask 转换为二值（mask 掉的部分为 True）
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # 强制转换为布尔类型，保证与模型期望一致
            args.attn_mask = attention_mask.to(torch.bool)

            # 如果使用旋转位置嵌入，提前缓存对应的嵌入（避免训练时频繁通信或计算）
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.seq_length)
        else:
            # 非 pipeline 模式下直接构造普通的 GPT 模型
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


# =============================================================================
# 数据批次生成模块
# -----------------------------------------------------------------------------
# 函数：get_batch
# 作用：从数据迭代器中获取一个 batch，并进行必要的预处理（广播、token 分离、mask 构建）
# 说明：
#   - 输入数据包含 'text' 字段，类型为 torch.int64
#   - 对应生成 tokens、labels（右移一位用于语言建模）、loss_mask、attention_mask 以及 position_ids
#   - 支持 DeepSpeed 的 sequence parallel 与 Megatron 的 sequence parallel 分割策略
#   - 变量说明：
#       tokens: shape = [batch_size, seq_length]，输入 token id 序列
#       labels: shape = [batch_size, seq_length - 1]，预测目标（右移）
#       loss_mask: 与 tokens 同尺寸的 mask，用于标记哪些 token 需要计算 loss
#       attention_mask: 用于自回归注意力机制，shape = [batch_size, 1, seq_length, seq_length]
#       position_ids: 每个 token 的位置编码索引
# -----------------------------------------------------------------------------
def get_batch(data_iterator):
    """【步骤2】生成一个训练批次

    参数:
      data_iterator: 数据迭代器，提供原始数据（字典格式）

    返回:
      tokens: 输入 token 序列张量，类型：torch.int64
      labels: 目标 token 序列张量（右移），类型：torch.int64
      loss_mask: 损失计算时的 mask，类型：torch.float32（后续转为相同设备的数据类型）
      attention_mask: 自回归注意力 mask，类型：torch.bool
      position_ids: 每个 token 的位置索引张量，类型：torch.int64
    """
    args = get_args()
    tokenizer = get_tokenizer()

    # 定义数据字典中需要的键及对应数据类型
    keys = ['text']
    datatype = torch.int64

    # 【阅读提示】若数据迭代器不为空，则调用 next(data_iterator)
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # 将数据广播到所有并行进程，保证数据一致性
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # 解包数据
    # tokens_ 的形状通常为 [batch_size, seq_length]
    tokens_ = data_b['text'].long()
    # labels 为 tokens 的右移版本，即从第二个 token 开始
    labels = tokens_[:, 1:].contiguous()
    # tokens 为去除最后一个 token 的输入
    tokens = tokens_[:, :-1].contiguous()

    # 调用工具函数获取左到右（left-to-right）注意力 mask、损失 mask 和 position_ids
    # 参数解释：
    #   tokenizer.eod: 结束标志（End Of Document）对应的 token id
    #   args.reset_position_ids / args.reset_attention_mask: 是否重置位置和注意力 mask
    #   args.eod_mask_loss: 是否在 EOD token 上 mask 掉 loss
    #   skip_mask: 如果使用 flash attention，不需要额外构造 mask
    skip_mask = args.use_flash_attn or args.use_flash_attn_triton
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        skip_mask)

    # 对于 DeepSpeed 的 sequence parallel：获取对应的并行信息
    seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
    seq_parallel_world_rank = mpu.get_sequence_parallel_rank()

    # 如果使用 DS 的 sequence parallel FPDT 模块，使用专用数据构造类进行 batch 生成
    if args.ds_sequence_parallel_fpdt:
        return FPDT_InputConstruct(tokens, labels, loss_mask, attention_mask, position_ids,
                                   args, seq_parallel_world_size, seq_parallel_world_rank).generate()

    # 对于 Megatron 的 sequence parallel：重新划分 tokens 和 position_ids
    if args.sequence_parallel:
        # 这里获取 tensor model parallel 的大小和当前 rank
        seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
        seq_parallel_world_rank = mpu.get_tensor_model_parallel_rank()
    seq_length = tokens.size(1)

    # 边界检查：确保序列长度能被并行世界大小整除，否则会报错
    assert seq_length % seq_parallel_world_size == 0, \
        f"Sequence length {seq_length} is not divisible by sequence parallel world size {seq_parallel_world_size}"

    # 将序列划分为多个子序列
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    # 切分 tokens 和 position_ids
    tokens = tokens[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]
    # 如果使用 DS 的 sequence parallel，labels 也需要切分
    if mpu.get_sequence_parallel_world_size() > 1:
        labels = labels[:, sub_seq_start:sub_seq_end]

    return tokens, labels, loss_mask, attention_mask, position_ids


# =============================================================================
# 数据后处理模块（用于课程学习策略）
# -----------------------------------------------------------------------------
# 函数：data_post_process
# 作用：根据数据采样器的状态对原始数据进行后处理，主要用于基于序列长度的课程学习
# 说明：
#   - 如果当前难度信息包含 'seqlen_truncate'，则截断数据使得序列长度减少
#   - 如果包含 'seqlen_reshape'，则对数据进行重排，使得 token 数量对齐新的序列长度
#   - 修改后的数据仍存放在 data['text'] 中
# -----------------------------------------------------------------------------
def data_post_process(data, data_sampler_state_dict):
    """【步骤3】数据后处理：课程学习策略

    参数:
      data: 原始数据字典（至少包含 'text' 键），值为张量
      data_sampler_state_dict: 数据采样器状态字典，包含当前难度信息

    返回:
      data: 后处理后的数据字典
    """
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        # 检查是否使用 seqlen_truncate 策略
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            # 若当前序列长度小于预设 seq_length，则截断文本
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen+1)].contiguous()
        # 检查是否使用 seqlen_reshape 策略
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                # 记录原始 token 数量，用于后续计算
                orig_num_token = torch.numel(data['text'])
                # 计算能够整除的新序列长度
                reshape_len = (data['text'].size()[1] // (current_seqlen+1)) * (current_seqlen+1)
                # 重新排列数据，拼接头尾部分
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen+1),
                                          data['text'][:, -(current_seqlen+1):]),
                                         0).contiguous()
                # 计算行数（batch 中的样本数）并调整为偶数（若有要求）
                num_row = math.ceil(orig_num_token / (current_seqlen+1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data


# =============================================================================
# Pipeline 模式下的 batch 生成（与 get_batch 类似，但处理的是单个数据块）
# -----------------------------------------------------------------------------
# 函数：get_batch_pipe
# 作用：为 pipeline 并行修改 get_batch 的输入方式，从单个 data 而非迭代器中取数据
# -----------------------------------------------------------------------------
def get_batch_pipe(data):
    """【步骤4】Pipeline 并行模式下的 batch 生成

    参数:
      data: 单个数据字典（由 pipeline 并行模式传入）

    返回:
      tuple: ((tokens, position_ids, attention_mask), (labels, loss_mask))
    """
    args = get_args()
    tokenizer = get_tokenizer()

    keys = ['text']
    datatype = torch.int64

    # 将数据广播到所有进程（保证数据一致性）
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # 解包数据
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # 获取左到右 mask 和位置索引
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # 如果使用 legacy 版本的课程学习，截断序列到指定长度
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


# =============================================================================
# 损失计算模块
# -----------------------------------------------------------------------------
# 函数：loss_func
# 作用：计算总损失（语言模型损失 + 可能的 MoE 和 KD 损失）
# 说明：
#   - losses: 模型原始损失张量，通常为交叉熵损失
#   - loss_mask: 指定哪些 token 的 loss 被计算（对应 positions），用于 mask 掉 padding 或特殊 token
#   - moe_loss: 来自混合专家模型（MoE）的额外损失
#   - mos_loss: 来自知识蒸馏（KD）或其他辅助任务的损失
#   - 通过 average_losses_across_data_parallel_group 对 loss 进行跨进程平均（便于日志记录）
# -----------------------------------------------------------------------------
def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    """【步骤5】计算训练损失

    参数:
      loss_mask (Tensor): 掩码张量，形状与 tokens 对应，一般为 [batch_size, seq_length]，类型 float
      moe_loss (Tensor or float): MoE 模块的损失项（若无则为 0）
      mos_loss (Tensor or float): 知识蒸馏或其他辅助损失项（若无则为 0）
      output_tensor (Tensor): 模型输出的原始损失，形状 [batch_size, seq_length]，类型 float

    返回:
      loss (Tensor): 标量总损失
      loss_dict (dict): 各损失项字典，便于记录日志
    """
    args = get_args()
    # 将输出损失转换为 float 类型，保证精度统一
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    # 仅对 mask 部分的 loss 求和，然后归一化
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # 跨数据并行组平均损失，用于日志记录
    averaged_loss = average_losses_across_data_parallel_group([loss])

    # 判断是否启用知识蒸馏（KD）或混合专家（MoE）模块
    if args.mos or args.kd:
        # 总损失 = 语言模型损失 + MoE 损失 + KD / MOS 损失
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0],
                          'moe loss': moe_loss, 'mos loss': mos_loss}
        elif args.kd:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0],
                          'moe loss': moe_loss, 'kd loss': mos_loss}
        print_rank_0('>>> total loss: {}, lm loss {}, kd loss {}'.format(loss, averaged_loss[0], mos_loss))
    else:
        # 如果 MoE 专家数目 <= 1，则不计算 moe_loss
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        else:
            loss = loss + moe_loss
            return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}


# -----------------------------------------------------------------------------
# 知识蒸馏损失计算
# -----------------------------------------------------------------------------
# 函数：calculate_mos_loss
# 作用：计算教师模型与学生模型输出之间的 KL 散度损失（KD 损失）
# 说明：
#   - 使用温度缩放（kd_temp）平滑 logits，再计算 KLDivLoss
#   - 对 logits 进行 softmax（教师）和 log_softmax（学生）的处理，保证数值稳定性
#   - 损失会根据序列长度归一化，并乘以超参数 beta 进行缩放
# -----------------------------------------------------------------------------
def calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask):
    """【步骤6】计算知识蒸馏（KD）损失

    参数:
      args: 全局参数对象
      stu_output (Tensor): 学生模型输出 logits，形状 [batch_size, seq_length, vocab_size]
      teacher_model: 教师模型对象（需处于 eval 模式），用于生成教师 logits
      tokens (Tensor): 输入 tokens，形状 [batch_size, seq_length]
      position_ids (Tensor): 位置索引张量，形状 [batch_size, seq_length]
      attention_mask (Tensor): 注意力 mask，形状 [batch_size, 1, seq_length, seq_length]

    返回:
      mos_loss (Tensor): 知识蒸馏损失（标量）
    """
    mos_loss = 0
    alpha = args.kd_alpha_ce    # KD 损失中交叉熵部分的权重（未在下文详细使用）
    beta = args.kd_beta_ce      # KD 损失的缩放系数
    kd_temp = args.kd_temp      # 温度参数，用于平滑 logits

    if teacher_model:
        with torch.no_grad():
            # 如果使用 legacy 的课程学习策略，则截断 tokens、position_ids、attention_mask
            if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                # labels 不需要截断，因为教师模型仅用于生成 logits

            # 教师模型正向传播，得到 logits，tea_other_losses 未使用
            tea_output, tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
            # 断言学生和教师的输出尺寸一致
            assert stu_output.size() == tea_output.size(), \
                'teacher and student output should match in size. Student: {}, Teacher: {}, CL seq length {}'.format(
                    stu_output.size(), tea_output.size(), args.curriculum_seqlen)

        # 温度缩放后计算 logits 的 softmax（教师）和 log_softmax（学生）
        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp, dim=2)
        # 计算 KL 散度损失，注意 nn.KLDivLoss 默认期望输入为对数概率（log-probabilities）
        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)
        # 根据序列长度归一化并乘以 beta
        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss


# =============================================================================
# 前向传播步骤
# -----------------------------------------------------------------------------
# 函数：forward_step
# 作用：模型前向传播的核心步骤，包含：
#       - 批次数据生成
#       - （可选）课程学习下的序列截断处理
#       - 模型正向传播：返回 logits 或直接计算 loss
#       - 计算 MoE 损失以及知识蒸馏损失（若启用）
#       - 返回输出 logits（或 loss）以及基于 partial 封装的 loss 函数
# 说明：
#   - 使用计时器记录 batch 生成时间，便于性能调试
#   - 针对不同训练策略（KD、MoE、课程学习）进行分支处理
# -----------------------------------------------------------------------------
def forward_step(data_iterator, model):
    """【步骤7】模型前向传播步骤

    参数:
      data_iterator: 数据迭代器，用于生成训练 batch
      model: GPT 模型实例

    返回:
      output_tensor: 模型输出的原始 loss 或 logits（取决于是否传入 labels）
      loss_func (function): 用 partial 封装后的损失计算函数
    """
    args = get_args()
    timers = get_timers()

    # 【阅读提示】开始计时，记录 batch 生成耗时
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    # 如果启用数据效率课程学习，记录当前序列长度信息
    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
        if hasattr(args, 'data_efficiency_curriculum_learning_seqlen_type') and \
           args.data_efficiency_curriculum_learning_seqlen_type == 'seqlen_reshape':
            args.data_efficiency_curriculum_learning_numel = torch.numel(tokens)

    # 模型正向传播分两种情况：是否启用 KD 或 MoS 损失
    if args.mos or args.kd:
        # forward 返回 stu_output（未经过交叉熵计算）及其他可能的损失（如 MoE 损失）
        stu_output, other_losses = model(tokens, position_ids, attention_mask)
        # 如果使用 legacy 的课程学习策略，调整 labels 长度
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            assert args.curriculum_seqlen is not None
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        # vocab_parallel_cross_entropy 将 logits 与 labels 计算交叉熵
        output_tensor = tensor_parallel.vocab_parallel_cross_entropy(
            stu_output.contiguous().float(), labels)
    else:
        # 如果直接传入 labels，则模型内部直接计算 loss
        output_tensor, other_losses = model(tokens, position_ids, attention_mask, labels=labels)

    # 同理，对于 legacy 课程学习，调整 loss_mask 长度
    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    # 累计 MoE 损失，遍历 other_losses 列表（可能有多个 MoE 模块的损失）
    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

    mos_loss = 0
    # 如果使用 KD 或 MOS，且 teacher_forward 启用且存在 teacher_model，则计算知识蒸馏损失
    if args.mos or args.kd:
        assert model.training
        if args.teacher_forward and args.teacher_model is not None:
            mos_loss = calculate_mos_loss(args, stu_output,
                                          args.teacher_model[0], tokens, position_ids, attention_mask)

    # 返回 output_tensor 与通过 partial 固定参数后的 loss 函数，便于在 training loop 中调用
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


# =============================================================================
# 数据集构造模块
# -----------------------------------------------------------------------------
# 函数：train_valid_test_datasets_provider
# 作用：根据提供的样本数量构造训练、验证和测试数据集
# 说明：
#   - 调用 build_train_valid_test_datasets 工具函数
#   - 参数中包括数据前缀、数据实现方式、split 字符串、序列长度、随机种子等
# -----------------------------------------------------------------------------
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """【步骤8】构建训练、验证、测试数据集

    参数:
      train_val_test_num_samples (tuple): 各数据集的样本数量（train, valid, test）

    返回:
      train_ds, valid_ds, test_ds: 分别为训练、验证和测试数据集对象
    """
    args = get_args()

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


# =============================================================================
# 系统命令与 Git 信息获取模块
# -----------------------------------------------------------------------------
# 函数：command_exists
# 作用：检查系统中是否存在指定命令（如 git），用于后续获取版本信息
# -----------------------------------------------------------------------------
def command_exists(cmd):
    """【辅助函数】检查系统中是否存在命令 cmd

    参数:
      cmd (str): 要检查的命令名称

    返回:
      bool: 存在返回 True，否则返回 False
    """
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


# -----------------------------------------------------------------------------
# 函数：git_ds_info
# 作用：输出 DeepSpeed 环境报告以及 Megatron 的 Git 信息
# 说明：
#   - 调用 deepspeed.env_report.main 输出环境信息
#   - 尝试通过 git 命令获取当前 commit hash 和分支名称，如失败则标记为 unknown
# -----------------------------------------------------------------------------
def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # 通过系统命令获取 Git 信息
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


# =============================================================================
# 主函数入口：训练启动
# -----------------------------------------------------------------------------
# 当直接运行该脚本时，执行以下步骤：
#   1. 输出 Git 和 DeepSpeed 环境报告
#   2. 调用 pretrain 函数，传入数据集提供者、模型提供者、模型类型、前向传播函数
#      以及额外的参数（如 tokenizer 类型、数据后处理函数）
# 说明：
#   - pretrain 为 Megatron-LM 框架的训练入口函数，内部会处理分布式训练、优化器构造等
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,  # GPT 模型属于 encoder 或 decoder 类型，根据任务需要
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             data_post_process=data_post_process)
