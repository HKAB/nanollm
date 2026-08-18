"""Ling 3.0 (BailingMoE v3) text model.

The implementation follows the public ``inclusionAI/Ling-3.0-tiny``
checkpoint while keeping nanollm's small model/cache API.  CUDA uses the
FlashLinearAttention KDA kernels when they are available; the recurrent
PyTorch path is deliberately kept as a correctness and portability fallback.
"""

from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from nanollm.common import COMPUTE_DTYPE, get_dist_info, print0
from nanollm.flash_attention import flash_attn
from nanollm.optim import DistMuonAdamW, MuonAdamW

try:
    from fla.ops.kda import chunk_kda as _chunk_kda
    from fla.ops.kda import fused_recurrent_kda as _fused_recurrent_kda

    HAS_FLA_KDA = True
except ImportError:
    _chunk_kda = _fused_recurrent_kda = None
    HAS_FLA_KDA = False

try:
    import transformer_engine.pytorch as _te
    from transformer_engine.pytorch.ops import GroupedLinear as _TEGroupedLinear

    HAS_TRANSFORMER_ENGINE = True
    TRANSFORMER_ENGINE_IMPORT_ERROR = None
except Exception as exc:
    _te = _TEGroupedLinear = None
    HAS_TRANSFORMER_ENGINE = False
    TRANSFORMER_ENGINE_IMPORT_ERROR = exc


@dataclass
class Ling3ModelConfig:
    vocab_size: int = 157184
    context_length: int = 131072
    emb_dim: int = 1536
    n_heads: int = 16
    n_layers: int = 24
    hidden_dim: int = 4608
    head_dim: int = 128
    n_kv_groups: int = 16
    rms_norm_eps: float = 1e-6
    rope_base: float = 6_000_000.0
    partial_rotary_factor: float = 0.5
    rope_interleave: bool = True
    layer_group_size: int = 4
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_head_dim: int = 192
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    gated_attention_proj_granularity_type: str | None = "head_wise"
    short_conv_kernel_size: int = 4
    no_kda_lora: bool = True
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    num_experts: int | None = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int | None = 1
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    first_k_dense_replace: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    moe_backend: str = "torch"
    hidden_act: str = "silu"
    pad_token_id: int = 156892
    architectures: list | None = None


class Linear(nn.Linear):
    """Linear layer that follows nanollm's explicit compute-dtype policy."""

    def forward(self, x):
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (x * self.weight.float()).to(input_dtype)


class RMSNormGated(RMSNorm):
    def forward(self, hidden_states, gate):
        return super().forward(hidden_states) * torch.sigmoid(gate.float()).to(hidden_states.dtype)


def compute_rope_params(head_dim, theta_base, context_length, dtype=torch.float32):
    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(context_length, dtype=torch.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_interleaved_rope(q, k, cos, sin):
    """Apply Ling's pair-interleaved RoPE weight layout exactly."""
    for tensor in (q, k):
        if tensor.shape[-1] % 2:
            raise ValueError("rotary head dimension must be even")
    q_shape, k_shape = q.shape, k.shape
    q = q.view(*q_shape[:-1], q_shape[-1] // 2, 2).transpose(-1, -2).reshape(q_shape)
    k = k.view(*k_shape[:-1], k_shape[-1] // 2, 2).transpose(-1, -2).reshape(k_shape)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Ling3Cache:
    """Cache for six MLA layers plus eighteen KDA recurrent states."""

    def __init__(self, config, batch_size, seq_len, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = config.n_layers
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.has_previous_state = False
        self.k_cache = []
        self.v_cache = []
        self.linear_recurrent_states = []
        self.linear_conv_states = []
        conv_width = config.short_conv_kernel_size - 1
        projection_size = config.n_heads * config.head_dim
        for layer_idx in range(config.n_layers):
            if _is_full_attention(config, layer_idx):
                self.k_cache.append(torch.zeros(
                    batch_size, seq_len, config.n_heads, config.qk_head_dim,
                    device=device, dtype=dtype,
                ))
                # Values are padded to qk_head_dim for fast attention kernels.
                self.v_cache.append(torch.zeros(
                    batch_size, seq_len, config.n_heads, config.qk_head_dim,
                    device=device, dtype=dtype,
                ))
                self.linear_recurrent_states.append(None)
                self.linear_conv_states.append(None)
            else:
                self.k_cache.append(None)
                self.v_cache.append(None)
                self.linear_recurrent_states.append(torch.zeros(
                    batch_size, config.n_heads, config.head_dim, config.head_dim,
                    device=device, dtype=torch.float32,
                ))
                self.linear_conv_states.append(tuple(
                    torch.zeros(batch_size, projection_size, conv_width, device=device, dtype=dtype)
                    for _ in range(3)
                ))

    @property
    def device(self):
        return self.cache_seqlens.device

    def get_pos(self):
        first = int(self.cache_seqlens[0].item())
        if not torch.all(self.cache_seqlens == first):
            raise ValueError("ragged cache has no single position")
        return first

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens
        self.has_previous_state = True

    def reset(self):
        self.cache_seqlens.zero_()
        self.has_previous_state = False
        for state in self.linear_recurrent_states:
            if state is not None:
                state.zero_()
        for states in self.linear_conv_states:
            if states is not None:
                for state in states:
                    state.zero_()

    def copy_row_from(self, other, src_row, dst_row):
        length = int(other.cache_seqlens[src_row].item())
        if length > self.max_seq_len:
            raise ValueError(f"cache row of length {length} exceeds {self.max_seq_len}")
        for layer_idx in range(self.n_layers):
            if self.k_cache[layer_idx] is not None:
                self.k_cache[layer_idx][dst_row, :length].copy_(
                    other.k_cache[layer_idx][src_row, :length]
                )
                self.v_cache[layer_idx][dst_row, :length].copy_(
                    other.v_cache[layer_idx][src_row, :length]
                )
            else:
                self.linear_recurrent_states[layer_idx][dst_row].copy_(
                    other.linear_recurrent_states[layer_idx][src_row]
                )
                for dst, src in zip(
                    self.linear_conv_states[layer_idx], other.linear_conv_states[layer_idx]
                ):
                    dst[dst_row].copy_(src[src_row])
        self.cache_seqlens[dst_row] = length
        self.has_previous_state = self.has_previous_state or other.has_previous_state

    def copy_from(self, other):
        if self.batch_size != other.batch_size:
            raise ValueError("cache batch sizes must match")
        for row in range(self.batch_size):
            self.copy_row_from(other, row, row)

    def prefill(self, other):
        if other.batch_size != 1:
            raise ValueError("prefill source must have batch size 1")
        if torch.any(self.cache_seqlens != 0):
            raise ValueError("cannot prefill a non-empty cache")
        for row in range(self.batch_size):
            self.copy_row_from(other, 0, row)


class FeedForward(nn.Module):
    def __init__(self, cfg, intermediate_size):
        super().__init__()
        self.gate_proj = Linear(cfg.emb_dim, intermediate_size, bias=False)
        self.up_proj = Linear(cfg.emb_dim, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, cfg.emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(cfg.num_experts, cfg.emb_dim))
        self.register_buffer("expert_bias", torch.zeros(cfg.num_experts))
        # Runtime routing statistics for Ling's auxiliary-loss-free load
        # balancing. This must not become part of checkpoint compatibility.
        self.register_buffer(
            "expert_load", torch.zeros(cfg.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(flat.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        routing_scores = scores + self.expert_bias.float()
        grouped = routing_scores.view(flat.shape[0], self.n_group, -1)
        group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(dim=-1)
        group_idx = group_scores.topk(self.topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        expert_mask = group_mask[:, :, None].expand_as(grouped).reshape_as(routing_scores)
        topk_idx = routing_scores.masked_fill(~expert_mask, -torch.inf).topk(
            self.top_k, dim=-1, sorted=False
        ).indices
        if self.training:
            # With gradient checkpointing this is accumulated once in the
            # forward and once in the recomputation. The common factor cancels
            # in the relative-load update performed after the optimizer step.
            with torch.no_grad():
                self.expert_load.add_(
                    torch.bincount(
                        topk_idx.reshape(-1), minlength=self.num_experts
                    ).to(self.expert_load.dtype)
                )
        weights = scores.gather(1, topk_idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return topk_idx, weights.to(hidden_states.dtype)


class SparseMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backend = cfg.moe_backend
        self.num_experts = cfg.num_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.intermediate_size = cfg.moe_intermediate_size
        if self.backend == "transformer_engine":
            if not HAS_TRANSFORMER_ENGINE:
                detail = (
                    f" ({TRANSFORMER_ENGINE_IMPORT_ERROR})"
                    if TRANSFORMER_ENGINE_IMPORT_ERROR is not None else ""
                )
                raise RuntimeError(
                    "Transformer Engine MoE backend requested but import failed"
                    f"{detail}. Install it with: uv pip install --no-build-isolation "
                    "'transformer_engine[pytorch]'"
                )
            device = torch.get_default_device()
            self.experts_gate = _TEGroupedLinear(
                num_groups=cfg.num_experts,
                in_features=cfg.emb_dim,
                out_features=cfg.moe_intermediate_size,
                bias=False,
                dtype=COMPUTE_DTYPE,
                device=device,
            )
            self.experts_up = _TEGroupedLinear(
                num_groups=cfg.num_experts,
                in_features=cfg.emb_dim,
                out_features=cfg.moe_intermediate_size,
                bias=False,
                dtype=COMPUTE_DTYPE,
                device=device,
            )
            self.experts_down = _TEGroupedLinear(
                num_groups=cfg.num_experts,
                in_features=cfg.moe_intermediate_size,
                out_features=cfg.emb_dim,
                bias=False,
                dtype=COMPUTE_DTYPE,
                device=device,
            )
        else:
            self.experts = nn.ModuleList(
                FeedForward(cfg, cfg.moe_intermediate_size) for _ in range(cfg.num_experts)
            )
        self.gate = MoEGate(cfg)
        if cfg.num_shared_experts is not None:
            self.shared_experts = FeedForward(
                cfg, cfg.moe_shared_expert_intermediate_size * cfg.num_shared_experts
            )

    def _forward_torch(self, x, topk_idx, topk_weight):
        output = torch.zeros_like(x)
        # index_add keeps this path differentiable and avoids materializing
        # top_k copies of every token at once.
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topk_idx == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(x.index_select(0, token_idx))
            expert_out = expert_out * topk_weight[token_idx, slot_idx, None]
            output.index_add_(0, token_idx, expert_out)
        return output

    def _forward_transformer_engine(self, x, topk_idx, topk_weight):
        flat_experts = topk_idx.reshape(-1)
        tokens_per_expert = torch.bincount(
            flat_experts, minlength=self.num_experts
        ).to(torch.int32)
        permuted, row_id_map = _te.moe_permute(
            x,
            topk_idx.to(torch.int32),
            num_out_tokens=topk_idx.numel(),
            map_type="index",
        )
        gate = self.experts_gate(permuted, tokens_per_expert)
        up = self.experts_up(permuted, tokens_per_expert)
        expert_out = self.experts_down(
            F.silu(gate) * up, tokens_per_expert
        )
        return _te.moe_unpermute(
            expert_out,
            row_id_map,
            merging_probs=topk_weight.float(),
            restore_shape=x.shape,
            map_type="index",
        ).to(x.dtype)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        topk_idx, topk_weight = self.gate(hidden_states)
        if self.backend == "transformer_engine":
            output = self._forward_transformer_engine(x, topk_idx, topk_weight)
        else:
            output = self._forward_torch(x, topk_idx, topk_weight)
        output = output.view(shape)
        if hasattr(self, "shared_experts"):
            output = output + self.shared_experts(hidden_states)
        return output


class MultiLatentAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.num_heads = cfg.n_heads
        if cfg.q_lora_rank is None:
            self.q_proj = Linear(cfg.emb_dim, cfg.n_heads * cfg.qk_head_dim, bias=False)
        else:
            self.q_a_proj = Linear(cfg.emb_dim, cfg.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
            self.q_b_proj = Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = Linear(
            cfg.emb_dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = Linear(
            cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False
        )
        if cfg.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = Linear(cfg.emb_dim, cfg.n_heads, bias=False)
        elif cfg.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = Linear(cfg.emb_dim, cfg.n_heads * cfg.v_head_dim, bias=False)
        else:
            self.g_proj = None
        self.dense = Linear(cfg.n_heads * cfg.v_head_dim, cfg.emb_dim, bias=False)

    def forward(self, x, cos, sin, cache=None, cu_seqlens=None, max_seqlen=None):
        b, t, _ = x.shape
        cfg = self.cfg
        if cfg.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(b, t, cfg.n_heads, cfg.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_rot = torch.split(
            compressed, [cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(b, t, cfg.n_heads, cfg.qk_nope_head_dim + cfg.v_head_dim).transpose(1, 2)
        k_pass, value = torch.split(kv, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)
        k_rot = k_rot.view(b, 1, t, cfg.qk_rope_head_dim)
        if not cfg.rope_interleave:
            raise NotImplementedError("Ling checkpoints currently require interleaved RoPE")
        q_rot, k_rot = apply_interleaved_rope(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(-1, cfg.n_heads, -1, -1)
        q = torch.cat((q_pass, q_rot), dim=-1).transpose(1, 2)
        key = torch.cat((k_pass, k_rot), dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)
        # FA kernels expect a common q/k/v head width. Padding is mathematically
        # neutral and mirrors the official FlashAttention path.
        value_padded = F.pad(value, (0, cfg.qk_head_dim - cfg.v_head_dim))

        if cu_seqlens is not None:
            qf = q.reshape(b * t, cfg.n_heads, cfg.qk_head_dim)
            kf = key.reshape(b * t, cfg.n_heads, cfg.qk_head_dim)
            vf = value_padded.reshape(b * t, cfg.n_heads, cfg.qk_head_dim)
            out = flash_attn.flash_attn_varlen_func(qf, kf, vf, cu_seqlens, max_seqlen)
            out = out.view(b, t, cfg.n_heads, cfg.qk_head_dim)
            if cache is not None:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                kc, vc = cache.get_layer_cache(self.layer_idx)
                rows = torch.repeat_interleave(torch.arange(cache.batch_size, device=x.device), lengths)
                flat = torch.arange(b * t, device=x.device)
                positions = flat - torch.repeat_interleave(cu_seqlens[:-1], lengths)
                kc[rows, positions] = kf
                vc[rows, positions] = vf
        elif cache is None:
            out = flash_attn.flash_attn_func(q, key, value_padded, causal=True)
        else:
            kc, vc = cache.get_layer_cache(self.layer_idx)
            out = flash_attn.flash_attn_with_kvcache(
                q, kc, vc, k=key, v=value_padded,
                cache_seqlens=cache.cache_seqlens, causal=True,
            )
        out = out[..., :cfg.v_head_dim]
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(x).float()).to(out.dtype)
            if cfg.gated_attention_proj_granularity_type == "head_wise":
                out = out * gate[..., None]
            else:
                out = out * gate.view(b, t, cfg.n_heads, cfg.v_head_dim)
        return self.dense(out.reshape(b, t, -1))


class ShortConvolution(nn.Conv1d):
    """Checkpoint-compatible causal depthwise convolution."""

    def __init__(self, hidden_size, kernel_size):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         bias=False, padding=kernel_size - 1)

    def forward(self, x, state=None):
        # x: (B, T, C); state stores the previous raw inputs as (B, C, K-1).
        xt = x.transpose(1, 2)
        width = self.kernel_size[0] - 1
        prefix = state if state is not None else xt.new_zeros(xt.shape[0], xt.shape[1], width)
        combined = torch.cat((prefix, xt), dim=-1)
        y = F.conv1d(combined, self.weight.to(xt.dtype), groups=self.groups)
        if state is not None:
            state.copy_(combined[..., -width:])
        return F.silu(y).transpose(1, 2)


def _torch_kda(q, k, v, raw_gate, beta, A_log, dt_bias, initial_state,
               lower_bound=-5.0, safe_gate=True):
    """Exact recurrent KDA equations; slow but device-independent."""
    input_dtype = q.dtype
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    beta = beta.float()
    h = q.shape[2]
    gate_input = raw_gate.float() + dt_bias.float().view(1, 1, h, -1)
    decay_rate = A_log.float().exp().view(1, 1, h, 1)
    if safe_gate:
        gate = lower_bound * torch.sigmoid(decay_rate * gate_input)
    else:
        gate = -decay_rate * F.softplus(gate_input)
    alpha = gate.exp()
    state = initial_state
    outputs = []
    scale = q.shape[-1] ** -0.5
    for token_idx in range(q.shape[1]):
        qt, kt, vt = q[:, token_idx], k[:, token_idx], v[:, token_idx]
        state = state * alpha[:, token_idx, :, :, None]
        prediction = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (vt - prediction) * beta[:, token_idx, :, None]
        state = state + kt[..., None] * delta[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", qt, state) * scale)
    return torch.stack(outputs, dim=1).to(input_dtype), state


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        size = cfg.n_heads * cfg.head_dim
        self.q_proj = Linear(cfg.emb_dim, size, bias=False)
        self.k_proj = Linear(cfg.emb_dim, size, bias=False)
        self.v_proj = Linear(cfg.emb_dim, size, bias=False)
        self.q_conv1d = ShortConvolution(size, cfg.short_conv_kernel_size)
        self.k_conv1d = ShortConvolution(size, cfg.short_conv_kernel_size)
        self.v_conv1d = ShortConvolution(size, cfg.short_conv_kernel_size)
        self.A_log = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        if cfg.no_kda_lora:
            self.f_proj = Linear(cfg.emb_dim, size, bias=False)
            self.g_proj = Linear(cfg.emb_dim, size, bias=False)
        else:
            self.f_a_proj = Linear(cfg.emb_dim, cfg.head_dim, bias=False)
            self.f_b_proj = Linear(cfg.head_dim, size, bias=False)
            self.g_a_proj = Linear(cfg.emb_dim, cfg.head_dim, bias=False)
            self.g_b_proj = Linear(cfg.head_dim, size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(size, dtype=torch.float32))
        self.b_proj = Linear(cfg.emb_dim, cfg.n_heads, bias=False)
        self.o_norm = RMSNormGated(cfg.head_dim, cfg.rms_norm_eps)
        self.o_proj = Linear(size, cfg.emb_dim, bias=False)
        if layer_idx == 0:
            backend = "FlashLinearAttention KDA" if HAS_FLA_KDA else "pure PyTorch KDA"
            print0(f"Ling linear attention backend: {backend}")

    def _project_gate(self, x, output_gate=False):
        if self.cfg.no_kda_lora:
            return self.g_proj(x) if output_gate else self.f_proj(x)
        if output_gate:
            return self.g_b_proj(self.g_a_proj(x))
        return self.f_b_proj(self.f_a_proj(x))

    def _run_one(self, hidden_states, states, recurrent_state):
        cfg = self.cfg
        q = self.q_conv1d(self.q_proj(hidden_states), states[0] if states else None)
        k = self.k_conv1d(self.k_proj(hidden_states), states[1] if states else None)
        v = self.v_conv1d(self.v_proj(hidden_states), states[2] if states else None)
        shape = (*hidden_states.shape[:2], cfg.n_heads, cfg.head_dim)
        q, k, v = (tensor.view(shape) for tensor in (q, k, v))
        raw_gate = self._project_gate(hidden_states).view(shape)
        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        use_fla = HAS_FLA_KDA and hidden_states.is_cuda
        if use_fla:
            fn = _fused_recurrent_kda if hidden_states.shape[1] <= 64 else _chunk_kda
            out, final_state = fn(
                q=q, k=k, v=v, g=raw_gate, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=cfg.kda_safe_gate, lower_bound=cfg.kda_lower_bound,
            )
        else:
            if recurrent_state is None:
                recurrent_state = torch.zeros(
                    hidden_states.shape[0], cfg.n_heads, cfg.head_dim, cfg.head_dim,
                    device=hidden_states.device, dtype=torch.float32,
                )
            out, final_state = _torch_kda(
                q, k, v, raw_gate, beta, self.A_log, self.dt_bias,
                recurrent_state, cfg.kda_lower_bound, cfg.kda_safe_gate,
            )
        output_gate = self._project_gate(hidden_states, output_gate=True).view(shape)
        out = self.o_norm(out, output_gate).reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return self.o_proj(out), final_state

    def forward(self, hidden_states, cache=None, cu_seqlens=None):
        if cu_seqlens is not None:
            # Segmenting also gives an exact CPU fallback for packed SFT and
            # populates one recurrent/conv state per logical cache row.
            pieces = []
            for row, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
                start, end = int(start.item()), int(end.item())
                states = None if cache is None else tuple(
                    state[row:row + 1] for state in cache.linear_conv_states[self.layer_idx]
                )
                recurrent = None if cache is None else cache.linear_recurrent_states[self.layer_idx][row:row + 1]
                out, final = self._run_one(hidden_states[:, start:end], states, recurrent)
                if cache is not None:
                    cache.linear_recurrent_states[self.layer_idx][row].copy_(final[0])
                pieces.append(out)
            return torch.cat(pieces, dim=1)
        states = None if cache is None else cache.linear_conv_states[self.layer_idx]
        recurrent = None if cache is None else cache.linear_recurrent_states[self.layer_idx]
        out, final = self._run_one(hidden_states, states, recurrent)
        if cache is not None:
            cache.linear_recurrent_states[self.layer_idx].copy_(final)
        return out


def _is_full_attention(cfg, layer_idx):
    return ((layer_idx + 1) % cfg.layer_group_size == 0
            or layer_idx >= cfg.n_layers // cfg.layer_group_size * cfg.layer_group_size)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.is_full_attention = _is_full_attention(cfg, layer_idx)
        self.attention = (MultiLatentAttention(cfg, layer_idx) if self.is_full_attention
                          else KimiDeltaAttention(cfg, layer_idx))
        self.mlp = (SparseMoE(cfg) if cfg.num_experts is not None
                    and layer_idx >= cfg.first_k_dense_replace
                    else FeedForward(cfg, cfg.hidden_dim))
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None, cu_seqlens=None, max_seqlen=None):
        residual = x
        normed = self.input_layernorm(x)
        if self.is_full_attention:
            mixed = self.attention(normed, cos, sin, cache, cu_seqlens, max_seqlen)
        else:
            mixed = self.attention(normed, cache, cu_seqlens)
        x = residual + mixed
        return x + self.mlp(self.post_attention_layernorm(x))


class Ling3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.moe_backend == "auto":
            config.moe_backend = (
                "transformer_engine" if HAS_TRANSFORMER_ENGINE else "torch"
            )
        if config.moe_backend not in {"torch", "transformer_engine"}:
            raise ValueError(
                "moe_backend must be one of: auto, torch, transformer_engine"
            )
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id),
            "h": nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layers)),
        })
        self.final_norm = RMSNorm(config.emb_dim, config.rms_norm_eps)
        self.lm_head = Linear(config.emb_dim, config.vocab_size, bias=False)
        cos, sin = compute_rope_params(
            config.qk_rope_head_dim, config.rope_base, config.context_length, COMPUTE_DTYPE
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.use_gradient_checkpointing = False
        self.logit_softcap = 0.0
        print0(f"Ling MoE backend: {config.moe_backend}")
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True

    @torch.no_grad()
    def prepare_for_checkpoint_load(self):
        """Materialize Transformer Engine runtime state after meta to_empty."""
        if self.config.moe_backend != "transformer_engine":
            return
        for block in self.transformer.h:
            if not isinstance(block.mlp, SparseMoE):
                continue
            for projection in (
                block.mlp.experts_gate,
                block.mlp.experts_up,
                block.mlp.experts_down,
            ):
                projection.reset_parameters()

    @torch.no_grad()
    def _reset_runtime_buffers(self):
        # Checkpoint loading constructs the model on meta and then uses
        # to_empty(), so non-persistent buffers must be populated explicitly.
        cos, sin = compute_rope_params(
            self.config.qk_rope_head_dim,
            self.config.rope_base,
            self.config.context_length,
            self.cos.dtype,
        )
        self.cos.copy_(cos.to(device=self.cos.device))
        self.sin.copy_(sin.to(device=self.sin.device))
        for module in self.modules():
            if isinstance(module, MoEGate):
                module.expert_load.zero_()

    def _map_expert_state_to_backend(self, state_dict):
        if self.config.moe_backend != "transformer_engine":
            return state_dict
        mapped = OrderedDict(state_dict)
        if hasattr(state_dict, "_metadata"):
            mapped._metadata = state_dict._metadata
        for layer_idx, block in enumerate(self.transformer.h):
            if not isinstance(block.mlp, SparseMoE):
                continue
            prefix = f"transformer.h.{layer_idx}.mlp."
            for expert_idx in range(block.mlp.num_experts):
                replacements = {
                    f"{prefix}experts.{expert_idx}.gate_proj.weight":
                        f"{prefix}experts_gate.weight{expert_idx}",
                    f"{prefix}experts.{expert_idx}.up_proj.weight":
                        f"{prefix}experts_up.weight{expert_idx}",
                    f"{prefix}experts.{expert_idx}.down_proj.weight":
                        f"{prefix}experts_down.weight{expert_idx}",
                }
                for source, destination in replacements.items():
                    if source in mapped:
                        mapped[destination] = mapped.pop(source)
        return mapped

    def _map_expert_state_from_backend(self, state_dict):
        if self.config.moe_backend != "transformer_engine":
            return state_dict
        for layer_idx, block in enumerate(self.transformer.h):
            if not isinstance(block.mlp, SparseMoE):
                continue
            prefix = f"transformer.h.{layer_idx}.mlp."
            for expert_idx in range(block.mlp.num_experts):
                replacements = {
                    f"{prefix}experts_gate.weight{expert_idx}":
                        f"{prefix}experts.{expert_idx}.gate_proj.weight",
                    f"{prefix}experts_up.weight{expert_idx}":
                        f"{prefix}experts.{expert_idx}.up_proj.weight",
                    f"{prefix}experts_down.weight{expert_idx}":
                        f"{prefix}experts.{expert_idx}.down_proj.weight",
                }
                for source, destination in replacements.items():
                    if source in state_dict:
                        state_dict[destination] = state_dict.pop(source)
        for key in [key for key in state_dict if key.endswith("_extra_state")]:
            state_dict.pop(key)
        return state_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return self._map_expert_state_from_backend(state_dict)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        backend_state = self._map_expert_state_to_backend(state_dict)
        # TE GroupedLinear runtime metadata is intentionally not checkpointed.
        # Validate every real tensor strictly while allowing only its optional
        # _extra_state entries to be absent.
        result = super().load_state_dict(backend_state, strict=False, assign=assign)
        missing = [key for key in result.missing_keys if not key.endswith("_extra_state")]
        unexpected = [key for key in result.unexpected_keys if not key.endswith("_extra_state")]
        if strict and (missing or unexpected):
            messages = []
            if missing:
                messages.append(f"Missing key(s): {missing}")
            if unexpected:
                messages.append(f"Unexpected key(s): {unexpected}")
            raise RuntimeError(
                f"Error(s) in loading state_dict for {type(self).__name__}:\n"
                + "\n".join(messages)
            )
        self._reset_runtime_buffers()
        return type(result)(missing, unexpected)

    def get_device(self):
        return self.transformer.wte.weight.device

    def supports_packed_prefill(self):
        return True

    def create_kv_cache(self, batch_size, seq_len, dtype):
        return Ling3Cache(
            self.config, batch_size, seq_len, self.get_device(), dtype
        )

    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        lm_head = self.lm_head.weight.numel()
        matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.final_norm.weight.numel()
        return {
            "wte": wte, "lm_head": lm_head, "transformer_matrices": matrices,
            "scalars": scalars, "total": sum(p.numel() for p in self.parameters()),
        }

    def estimate_flops(self, global_batch_size, seq_len):
        # A transparent dense-equivalent estimate; routed experts count only
        # selected and shared expert work.
        c = self.config
        tokens = global_batch_size * seq_len
        full_layers = sum(_is_full_attention(c, i) for i in range(c.n_layers))
        linear_layers = c.n_layers - full_layers
        full_attn = 6 * tokens * full_layers * (
            c.emb_dim * (c.q_lora_rank + c.kv_lora_rank + c.n_heads * c.v_head_dim)
            + seq_len * c.n_heads * c.qk_head_dim
        )
        kda = 6 * tokens * linear_layers * (
            6 * c.emb_dim * c.n_heads * c.head_dim + 4 * c.n_heads * c.head_dim ** 2
        )
        dense_mlp = 6 * tokens * c.emb_dim * 3 * c.hidden_dim
        moe_layers = max(0, c.n_layers - c.first_k_dense_replace)
        moe_mlp = 6 * tokens * moe_layers * c.emb_dim * 3 * (
            c.num_experts_per_tok * c.moe_intermediate_size
            + (c.num_shared_experts or 0) * c.moe_shared_expert_intermediate_size
        )
        vocab = 6 * tokens * c.emb_dim * c.vocab_size
        return full_attn + kda + dense_mlp + moe_mlp + vocab

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2,
                        matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5,
                        use_muon=True):
        model_dim = self.config.emb_dim
        ddp, *_ = get_dist_info()
        embeddings = list(self.transformer.wte.parameters())
        head = list(self.lm_head.parameters())
        others = list(self.transformer.h.parameters()) + list(self.final_norm.parameters())
        matrices = [p for p in others if p.ndim == 2]
        nd_params = [p for p in others if p.ndim > 2]
        scalars = [p for p in others if p.ndim < 2]
        scale = (model_dim / 1024) ** -0.5
        groups = [
            dict(kind="adamw", params=head, lr=unembedding_lr * scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embeddings, lr=embedding_lr * scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=scalars, lr=scalar_lr * scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=nd_params, lr=matrix_lr * scale, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
        ]
        if use_muon:
            # Ling has thousands of identically-shaped expert matrices. Muon
            # stacks every parameter in a group, so a single group would add
            # several GiB of temporary tensors on one GPU. Keep each stack at
            # most 64M elements (~128 MiB in BF16).
            max_group_elements = 64 * 1024 * 1024
            for shape in sorted({p.shape for p in matrices}):
                shape_params = [p for p in matrices if p.shape == shape]
                params_per_group = max(1, max_group_elements // shape.numel())
                for start in range(0, len(shape_params), params_per_group):
                    groups.append(dict(
                        kind="muon",
                        params=shape_params[start:start + params_per_group],
                        lr=matrix_lr,
                        momentum=0.95,
                        ns_steps=5,
                        beta2=0.9,
                        weight_decay=weight_decay,
                    ))
        else:
            groups.append(dict(kind="adamw", params=matrices, lr=matrix_lr * scale,
                               betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay))
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @torch.no_grad()
    def update_moe_expert_bias(self, update_rate=1e-3):
        """Apply Ling's loss-free expert load-balancing update once per step.

        Biased routing scores select experts, while the original sigmoid scores
        continue to weight their outputs. Underloaded experts receive a positive
        bias update and overloaded experts a negative one.
        """
        gates = [
            block.mlp.gate for block in self.transformer.h
            if isinstance(block.mlp, SparseMoE)
        ]
        if not gates:
            return None
        loads = torch.stack([gate.expert_load for gate in gates])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loads, op=dist.ReduceOp.SUM)
        mean_load = loads.mean(dim=-1, keepdim=True)
        has_load = mean_load.squeeze(-1) > 0
        updates = update_rate * (mean_load - loads).sign() * has_load[:, None]
        for gate, update in zip(gates, updates):
            gate.expert_bias.add_(update)
            gate.expert_load.zero_()
        if not has_load.any():
            return None
        relative_error = (loads[has_load] - mean_load[has_load]).abs() / mean_load[has_load]
        return relative_error.max().item()

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, MoEGate):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
            if isinstance(module, KimiDeltaAttention):
                module.A_log.copy_(torch.log(torch.empty_like(module.A_log).uniform_(1, 16)))
                nn.init.zeros_(module.dt_bias)
            if isinstance(module, SparseMoE) and module.backend == "transformer_engine":
                for projection in (
                    module.experts_gate, module.experts_up, module.experts_down
                ):
                    for expert_idx in range(module.num_experts):
                        nn.init.normal_(
                            getattr(projection, f"weight{expert_idx}"),
                            mean=0.0,
                            std=0.02,
                        )
        self._reset_runtime_buffers()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean",
                cu_seqlens=None, position_ids=None, logit_positions=None):
        b, t = idx.shape
        if targets is not None and logit_positions is not None:
            raise ValueError("logit_positions cannot be used when targets are provided")
        if position_ids is None:
            start = 0 if kv_cache is None else kv_cache.get_pos()
            position_ids = torch.arange(start, start + t, device=idx.device).unsqueeze(0)
        cos, sin = self.cos[position_ids], self.sin[position_ids]
        max_seqlen = None
        if cu_seqlens is not None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        for block in self.transformer.h:
            if self.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(
                    block, x, cos, sin, None, cu_seqlens, max_seqlen,
                    use_reentrant=False,
                )
            else:
                x = block(x, cos, sin, kv_cache, cu_seqlens, max_seqlen)
        if kv_cache is not None and cu_seqlens is not None:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
                kv_cache.cache_seqlens.device, kv_cache.cache_seqlens.dtype
            )
            if lengths.shape != kv_cache.cache_seqlens.shape:
                raise ValueError("packed sequence count must match cache batch size")
            kv_cache.cache_seqlens.copy_(lengths)
            kv_cache.has_previous_state = True
        elif kv_cache is not None:
            kv_cache.advance(t)
        x = self.final_norm(x)
        if logit_positions is not None:
            if cu_seqlens is not None and b == 1 and not isinstance(logit_positions, int):
                x = x[0, logit_positions].unsqueeze(1)
            elif isinstance(logit_positions, int):
                x = x[:, logit_positions].unsqueeze(1)
            else:
                if logit_positions.shape != (b,):
                    raise ValueError("logit_positions must have shape (batch_size,)")
                x = x[torch.arange(b, device=x.device), logit_positions].unsqueeze(1)
        logits = self.lm_head(x)
        if targets is not None:
            if self.logit_softcap > 0:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
