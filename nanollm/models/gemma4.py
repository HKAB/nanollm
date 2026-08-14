import torch
import torch.nn as nn


def compute_rope_params(
    head_dim,
    theta_base=10_000.0,
    context_length=4096,
    rope_type="default",
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    if rope_type == "proportional":
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            theta_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)], dim=0)
        else:
            inv_freq = inv_freq_rotated
    else:
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    positions = torch.arange(context_length, dtype=torch.float32)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)

class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        mean_squared = x_float.pow(2).mean(-1, keepdim=True) + self.eps
        x_norm = x_float * torch.rsqrt(mean_squared)
        if self.with_scale:
            x_norm = x_norm * self.weight.float()
        return x_norm.to(x.dtype)

class Gemma4FeedForward(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        first_kv_shared_layer_idx = cfg["n_layers"] - cfg["num_kv_shared_layers"]
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = cfg["use_double_wide_mlp"] and is_kv_shared_layer
        intermediate_size = cfg["hidden_size"] * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(cfg["emb_dim"], intermediate_size, bias=False, dtype=cfg["dtype"])
        self.up_proj = nn.Linear(cfg["emb_dim"], intermediate_size, bias=False, dtype=cfg["dtype"])
        self.down_proj = nn.Linear(intermediate_size, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = nn.functional.gelu(x_gate, approximate="tanh") * x_up
        x = self.down_proj(x)
        return x

class Gemma4Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_type = cfg["layer_types"][layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = cfg["head_dim"] if self.is_sliding else cfg["global_head_dim"]
        self.num_heads = cfg["n_heads"]
        self.num_kv_heads = cfg["n_kv_heads"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg["emb_dim"], self.head_dim * self.num_heads, bias=False, dtype=cfg["dtype"])
        self.k_proj = nn.Linear(cfg["emb_dim"], self.head_dim * self.num_kv_heads, bias=False, dtype=cfg["dtype"])
        self.v_proj = nn.Linear(cfg["emb_dim"], self.head_dim * self.num_kv_heads, bias=False, dtype=cfg["dtype"])
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=cfg["layer_norm_eps"])
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=cfg["layer_norm_eps"])
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=cfg["layer_norm_eps"])

        first_kv_shared_layer_idx = cfg["n_layers"] - cfg["num_kv_shared_layers"]
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = cfg["layer_types"][:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
        else:
            self.kv_shared_layer_index = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, shared_kv=None):
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        query = apply_rope(query, cos, sin, offset=start_pos)

        if shared_kv is None:
            key = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            key = self.k_norm(key)
            value = self.v_norm(value)
            key = apply_rope(key, cos, sin, offset=start_pos)

            if cache is not None and cache[0] is not None:
                key = torch.cat([cache[0], key], dim=2)
                value = torch.cat([cache[1], value], dim=2)

            next_cache = (key, value)
        else:
            key, value = shared_kv
            next_cache = None

        key_for_attn = repeat_kv(key, self.num_key_value_groups)
        value_for_attn = repeat_kv(value, self.num_key_value_groups)

        attn_scores = query @ key_for_attn.transpose(-1, -2)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(query.dtype)
        context = attn_weights @ value_for_attn
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(context)
        return output, next_cache

class Gemma4DenseBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg["layer_types"][layer_idx]
        self.sliding_window = cfg["sliding_window"]
        self.att = Gemma4Attention(cfg, layer_idx)
        self.mlp = Gemma4FeedForward(cfg, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.post_attention_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.pre_feedforward_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.post_feedforward_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.hidden_size_per_layer_input = cfg["hidden_size_per_layer_input"]
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                cfg["emb_dim"],
                self.hidden_size_per_layer_input,
                bias=False,
                dtype=cfg["dtype"]
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input,
                cfg["emb_dim"],
                bias=False,
                dtype=cfg["dtype"]
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])

    def forward(self, x, per_layer_input, mask_local, mask_global,
                cos_local, sin_local, cos_global, sin_global, start_pos=0, cache=None, shared_kv=None):
        if self.layer_type == "sliding_attention":
            mask = mask_local
            cos, sin = cos_local, sin_local
        else:
            mask = mask_global
            cos, sin = cos_global, sin_global


        if shared_kv is not None:
            eff_kv_len = shared_kv[0].size(2)
        elif cache is not None and cache[0] is not None:
            eff_kv_len = cache[0].size(2) + x.size(1)
        else:
            eff_kv_len = x.size(1)
        mask = mask[..., -eff_kv_len:]

        residual = x
        x = self.input_layernorm(x)
        x_attn, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache, shared_kv=shared_kv)
        if next_cache is not None and self.layer_type == "sliding_attention":
            key, value = next_cache
            if key.size(2) > self.sliding_window:
                key = key[:, :, -self.sliding_window:, :]
                value = value[:, :, -self.sliding_window:, :]
                next_cache = (key, value)

        x_attn = self.post_attention_layernorm(x_attn)
        x = residual + x_attn

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        if self.hidden_size_per_layer_input:
            residual = x
            x_per_layer = self.per_layer_input_gate(per_layer_input)
            x_per_layer = nn.functional.gelu(x_per_layer, approximate="tanh")
            x_per_layer = x_per_layer * per_layer_input
            x_per_layer = self.per_layer_projection(x_per_layer)
            x_per_layer = self.post_per_layer_input_norm(x_per_layer)
            x = residual + x_per_layer

        return x * self.layer_scalar.to(dtype=x.dtype), next_cache

class Gemma4DenseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"], "layer_types must be provided and match n_layers"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
            padding_idx=cfg.get("pad_token_id", 0),
            dtype=cfg["dtype"]
        )
        self.blocks = nn.ModuleList([Gemma4DenseBlock(cfg, i) for i in range(cfg["n_layers"])])
        self.final_norm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        if cfg.get("tie_word_embeddings", False):
            self.out_head.weight = self.tok_emb.weight
        self.current_pos = 0

        self.hidden_size_per_layer_input = cfg["hidden_size_per_layer_input"]
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                cfg["vocab_size_per_layer_input"],
                cfg["n_layers"] * self.hidden_size_per_layer_input,
                padding_idx=cfg.get("pad_token_id", 0),
                dtype=cfg["dtype"]
            )
            self.per_layer_model_projection = nn.Linear(
                cfg["emb_dim"],
                cfg["n_layers"] * self.hidden_size_per_layer_input,
                bias=False,
                dtype=cfg["dtype"]
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                eps=cfg["layer_norm_eps"]
            )

        rope_local_type = cfg.get("rope_local_type", "default")
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            rope_type=rope_local_type,
            dtype=torch.float32
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["global_head_dim"],
            theta_base=cfg["rope_global_base"],
            context_length=cfg["context_length"],
            rope_type=cfg["rope_global_type"],
            partial_rotary_factor=cfg["rope_global_partial_rotary_factor"],
            dtype=torch.float32
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, cur_len, device, pos_start=0, pos_end=None):
        if pos_end is None:
            pos_end = cur_len

        ones = torch.ones((pos_end, pos_end), dtype=torch.bool, device=device)
        mask_global_full = torch.triu(ones, diagonal=1)
        far_past_full = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local_full = mask_global_full | far_past_full

        row_slice = slice(pos_start, pos_end)
        mask_global = mask_global_full[row_slice, :pos_end]
        mask_local = mask_local_full[row_slice, :pos_end]
        return mask_local, mask_global

    def get_per_layer_input(self, input_ids):
        if not self.hidden_size_per_layer_input:
            return None
        return (self.embed_tokens_per_layer(input_ids) * (self.hidden_size_per_layer_input ** 0.5)).reshape(
            *input_ids.shape,
            self.cfg["n_layers"],
            self.hidden_size_per_layer_input
        )

    def project_per_layer_input(self, input_embeds, per_layer_inputs=None):
        if not self.hidden_size_per_layer_input:
            return None
        projected = self.per_layer_model_projection(input_embeds) * (self.cfg["emb_dim"] ** -0.5)
        projected = projected.reshape(
            *input_embeds.shape[:-1],
            self.cfg["n_layers"],
            self.hidden_size_per_layer_input
        )
        projected = self.per_layer_projection_norm(projected)
        if per_layer_inputs is None:
            return projected
        return (projected + per_layer_inputs) * (2.0 ** -0.5)

    def forward(self, input_ids, cache=None):
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        per_layer_inputs = self.get_per_layer_input(input_ids)
        per_layer_inputs = self.project_per_layer_input(x, per_layer_inputs)

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + input_ids.size(1)
            self.current_pos = pos_end
            mask_global, mask_local = self._create_masks(
                cur_len=input_ids.size(1),
                device=input_ids.device,
                pos_start=pos_start,
                pos_end=pos_end
            )
        else:
            pos_start = 0
            mask_global, mask_local = self._create_masks(
                cur_len=input_ids.size(1),
                device=input_ids.device,
                pos_start=0,
                pos_end=input_ids.size(1)
            )

        for i, block in enumerate(self.blocks):
            per_layer_input = per_layer_inputs[..., i, :] if per_layer_inputs is not None else None
            block_cache = cache.get(i) if cache is not None else None
            shared_kv = None
            if cache is not None and block.att.is_kv_shared_layer:
                shared_kv = cache.get(block.att.kv_shared_layer_index)

            x, next_cache = block(
                x,
                per_layer_input,
                mask_local,
                mask_global,
                self.cos_local,
                self.sin_local,
                self.cos_global,
                self.sin_global,
                start_pos=pos_start,
                cache=block_cache,
                shared_kv=shared_kv
            )
            if cache is not None and next_cache is not None:
                cache.update(i, next_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        if self.cfg.get("final_logit_softcap") is not None:
            logits = logits / self.cfg["final_logit_softcap"]
            logits = torch.tanh(logits) * self.cfg["final_logit_softcap"]
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0