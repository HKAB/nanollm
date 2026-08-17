import re

from nanollm.tokenizers.qwen_tokenizer import QwenTokenizer
from nanollm.tokenizers.ling_tokenizer import LingTokenizer

from .ling import Ling3Model, Ling3ModelConfig
from .qwen import Qwen3_5Model, Qwen3_5ModelConfig

MODEL_REGISTRY = {
    "Qwen3_5ForConditionalGeneration": {
        "model_class": Qwen3_5Model,
        "config_class": Qwen3_5ModelConfig,
        "tokenizer_class": QwenTokenizer,
        "config_mapper": "qwen3_5",
        "state_dict_mapper": "qwen3_5",
    },
    "BailingMoeV3ForCausalLM": {
        "model_class": Ling3Model,
        "config_class": Ling3ModelConfig,
        "tokenizer_class": LingTokenizer,
        "config_mapper": "ling3",
        "state_dict_mapper": "ling3",
    },
}

def map_hf_config(hf_config, mapper_type):
    # Some checkpoints nest the language model config under 'text_config'
    config = hf_config.get("text_config", hf_config)

    if mapper_type == "qwen3_5":
        return {
            "vocab_size": config.get("vocab_size", 152064),
            "context_length": config.get("max_position_embeddings", 4096),
            "emb_dim": config.get("hidden_size", 1024),
            "n_heads": config.get("num_attention_heads", 8),
            "n_layers": config.get("num_hidden_layers", 24),
            "hidden_dim": config.get("intermediate_size", 3584),
            "head_dim": config.get("head_dim", 256),
            "qk_norm": config.get("qk_norm", True),
            "n_kv_groups": config.get("num_key_value_heads", 2),
            "rope_base": config.get("rope_theta", 1000000.0) or config.get("rope_parameters", {}).get("rope_theta", 1000000.0),
            "partial_rotary_factor": config.get("partial_rotary_factor", 1.0) or config.get("rope_parameters", {}).get("partial_rotary_factor", 1.0),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
            "layer_types": config.get("layer_types", ["full_attention"] * config.get("num_hidden_layers", 24)),
            "linear_num_value_heads": config.get("linear_num_value_heads", 16),
            "linear_num_key_heads": config.get("linear_num_key_heads", 16),
            "linear_key_head_dim": config.get("linear_key_head_dim", 128),
            "linear_value_head_dim": config.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": config.get("linear_conv_kernel_dim", 4),
            "hidden_act": config.get("hidden_act", "silu"),
            "architectures": hf_config.get("architectures", ["Qwen3_5ForConditionalGeneration"]),
        }
    if mapper_type == "ling3":
        return {
            "vocab_size": config.get("vocab_size", 157184),
            "context_length": config.get("max_position_embeddings", 131072),
            "emb_dim": config.get("hidden_size", 1536),
            "n_heads": config.get("num_attention_heads", 16),
            "n_layers": config.get("num_hidden_layers", 24),
            "hidden_dim": config.get("intermediate_size", 4608),
            "head_dim": config.get("head_dim", 128),
            "n_kv_groups": config.get("num_key_value_heads", 16),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
            "rope_base": config.get("rope_theta", 6_000_000.0),
            "partial_rotary_factor": config.get("partial_rotary_factor", 0.5),
            "rope_interleave": config.get("rope_interleave", True),
            "layer_group_size": config.get("layer_group_size", 4),
            "q_lora_rank": config.get("q_lora_rank", 256),
            "kv_lora_rank": config.get("kv_lora_rank", 512),
            "qk_head_dim": config.get("qk_head_dim", 192),
            "qk_nope_head_dim": config.get("qk_nope_head_dim", 128),
            "qk_rope_head_dim": config.get("qk_rope_head_dim", 64),
            "v_head_dim": config.get("v_head_dim", 128),
            "gated_attention_proj_granularity_type": config.get(
                "gated_attention_proj_granularity_type", "head_wise"
            ),
            "short_conv_kernel_size": config.get("short_conv_kernel_size", 4),
            "no_kda_lora": config.get("no_kda_lora", True),
            "kda_safe_gate": config.get("kda_safe_gate", True),
            "kda_lower_bound": config.get("kda_lower_bound", -5.0),
            "num_experts": config.get("num_experts", 128),
            "num_experts_per_tok": config.get("num_experts_per_tok", 8),
            "num_shared_experts": config.get("num_shared_experts", 1),
            "moe_intermediate_size": config.get("moe_intermediate_size", 512),
            "moe_shared_expert_intermediate_size": config.get(
                "moe_shared_expert_intermediate_size", 512
            ),
            "first_k_dense_replace": config.get("first_k_dense_replace", 1),
            "n_group": config.get("n_group", 8),
            "topk_group": config.get("topk_group", 4),
            "routed_scaling_factor": config.get("routed_scaling_factor", 2.5),
            "hidden_act": config.get("hidden_act", "silu"),
            "pad_token_id": config.get("pad_token_id", 156892),
            "architectures": hf_config.get("architectures", ["BailingMoeV3ForCausalLM"]),
        }
    raise ValueError(f"Unknown config mapper {mapper_type}")

def map_hf_state_dict(hf_state_dict, mapper_type):
    if mapper_type == "qwen3_5":
        def hf_to_nano_key(k):
            # For multimodal (Qwen-VL)
            k = k.replace("model.language_model.embed_tokens.", "transformer.wte.")
            k = k.replace("model.language_model.norm.", "final_norm.")
            k = re.sub(r"model\.language_model\.layers\.(\d+)\.", r"transformer.h.\1.", k)
            # For standard text model
            k = k.replace("model.embed_tokens.", "transformer.wte.")
            k = k.replace("model.norm.", "final_norm.")
            k = re.sub(r"model\.layers\.(\d+)\.", r"transformer.h.\1.", k)
            return k
        return {hf_to_nano_key(k): v for k, v in hf_state_dict.items()}
    if mapper_type == "ling3":
        def hf_to_nano_key(k):
            k = k.replace("model.word_embeddings.", "transformer.wte.")
            k = k.replace("model.norm.", "final_norm.")
            return re.sub(r"model\.layers\.(\d+)\.", r"transformer.h.\1.", k)
        return {hf_to_nano_key(k): v for k, v in hf_state_dict.items()}
    raise ValueError(f"Unknown state dict mapper {mapper_type}")

def get_model_entry(architectures):
    if not architectures:
        raise ValueError("HF config does not specify 'architectures'")
    for arch in architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(f"Architecture {architectures} not supported. Supported: {list(MODEL_REGISTRY.keys())}")
