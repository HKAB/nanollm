import re

from nanollm.tokenizers.qwen_tokenizer import QwenTokenizer

from .qwen import Qwen3_5Model, Qwen3_5ModelConfig

MODEL_REGISTRY = {
    "Qwen3_5ForConditionalGeneration": {
        "model_class": Qwen3_5Model,
        "config_class": Qwen3_5ModelConfig,
        "tokenizer_class": QwenTokenizer,
        "config_mapper": "qwen3_5",
        "state_dict_mapper": "qwen3_5",
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
    raise ValueError(f"Unknown state dict mapper {mapper_type}")

def get_model_entry(architectures):
    if not architectures:
        raise ValueError("HF config does not specify 'architectures'")
    for arch in architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(f"Architecture {architectures} not supported. Supported: {list(MODEL_REGISTRY.keys())}")
