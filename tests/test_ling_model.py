import re

import pytest
import torch

import nanollm.flash_attention as flash_attention
from nanollm.models.ling import Ling3Model, Ling3ModelConfig
from nanollm.models.registry import get_model_entry, map_hf_config, map_hf_state_dict
from nanollm.tokenizers.ling_tokenizer import LingTokenizer


@pytest.fixture(autouse=True)
def force_cpu_attention_fallback():
    """Keep the tiny CPU tests off CUDA-only FA3 kernels on GPU hosts."""
    saved_override = flash_attention._override_impl
    saved_use_fa3 = flash_attention.USE_FA3
    flash_attention._override_impl = "sdpa"
    flash_attention.USE_FA3 = False
    yield
    flash_attention._override_impl = saved_override
    flash_attention.USE_FA3 = saved_use_fa3


def tiny_config():
    return Ling3ModelConfig(
        vocab_size=64,
        context_length=32,
        emb_dim=32,
        n_heads=2,
        n_layers=4,
        hidden_dim=64,
        head_dim=16,
        n_kv_groups=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_head_dim=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        layer_group_size=4,
        short_conv_kernel_size=3,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        pad_token_id=0,
        architectures=["BailingMoeV3ForCausalLM"],
    )


def make_model():
    torch.manual_seed(0)
    model = Ling3Model(tiny_config())
    model.init_weights()
    return model.eval()


def test_ling_registry_and_official_config_mapping():
    hf_config = {
        "architectures": ["BailingMoeV3ForCausalLM"],
        "vocab_size": 157184,
        "max_position_embeddings": 131072,
        "hidden_size": 1536,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "intermediate_size": 4608,
        "head_dim": 128,
        "num_key_value_heads": 16,
        "layer_group_size": 4,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_head_dim": 192,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    }
    entry = get_model_entry(hf_config["architectures"])
    mapped = map_hf_config(hf_config, entry["config_mapper"])

    assert entry["model_class"] is Ling3Model
    assert mapped["emb_dim"] == 1536
    assert mapped["qk_head_dim"] == 192
    assert mapped["num_experts"] == 128


def test_ling_tokenizer_special_tokens_and_tool_parser():
    class FakeTokenizer:
        tokens = {
            "<|startoftext|>": 10,
            "<|endoftext|>": 11,
            "<|role_end|>": 12,
        }

        def token_to_id(self, token):
            return self.tokens.get(token)

    tokenizer = LingTokenizer(FakeTokenizer())
    assert tokenizer.get_bos_token_id() == 10
    assert tokenizer.get_eos_token_ids() == {11, 12}
    assert tokenizer.parse_tool_call(
        "<tool_call>weather<arg_key>city</arg_key>"
        "<arg_value>Hanoi</arg_value></tool_call>"
    ) == ("weather", {"city": "Hanoi"})


def test_official_state_dict_names_map_to_local_modules():
    model = make_model()
    hf_state = {}
    for key, value in model.state_dict().items():
        hf_key = key.replace("transformer.wte.", "model.word_embeddings.")
        hf_key = hf_key.replace("final_norm.", "model.norm.")
        hf_key = re.sub(r"transformer\.h\.(\d+)\.", r"model.layers.\1.", hf_key)
        hf_state[hf_key] = value

    mapped = map_hf_state_dict(hf_state, "ling3")
    assert set(mapped) == set(model.state_dict())
    model.load_state_dict(mapped, strict=True)


def test_forward_loss_and_selected_logits():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, -1]])

    with torch.no_grad():
        logits = model(tokens)
        selected = model(tokens, logit_positions=torch.tensor([2, 1]))
        loss = model(tokens, targets=targets)

    assert logits.shape == (2, 3, 64)
    expected = logits[torch.arange(2), torch.tensor([2, 1])].unsqueeze(1)
    assert torch.allclose(selected, expected, atol=1e-6, rtol=1e-6)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_cached_decode_matches_full_forward():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4]])
    cache = model.create_kv_cache(
        batch_size=1,
        seq_len=8,
        dtype=model.transformer.wte.weight.dtype,
    )

    with torch.no_grad():
        full = model(tokens)
        model(tokens[:, :3], kv_cache=cache)
        cached_last = model(tokens[:, 3:], kv_cache=cache)

    assert cache.cache_seqlens.tolist() == [4]
    assert torch.allclose(cached_last[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)


def test_packed_forward_matches_independent_sequences():
    model = make_model()
    first = torch.tensor([[1, 2, 3]])
    second = torch.tensor([[4, 5]])
    packed = torch.tensor([[1, 2, 3, 4, 5]])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    with torch.no_grad():
        expected_first = model(first)
        expected_second = model(second)
        actual = model(packed, cu_seqlens=cu_seqlens, position_ids=position_ids)

    assert torch.allclose(actual[:, :3], expected_first, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual[:, 3:], expected_second, atol=1e-5, rtol=1e-5)


def test_loss_free_moe_bias_update():
    model = make_model().train()
    gates = [block.mlp.gate for block in model.transformer.h[1:]]
    gates[0].expert_load.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    imbalance = model.update_moe_expert_bias(update_rate=1e-3)

    assert imbalance == pytest.approx(1.0)
    assert torch.equal(
        gates[0].expert_bias,
        torch.tensor([1e-3, 1e-3, -1e-3, -1e-3]),
    )
    assert all(torch.count_nonzero(gate.expert_load) == 0 for gate in gates)
    assert not any(key.endswith("expert_load") for key in model.state_dict())
