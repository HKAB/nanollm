import pytest
from transformers import AutoTokenizer

from nanollm.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def hf_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    except Exception:
        pytest.skip("Could not load Qwen/Qwen3.5-0.8B-Base from HuggingFace.")

@pytest.fixture(scope="module")
def custom_tokenizer():
    try:
        return get_tokenizer("Qwen/Qwen3.5-0.8B-Base", architectures=["Qwen3_5ForConditionalGeneration"])
    except Exception:
        pytest.skip("Could not load custom tokenizer from Qwen/Qwen3.5-0.8B-Base.")

def test_render_conversation_compat(hf_tokenizer, custom_tokenizer):
    # A simple chat conversation
    conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"},
            {"role": "assistant", "content": "I am fine, thank you!"},
            {"role": "user", "content": "Tell me a joke."}
        ]
    }
    
    # Render with HF apply_chat_template (assuming chat template is set correctly, 
    # Qwen Base might not have a default chat template, but Instruct does.
    # So let's provide the default chatml template if missing)
    if not hf_tokenizer.chat_template:
        hf_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        
    hf_rendered_ids = hf_tokenizer.apply_chat_template(
        conversation["messages"],
        tokenize=True,
        add_generation_prompt=False
    )
    
    # Render with our custom tokenizer
    custom_rendered_ids, mask = custom_tokenizer.render_conversation(conversation)
    
    assert hf_rendered_ids == custom_rendered_ids, "Mismatch between HF apply_chat_template and nanollm render_conversation"
