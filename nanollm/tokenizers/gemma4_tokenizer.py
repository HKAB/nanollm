from nanollm.tokenizer import Tokenizer


class Gemma4Tokenizer(Tokenizer):
    """Gemma 4 tokenizer for base-model training and completion."""

    def get_bos_token_id(self):
        return self.token_to_id("<bos>")

    def get_eos_token_ids(self):
        return {self.token_to_id("<eos>")}
