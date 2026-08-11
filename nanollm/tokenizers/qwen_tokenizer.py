import copy
import json
import re

from nanollm.tokenizer import Tokenizer


class QwenTokenizer(Tokenizer):
    """Tokenizer specifically tuned for Qwen3.5."""

    def get_eos_token_ids(self):
        # Return all possible EOS tokens for Qwen3.5
        im_end_id = self.encode_special("<|im_end|>")
        eos_id = self.token_to_id("<|endoftext|>")
        return {t for t in [im_end_id, eos_id] if t is not None}

    def parse_tool_call(self, text):
        """
        Parse a Qwen3.5 <tool_call>...</tool_call> block.
        Returns (func_name, kwargs_dict) or None on parse failure.
        """
        m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not m:
            return None
        content = m.group(1)
        fn_m = re.search(r'<function=(\w+)', content)
        if not fn_m:
            return None
        func_name = fn_m.group(1)
        kwargs = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', content, re.DOTALL):
            kwargs[pm.group(1)] = pm.group(2).strip()
        return func_name, kwargs

    def render_tool_response(self, result):
        return f"\n<tool_response>\n{result}\n</tool_response>\n"

    def render_conversation(self, conversation, max_tokens=2048, mask_history=False,
                            return_boundaries=False):
        """
        Render a conversation dict into token ids and loss mask.

        mask_history=False (default): all assistant turns are supervised (mask=1).
        mask_history=True: only the last assistant turn (after the last real user query)
            is supervised — useful when only the final response is the training target.

        return_boundaries=True: additionally return a list of token offsets marking the
            start of each user turn. These are the natural cut points for "smart chunking"
            long conversations (split between complete user→assistant exchanges). The
            leading system header (before the first user turn) belongs to the first piece,
            so callers should only cut at boundaries[1:].
        """
        ids, mask = [], []
        boundaries = []  # token offsets at the start of each user turn

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            if token_ids is None:
                return
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        tools = conversation.get("tools") or []

        im_start = self.encode_special("<|im_start|>")
        im_end = self.encode_special("<|im_end|>")
        if im_start is None:
            im_start = self.encode_special("<|user_start|>")
            im_end = self.encode_special("<|user_end|>")

        # Find last_query_index: last user message that is NOT a bare tool_response.
        # Assistant turns after this index receive <think> wrapping.
        last_query_index = 0
        for idx, msg in enumerate(messages):
            if msg["role"] == "user":
                c = msg.get("content") or ""
                if isinstance(c, str):
                    c = c.strip()
                    if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                        last_query_index = idx
                else:
                    last_query_index = idx

        # --- system / tools header ---
        start_idx = 0
        if tools:
            add_tokens(im_start, 0)
            add_tokens(self.encode("system\n"), 0)
            add_tokens(self.encode("# Tools\n\nYou have access to the following functions:\n\n<tools>"), 0)
            for tool in tools:
                add_tokens(self.encode("\n" + json.dumps(tool)), 0)
            add_tokens(self.encode("\n</tools>"), 0)
            tool_instructions = (
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n<function=example_function_name>\n"
                "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
                "<IMPORTANT>\nReminder:\n"
                "- Function calls MUST follow the specified format: an inner <function=...></function> "
                "block must be nested within <tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language BEFORE "
                "the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with your "
                "current knowledge and do not tell the user about function calls\n</IMPORTANT>"
            )
            add_tokens(self.encode(tool_instructions), 0)
            if messages[0]["role"] == "system":
                sys_content = (messages[0].get("content") or "").strip()
                if sys_content:
                    add_tokens(self.encode("\n\n" + sys_content), 0)
                start_idx = 1
            add_tokens(im_end, 0)
            add_tokens(self.encode("\n"), 0)
        elif messages and messages[0]["role"] == "system":
            sys_content = (messages[0].get("content") or "").strip()
            add_tokens(im_start, 0)
            add_tokens(self.encode("system\n"), 0)
            add_tokens(self.encode(sys_content), 0)
            add_tokens(im_end, 0)
            add_tokens(self.encode("\n"), 0)
            start_idx = 1

        # --- message loop ---
        for i, message in enumerate(messages[start_idx:], start=start_idx):
            role = message["role"]
            content = message.get("content") or ""

            is_after_last_query = i > last_query_index
            # When mask_history=True only supervise turns after the last real user query.
            assistant_mask = 1 if (not mask_history or is_after_last_query) else 0

            if role == "system":
                raise ValueError("System message must be at the beginning.")

            elif role == "user":
                boundaries.append(len(ids))  # cut point: start of a user turn
                add_tokens(im_start, 0)
                add_tokens(self.encode("user\n"), 0)
                add_tokens(self.encode(content), 0)
                add_tokens(im_end, 0)
                add_tokens(self.encode("\n"), 0)

            elif role == "assistant":
                # Split out reasoning content if not already a separate field.
                reasoning_content = message.get("reasoning_content") or ""
                if not reasoning_content and "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = parts[0].split("<think>")[-1].strip("\n")
                    content = parts[-1].lstrip("\n")
                reasoning_content = reasoning_content.strip()

                add_tokens(im_start, 0)
                add_tokens(self.encode("assistant\n"), 0)

                if is_after_last_query:
                    add_tokens(self.encode("<think>\n"), assistant_mask)
                    if reasoning_content:
                        add_tokens(self.encode(reasoning_content), assistant_mask)
                    add_tokens(self.encode("\n</think>\n\n"), assistant_mask)

                if content:
                    add_tokens(self.encode(content), assistant_mask)

                # Tool calls embedded in the assistant turn.
                tool_calls = message.get("tool_calls") or []
                for j, tool_call in enumerate(tool_calls):
                    fn = tool_call.get("function", tool_call)
                    if j == 0:
                        prefix = "\n\n<tool_call>\n" if content.strip() else "<tool_call>\n"
                    else:
                        prefix = "\n<tool_call>\n"
                    add_tokens(self.encode(prefix + f"<function={fn['name']}>\n"), assistant_mask)
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    for arg_name, arg_value in args.items():
                        if isinstance(arg_value, (dict, list)):
                            arg_str = json.dumps(arg_value)
                        else:
                            arg_str = str(arg_value)
                        add_tokens(self.encode(f"<parameter={arg_name}>\n{arg_str}\n</parameter>\n"), assistant_mask)
                    add_tokens(self.encode("</function>\n</tool_call>"), assistant_mask)

                add_tokens(im_end, assistant_mask)
                add_tokens(self.encode("\n"), assistant_mask)

            elif role == "tool":
                # Consecutive tool messages are batched inside a single user turn.
                prev_role = messages[start_idx + (i - start_idx) - 1]["role"] if i > start_idx else None
                if prev_role != "tool":
                    add_tokens(im_start, 0)
                    add_tokens(self.encode("user"), 0)
                add_tokens(self.encode("\n<tool_response>\n"), 0)
                add_tokens(self.encode(content), 0)
                add_tokens(self.encode("\n</tool_response>"), 0)
                next_idx = i + 1
                next_role = messages[next_idx]["role"] if next_idx < len(messages) else None
                if next_role != "tool":
                    add_tokens(im_end, 0)
                    add_tokens(self.encode("\n"), 0)

            else:
                raise ValueError(f"Unexpected message role: {role!r}")

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        if return_boundaries:
            boundaries = [b for b in boundaries if b < len(ids)]
            return ids, mask, boundaries
        return ids, mask

    def render_for_completion(self, conversation, enable_thinking=True, max_tokens=2048):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation, max_tokens=max_tokens)

        im_start = self.encode_special("<|im_start|>")
        if im_start is None:
            im_start = self.encode_special("<|user_start|>")
        ids.append(im_start)
        ids.extend(self.encode("assistant\n"))
        if enable_thinking:
            ids.extend(self.encode("<think>\n"))
        else:
            ids.extend(self.encode("<think>\n\n</think>\n\n"))
        return ids
