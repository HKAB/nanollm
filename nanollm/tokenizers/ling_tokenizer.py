import copy
import json
import re

from nanollm.tokenizer import Tokenizer


class LingTokenizer(Tokenizer):
    """Tokenizer/chat formatting for Bailing v3 checkpoints."""

    def get_bos_token_id(self):
        return self.token_to_id("<|startoftext|>")

    def get_eos_token_ids(self):
        return {
            token_id for token_id in (
                self.encode_special("<|role_end|>"),
                self.token_to_id("<|endoftext|>"),
            ) if token_id is not None
        }

    def parse_tool_call(self, text):
        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if match is None:
            return None
        body = match.group(1).strip()
        name_match = re.match(r"([^\s<]+)", body)
        if name_match is None:
            return None
        keys = re.findall(r"<arg_key>(.*?)</arg_key>", body, re.DOTALL)
        values = re.findall(r"<arg_value>(.*?)</arg_value>", body, re.DOTALL)
        return name_match.group(1), {
            key.strip(): value.strip() for key, value in zip(keys, values)
        }

    def render_tool_response(self, result):
        return f"\n<tool_response>\n{result}\n</tool_response>"

    def _add(self, ids, mask, value, supervised):
        tokens = [value] if isinstance(value, int) else self.encode(value)
        ids.extend(tokens)
        mask.extend([int(supervised)] * len(tokens))

    def render_conversation(self, conversation, max_tokens=2048, mask_history=False,
                            return_boundaries=False):
        messages = conversation["messages"]
        tools = conversation.get("tools") or []
        thinking = "on" if conversation.get("enable_thinking", True) else "off"
        ids, mask, boundaries = [], [], []
        role_end = self.encode_special("<|role_end|>")

        system = ""
        start = 0
        if messages and messages[0]["role"] == "system":
            system = messages[0].get("content") or ""
            start = 1
        self._add(ids, mask, "<role>SYSTEM</role>", False)
        if system:
            self._add(ids, mask, system + "\n", False)
        if tools:
            self._add(
                ids, mask,
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
                False,
            )
            for tool in tools:
                self._add(ids, mask, "\n" + json.dumps(tool, ensure_ascii=False), False)
            self._add(
                ids, mask,
                "\n</tools>\n\nIf you need to use a function, output its name and arguments as "
                "<tool_call>{function-name}<arg_key>{key}</arg_key>"
                "<arg_value>{value}</arg_value></tool_call>.\n",
                False,
            )
        self._add(ids, mask, f"detailed thinking {thinking}", False)
        self._add(ids, mask, role_end, False)

        last_user = max(
            (i for i, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )
        in_observation = False
        for i, message in enumerate(messages[start:], start=start):
            role = message["role"]
            content = message.get("content") or ""
            supervised = role == "assistant" and (not mask_history or i > last_user)
            if role == "user":
                boundaries.append(len(ids))
                self._add(ids, mask, "<role>HUMAN</role>" + content, False)
                self._add(ids, mask, role_end, False)
            elif role == "system":
                self._add(ids, mask, "<role>SYSTEM</role>" + content, False)
                self._add(ids, mask, role_end, False)
            elif role == "assistant":
                reasoning = message.get("reasoning_content") or ""
                if not reasoning and "</think>" in content:
                    reasoning, content = content.split("</think>", 1)
                    reasoning = reasoning.split("<think>")[-1].strip("\n")
                    content = content.lstrip("\n")
                self._add(ids, mask, "<role>ASSISTANT</role>\n<think>", supervised)
                self._add(ids, mask, reasoning, supervised)
                self._add(ids, mask, "</think>" + content, supervised)
                for tool_call in message.get("tool_calls") or []:
                    fn = tool_call.get("function", tool_call)
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    self._add(ids, mask, "\n<tool_call>" + fn["name"], supervised)
                    for key, value in args.items():
                        if not isinstance(value, str):
                            value = json.dumps(value, ensure_ascii=False)
                        self._add(
                            ids, mask,
                            f"<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>",
                            supervised,
                        )
                    self._add(ids, mask, "\n</tool_call>", supervised)
                self._add(ids, mask, role_end, supervised)
            elif role == "tool":
                if not in_observation:
                    self._add(ids, mask, "<role>OBSERVATION</role>", False)
                    in_observation = True
                self._add(ids, mask, "\n<tool_response>\n" + content + "\n</tool_response>", False)
                next_role = messages[i + 1]["role"] if i + 1 < len(messages) else None
                if next_role != "tool":
                    self._add(ids, mask, role_end, False)
                    in_observation = False
            else:
                raise ValueError(f"Unexpected message role: {role!r}")

        ids, mask = ids[:max_tokens], mask[:max_tokens]
        if return_boundaries:
            return ids, mask, [boundary for boundary in boundaries if boundary < len(ids)]
        return ids, mask

    def render_for_completion(self, conversation, enable_thinking=True, max_tokens=2048):
        conversation = copy.deepcopy(conversation)
        if conversation["messages"] and conversation["messages"][-1]["role"] == "assistant":
            conversation["messages"].pop()
        conversation["enable_thinking"] = enable_thinking
        ids, _ = self.render_conversation(conversation, max_tokens=max_tokens)
        ids.extend(self.encode("<role>ASSISTANT</role>\n<think>" if enable_thinking
                               else "<role>ASSISTANT</role>\n<think></think>"))
        return ids[:max_tokens]
