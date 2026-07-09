import json
import re
import warnings
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChatTemplate:
    assistant_header: str | None
    user_header: str | None
    system_prompt: str | None
    end_of_turn_token: str | None
    assistant_loss_prefix: str | None = None


class TemplateRegistry:
    def __init__(self):
        self._templates = {}

    def register(self, name, template):
        assert name not in self._templates, f"Chat template {name} already exists."
        self._templates[name] = template

    def get(self, name):
        return self._templates[name]


TEMPLATE_REGISTRY = TemplateRegistry()

TEMPLATE_REGISTRY.register(
    "qwen",
    ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    "gemma4",
    ChatTemplate(
        assistant_header="<|turn>model\n",
        user_header="<|turn>user\n",
        system_prompt=None,
        end_of_turn_token="<turn|>\n",
        assistant_loss_prefix="<|channel>thought\n<channel|>",
    ),
)

TEMPLATE_REGISTRY.register(
    "ministral3",
    ChatTemplate(
        assistant_header="[/INST]",
        user_header="[INST]",
        system_prompt=None,
        end_of_turn_token="</s>",
    ),
)


class GeneralParser:
    def __init__(self, tokenizer, chat_template):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.system_prompt = chat_template.system_prompt
        self.assistant_loss_prefix = chat_template.assistant_loss_prefix or ""
        self.assistant_message_separator = chat_template.assistant_header or ""
        self.assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"([\s\S]*?(?:"
            + re.escape(chat_template.end_of_turn_token or "")
            + "|$))"
        )

    def parse(
        self,
        conversation,
        max_length,
    ):
        messages = []
        if conversation[0]["role"] == "system":
            warnings.warn(
                "System prompt from the sample overrides the registered template.",
                stacklevel=2,
            )
            messages.append({"role": "system", "content": conversation[0]["content"]})
            conversation = conversation[1:]
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for idx, sentence in enumerate(conversation):
            role = sentence["role"]
            assert idx != 0 or role == "user", (
                f"Conversation must start with user, got {role}."
            )
            tool_calls = sentence.get("tool_calls")
            if isinstance(tool_calls, str):
                try:
                    sentence["tool_calls"] = json.loads(tool_calls)
                except json.JSONDecodeError:
                    assert False, f"Failed to parse tool_calls JSON: {tool_calls}"
            messages.append(sentence)
        render_messages = self._prepare_render_messages(messages)
        conversation_text = render_chat_messages(
            self.tokenizer,
            render_messages,
            add_generation_prompt=False,
        )

        encoding = self.tokenizer(
            conversation_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        matches = list(re.finditer(self.assistant_pattern, conversation_text, re.DOTALL))
        for match in matches:
            content_start_char = match.start(1)
            if self.assistant_loss_prefix and conversation_text.startswith(
                self.assistant_loss_prefix,
                content_start_char,
            ):
                content_start_char += len(self.assistant_loss_prefix)
            content_end_char = match.end(1)
            prefix_ids = self.tokenizer.encode(
                conversation_text[:content_start_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_ids = self.tokenizer.encode(
                conversation_text[:content_end_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            start_token_idx = min(len(prefix_ids), len(input_ids))
            end_token_idx = min(len(full_ids), len(input_ids))
            if start_token_idx < end_token_idx:
                loss_mask[start_token_idx:end_token_idx] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    def _prepare_render_messages(self, messages):
        if not self.assistant_loss_prefix:
            return messages

        render_messages = []
        for message in messages:
            if message["role"] != "assistant":
                render_messages.append(message)
                continue

            content = message["content"]
            assert isinstance(content, str), (
                "Gemma4 non-thinking training expects assistant content to be text."
            )
            render_message = dict(message)
            if not content.startswith(self.assistant_loss_prefix):
                render_message["content"] = f"{self.assistant_loss_prefix}{content}"
            render_messages.append(render_message)
        return render_messages


def render_chat_messages(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    chat_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **chat_kwargs)


def encode_chat_messages(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> torch.LongTensor:
    conversation_text = render_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return tokenizer(
        conversation_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def preprocess_record(
    record,
    tokenizer,
    chat_template,
    max_length,
):
    try:
        template = TEMPLATE_REGISTRY.get(chat_template)
    except KeyError:
        assert False, f"Unknown chat template: {chat_template}"
    parser = GeneralParser(tokenizer=tokenizer, chat_template=template)
    assert "conversations" in record, "Expected `conversations` field for JSONL records."
    return parser.parse(
        record["conversations"],
        max_length=max_length,
    )
