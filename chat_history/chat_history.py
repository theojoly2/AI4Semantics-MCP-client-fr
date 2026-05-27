from __future__ import annotations


from copy import deepcopy
from typing import (
    Iterator,
    Any,
    Optional,
)
from os import (
    makedirs,
)
from os.path import (
    exists,
)
from json import (
    load,
    dump,
    loads,
    dumps,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCallParam,
)


PATH_DISPLAY_HISTORIES: str = "./chat_history/display_histories"
PATH_LLM_HISTORIES: str = "./chat_history/llm_histories"
makedirs(PATH_DISPLAY_HISTORIES, exist_ok=True)
makedirs(PATH_LLM_HISTORIES, exist_ok=True)

PATH_SYSTEM_PROMPT: str = "./chat_history/system_prompt_v3.txt"
PATH_WELCOME_PROMPT: str = "./chat_history/welcome_prompt.txt"


class ChatHistory:
    def __init__(
        self,
        user: str = "",
        name: str = "",
        display_messages: Optional[list[ChatCompletionMessageParam]] = None,
        system_messages: Optional[list[ChatCompletionMessageParam]] = None,
        conversation_summary: Optional[list[str]] = None,
        current_request_trace: Optional[list[dict[str, Any]]] = None,
        current_request_llm_messages: Optional[list[ChatCompletionMessageParam]] = None,
        current_request_user_input: str = "",
        system_prompt: str = "",
        welcome_prompt: str = "",
    ) -> None:
        self.user: str = user
        self.name: str = name

        self.display_messages: list[ChatCompletionMessageParam] = (
            list(display_messages) if display_messages else []
        )
        self.system_messages: list[ChatCompletionMessageParam] = (
            list(system_messages) if system_messages else []
        )
        self.conversation_summary: list[str] = (
            list(conversation_summary) if conversation_summary else []
        )
        self.current_request_trace: list[dict[str, Any]] = (
            list(current_request_trace) if current_request_trace else []
        )
        self.current_request_llm_messages: list[ChatCompletionMessageParam] = (
            list(current_request_llm_messages) if current_request_llm_messages else []
        )
        self.current_request_user_input: str = current_request_user_input

        if not self.system_messages:
            if system_prompt:
                self.add_system_message(system_prompt)
            else:
                with open(PATH_SYSTEM_PROMPT, "r", encoding="utf-8") as prompt:
                    self.add_system_message(prompt.read())

        if not self.display_messages:
            if welcome_prompt:
                self.add_assistant_message(
                    welcome_prompt,
                    add_to_llm_request=False,
                    track_trace=False,
                )
            else:
                with open(PATH_WELCOME_PROMPT, "r", encoding="utf-8") as prompt:
                    self.add_assistant_message(
                        prompt.read(),
                        add_to_llm_request=False,
                        track_trace=False,
                    )

    def __repr__(self) -> str:
        return "\n".join(
            [
                str(msg["role"]) + ": " + str(msg.get("content", ""))
                for msg in self.display_messages
            ]
        )

    def __iter__(self) -> Iterator[ChatCompletionMessageParam]:
        return iter(self.display_messages)

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        return self.display_messages

    @messages.setter
    def messages(self, value: list[ChatCompletionMessageParam]) -> None:
        self.display_messages = value

    @staticmethod
    def _truncate(
        value: Any,
        limit: int = 300,
    ) -> str:
        text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _safe_json_loads(text: Optional[str]) -> Any:
        if not text:
            return None
        try:
            return loads(text)
        except Exception:
            return None

    @staticmethod
    def _json_text(
        value: Any,
        limit: int = 1600,
    ) -> str:
        try:
            text = dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _extract_tool_name_from_call(call: ChatCompletionMessageToolCallParam) -> str:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        return str(function.get("name", "tool_call"))

    @staticmethod
    def _extract_tool_arguments_from_call(call: ChatCompletionMessageToolCallParam) -> str:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        return str(function.get("arguments", "{}"))

    def start_new_request(
        self,
        user_input: str,
    ) -> None:
        if self.current_request_user_input or self.current_request_trace:
            self.finalize_current_request_summary()

        self.current_request_user_input = user_input
        self.current_request_trace = []
        self.current_request_llm_messages = []

    def add_system_message(
        self,
        content: str,
    ) -> None:
        self.system_messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content=content,
            )
        )

    def __iadd__(self, other):
        self.display_messages.extend(other.display_messages)
        self.system_messages.extend(other.system_messages)
        self.conversation_summary.extend(other.conversation_summary)
        self.current_request_trace.extend(other.current_request_trace)
        self.current_request_llm_messages.extend(other.current_request_llm_messages)
        return self

    def add_user_message(
        self,
        content: str,
        track_trace: bool = False,
    ) -> None:
        self.display_messages.append(
            ChatCompletionUserMessageParam(role="user", content=content)
        )
        if track_trace:
            self.current_request_trace.append(
                {"type": "user_message", "content": content}
            )

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[list[ChatCompletionMessageToolCallParam]] = None,
        add_to_llm_request: bool = True,
        track_trace: bool = True,
    ) -> None:
        message = ChatCompletionAssistantMessageParam(role="assistant", content=content)
        if tool_calls:
            message["tool_calls"] = tool_calls

        self.display_messages.append(message)

        if add_to_llm_request:
            self.current_request_llm_messages.append(deepcopy(message))

        if not track_trace:
            return

        if tool_calls:
            self.current_request_trace.append(
                {
                    "type": "assistant_tool_calls",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call.get("id"),
                            "name": self._extract_tool_name_from_call(call),
                            "arguments": self._extract_tool_arguments_from_call(call),
                        }
                        for call in tool_calls
                    ],
                }
            )
        else:
            self.current_request_trace.append(
                {"type": "assistant_message", "content": content}
            )

    def summarize_tool_content(
        self,
        content: str,
        tool_name: str = "",
    ) -> str:
        parsed = self._safe_json_loads(content)
        name = tool_name or "retrieve_documents"

        results = parsed
        if isinstance(parsed, dict) and "tool_results" in parsed:
            results = parsed["tool_results"]

        if not isinstance(results, list):
            return f"{name} | preview={self._truncate(content, 500)}"

        filenames: list[str] = []

        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                filename = item[0]
                if isinstance(filename, str) and filename.strip():
                    filenames.append(filename.strip())
            elif isinstance(item, dict):
                filename = item.get("filename")
                if isinstance(filename, str) and filename.strip():
                    filenames.append(filename.strip())

        if not filenames:
            return f"{name} | results.count={len(results)}"

        return f"{name} | documents=[{', '.join(filenames)}]"

    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        llm_content: Optional[str] = None,
        tool_name: str = "",
        arguments: Optional[dict[str, Any]] = None,
        raw_arguments_text: str = "",
        add_to_llm_request: bool = True,
        track_trace: bool = True,
    ) -> None:
        self.display_messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
            )
        )

        if add_to_llm_request:
            # All tool results are kept in full during the current turn so the
            # LLM has complete context for multi-step reasoning (plan execution,
            # validation, completeness checks, etc.).
            # They are discarded at turn boundary via finalize_current_request_summary;
            # only conversation_summary (short traces) survives to the next turn.
            self.current_request_llm_messages.append(
                deepcopy(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=llm_content if llm_content is not None else content,
                        tool_call_id=tool_call_id,
                    )
                )
            )

        if track_trace:
            self.current_request_trace.append(
                {
                    "type": "tool_result",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name or "tool_call",
                    "arguments": deepcopy(arguments) if isinstance(arguments, dict) else {},
                    "raw_arguments_text": raw_arguments_text or self._json_text(arguments or {}),
                    "result_summary": self.summarize_tool_content(content, tool_name=tool_name),
                    "result_preview": self._truncate(content, 1200),
                }
            )

    def _trace_to_lines(self) -> list[str]:
        lines: list[str] = []

        if self.current_request_user_input:
            lines.append(
                "0. User request: "
                + self._truncate(self.current_request_user_input, 500)
            )

        step_index = 1
        for step in self.current_request_trace:
            step_type = step.get("type")

            if step_type == "assistant_message":
                lines.append(
                    f"{step_index}. Assistant message: "
                    f"{self._truncate(step.get('content', ''), 500)}"
                )
                step_index += 1
                continue

            if step_type == "assistant_tool_calls":
                content = self._truncate(step.get("content", ""), 350)
                if content:
                    lines.append(
                        f"{step_index}. Assistant intent before tool call(s): {content}"
                    )
                    step_index += 1
                for call in step.get("tool_calls", []):
                    lines.append(
                        f"{step_index}. Tool call prepared: "
                        f"name={call.get('name', 'tool_call')} | "
                        f"arguments={call.get('arguments', '{}')}"
                    )
                    step_index += 1
                continue

            if step_type == "tool_result":
                lines.append(
                    f"{step_index}. Tool result: "
                    f"name={step.get('tool_name', 'tool_call')} | "
                    f"arguments={step.get('raw_arguments_text', '{}')} | "
                    f"summary={step.get('result_summary', '')}"
                )
                step_index += 1
                continue

            if step_type == "user_message":
                lines.append(
                    f"{step_index}. User message: "
                    f"{self._truncate(step.get('content', ''), 500)}"
                )
                step_index += 1

        return lines

    def finalize_current_request_summary(self) -> None:
        if not self.current_request_user_input and not self.current_request_trace:
            return

        lines = self._trace_to_lines()
        if lines:
            self.conversation_summary.append("\n".join(lines))
            self.conversation_summary = self.conversation_summary[-20:]

        # Discard full tool results — only conversation_summary (short traces)
        # survives to the next turn.
        self.current_request_user_input = ""
        self.current_request_trace = []
        self.current_request_llm_messages = []

    def build_messages_for_llm(
        self,
        current_user_input: str,
        current_model_prompt: str = "",
        max_summary_items: int = 10,
    ) -> list[ChatCompletionMessageParam]:
        llm_messages: list[ChatCompletionMessageParam] = [
            deepcopy(msg) for msg in self.system_messages
        ]

        if self.conversation_summary:
            summary_text = "\n\n".join(self.conversation_summary[-max_summary_items:])
            llm_messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=(
                        "[STEP BY STEP SUMMARY OF PREVIOUS REQUESTS]\n"
                        "Use these exact previous tool arguments if you need to replay or "
                        "continue the same investigation later.\n\n"
                        f"{summary_text}"
                    ),
                )
            )

        user_content = (
            f"{current_model_prompt}{current_user_input}"
            if current_model_prompt
            else current_user_input
        )
        llm_messages.append(
            ChatCompletionUserMessageParam(role="user", content=user_content)
        )

        # All current-turn tool exchanges in full — discarded at turn boundary
        # by finalize_current_request_summary.
        llm_messages.extend(deepcopy(self.current_request_llm_messages))
        return llm_messages

    def save(self) -> None:
        if not self.user or not self.name:
            return

        display_dp = f"{PATH_DISPLAY_HISTORIES}/{self.user}"
        llm_dp = f"{PATH_LLM_HISTORIES}/{self.user}"
        makedirs(display_dp, exist_ok=True)
        makedirs(llm_dp, exist_ok=True)

        with open(f"{display_dp}/{self.name}.json", "w", encoding="utf-8") as file:
            dump(
                {"display_messages": self.display_messages},
                file,
                ensure_ascii=False,
                indent=2,
            )

        with open(f"{llm_dp}/{self.name}.json", "w", encoding="utf-8") as file:
            dump(
                {
                    "system_messages": self.system_messages,
                    "conversation_summary": self.conversation_summary,
                    "current_request_user_input": self.current_request_user_input,
                    "current_request_trace": self.current_request_trace,
                    "current_request_llm_messages": self.current_request_llm_messages,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, name: str = "") -> None:
        if not self.user or not name:
            return

        display_dp = f"{PATH_DISPLAY_HISTORIES}/{self.user}"
        llm_dp = f"{PATH_LLM_HISTORIES}/{self.user}"
        display_fp = f"{display_dp}/{name}.json"
        llm_fp = f"{llm_dp}/{name}.json"

        if not exists(display_fp) and not exists(llm_fp):
            return

        self.name = name

        if exists(display_fp):
            with open(display_fp, "r", encoding="utf-8") as file:
                params: dict[str, Any] = load(file)
                self.display_messages = params.get("display_messages", [])

        if exists(llm_fp):
            with open(llm_fp, "r", encoding="utf-8") as file:
                params = load(file)
                self.system_messages = params.get("system_messages", [])
                self.conversation_summary = params.get("conversation_summary", [])
                self.current_request_user_input = params.get("current_request_user_input", "")
                self.current_request_trace = params.get("current_request_trace", [])
                self.current_request_llm_messages = params.get("current_request_llm_messages", [])

        if not self.system_messages:
            with open(PATH_SYSTEM_PROMPT, "r", encoding="utf-8") as prompt:
                self.add_system_message(prompt.read())
