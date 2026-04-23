from __future__ import annotations
from typing import (
    Iterator,
    Any,
)
from os import (
    mkdir,
)
from os.path import (
    exists,
)
from json import (
    load,
    dump,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCallParam,
)

PATH_HISTORIES: str = './chat_history/histories'
if not exists(PATH_HISTORIES):
    mkdir(PATH_HISTORIES)

PATH_SYSTEM_PROMPT: str = "./chat_history/system_prompt_v3.txt"
PATH_WELCOME_PROMPT: str = "./chat_history/welcome_prompt.txt"


class ChatHistory:

    def __init__(
        self,
        user: str = '',
        name: str = '',
        messages: list[ChatCompletionMessageParam] = [],
        system_prompt: str = "",
        welcome_prompt: str = "",
    ) -> None:
        self.user: str = user
        self.name: str = name
        if messages:
            self.messages = messages

        else:
            self.messages = []

        if not system_prompt:
            with open(PATH_SYSTEM_PROMPT, "r", encoding="utf-8") as prompt:
                self.add_system_message(prompt.read())

        if not welcome_prompt:
            with open(PATH_WELCOME_PROMPT, "r", encoding="utf-8") as prompt:
                self.add_assistant_message(prompt.read())

    def __repr__(self) -> str:
        return "\n".join(
            [
                str(msg['role']) + ": " + str(msg.get('content', ''))
                for msg
                in self.messages
            ]
        )

    def __iter__(self) -> Iterator[ChatCompletionMessageParam]:
        return iter(self.messages)

    def add_system_message(
        self,
        content: str,
    ) -> None:
        self.messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content=content,
            )
        )
        return

    def __iadd__(
        self,
        other,
    ):
        self.messages.extend(other.messages)
        return self

    def add_user_message(
        self,
        content: str,
    ) -> None:
        self.messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=content,
            )
        )
        return

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[ChatCompletionMessageToolCallParam] = [],
    ) -> None:
        message = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content,
        )
        if tool_calls:
            message["tool_calls"] = tool_calls

        self.messages.append(message)
        return

    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
    ) -> None:
        self.messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
            )
        )
        return

    def save(
        self,
    ) -> None:
        if not self.user:
            return

        dp: str = f'{PATH_HISTORIES}/{self.user}'
        if not exists(dp):
            mkdir(dp)

        if not self.name:
            return

        fp: str = f'{dp}/{self.name}.json'
        with open(fp, 'w') as file:
            dump(
                {
                    'messages': self.messages,
                },
                file
            )

        return

    def load(
        self,
        name: str = '',
    ) -> None:
        if not self.user or not name:
            return

        dp: str = f'{PATH_HISTORIES}/{self.user}'
        fp: str = f'{dp}/{name}.json'
        if not exists(fp):
            return

        self.name = name
        with open(fp, 'r') as file:
            params: dict[str, Any] = load(file)
            self.messages = params['messages']

        return
