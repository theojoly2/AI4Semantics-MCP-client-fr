from __future__ import annotations


# Standard library imports
from typing import Optional, Any
import asyncio
import logging
from json import loads
from os import environ


# Third-party imports
import streamlit as st


# OpenAI and local application imports
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallParam,
    ChatCompletionUserMessageParam,
)
from openai.resources.chat.completions import AsyncCompletions
from clients import MCPClient
from chat_history import ChatHistory
from .data_model_utils.chat_data_structure import shorten_json


# ----------------------------------------------------------------------
# Config & logging
# ----------------------------------------------------------------------
CONTACT_EMAIL = "emilien.caudron@pwc.com"
LOGGER_NAME = "data_modelling_chat_chatbox"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
def show_user_error(title: str, details: Optional[str] = None) -> None:
    """
    Persist error info so it survives reruns, and show immediate UI feedback.
    """
    st.session_state["ui_error"] = {
        "title": title,
        "details": details or "",
        "contact_email": CONTACT_EMAIL,
    }

    try:
        with st.status(title, expanded=True, state="error") as status:
            if details:
                st.write(details)
            st.write(
                f"**What you can do now:**\n"
                f"1) Review your inputs and correct the bug if possible.\n"
                f"2) Re-launch the UI.\n"
                f"3) If the error keeps happening, contact the tech team at **{CONTACT_EMAIL}**."
            )
            status.update(label="Action required", state="error")
    except Exception:
        st.error(title)
        if details:
            st.write(details)
        st.write(
            f"**What you can do now:**\n"
            f"1) Review your inputs and correct the bug if possible.\n"
            f"2) Re-launch the UI.\n"
            f"3) If the error keeps happening, contact the tech team at **{CONTACT_EMAIL}**."
        )


def with_timeout(coro, seconds: float = 45.0, on_timeout_msg: str = ""):
    """
    Await a coroutine with a timeout; surface a user message on timeout.
    """
    async def _runner():
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            show_user_error(
                "The operation timed out.",
                details=on_timeout_msg or "The server took too long to respond.",
            )
            return None
    return _runner()


def safe_json_loads(text: Optional[str]) -> dict:
    """
    Parse JSON safely, returning {} on failure and logging the error.
    """
    if not text:
        return {}
    try:
        return loads(text)
    except Exception as e:
        logger.exception("JSON parsing failed: %s", e)
        return {}


def _consume_skip_user_echo_flag() -> bool:
    """
    Consume one-shot flag used to avoid rendering the same user message twice.
    """
    skip = bool(st.session_state.get("_suppress_user_echo_once", False))
    if skip:
        st.session_state["_suppress_user_echo_once"] = False
    return skip


def _build_tool_expander_label(parsed_tool: Any, fallback_name: Optional[str] = None) -> str:
    """
    Build a readable collapsed title for tool output blocks.
    """
    tool_name = fallback_name or "tool_call"

    if isinstance(parsed_tool, dict):
        tool_name = parsed_tool.get("tool_name") or fallback_name or "tool_call"

    return f"🛠 {tool_name}"


def _render_tool_output(content: str, fallback_name: Optional[str] = None) -> None:
    """
    Render tool output in a collapsed expander with a readable title.
    """
    parsed = safe_json_loads(content)
    label = _build_tool_expander_label(parsed, fallback_name=fallback_name)

    with st.expander(label, expanded=False):
        if parsed:
            st.json(parsed, expanded=2)
        else:
            st.code(content)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def set_chatbox_layout() -> None:
    """
    Sets up the chat interface layout in Streamlit, including message display and UI fixes.
    Displays the conversation history for user, assistant, and tool messages.
    """
    st.title("Model Bot")

    history = st.session_state.get("history")
    if not history:
        return

    for msg in history.messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user" and content:
            with st.chat_message("user"):
                st.write(content)

        elif role == "assistant" and content:
            with st.chat_message("assistant"):
                st.write(content)

        elif role == "tool" and content:
            _render_tool_output(content)

    st.markdown(
        """
        <style>
            .stChatInput {
                position: fixed;
                bottom: 50px;
                width: 65%;
                z-index: 3;
            }
            .fixed-square {
                position: fixed;
                bottom: 0;
                left: 28%;
                width: 67%;
                height: 100px;
                background-color: white;
                z-index: 2;
            }
            main {
                z-index: 1;
            }
        </style>
        <div class="fixed-square"></div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Chat processing
# ----------------------------------------------------------------------
async def process_user_input(user_input: str | None) -> None:
    """
    Handles user input, generates LLM responses, and processes tool calls in the chat interface.
    Ensures robust error handling and user feedback for all major operations.
    """
    try:
        if user_input is None:
            return

        skip_user_echo = _consume_skip_user_echo_flag()

        if not skip_user_echo:
            with st.chat_message("user"):
                st.write(user_input)

        model = st.session_state.get("model", {})
        model_prompt = ""
        if model and "elements" in model:
            model_prompt = "\n".join([
                "[USER.MODEL]",
                str(shorten_json(model)),
                "[USER.INPUT]",
                "",
            ])
        elif model and "ttl" in model:
            model_prompt = "\n".join([
                "[USER.MODEL]",
                str(model["ttl"]),
                "[USER.INPUT]",
                "",
            ])

        async with st.session_state["mcp_client"] as mcp_client:
            tools = await with_timeout(
                mcp_client.tools(),
                seconds=3000.0,
                on_timeout_msg="listing tools took too long."
            ) or []
            if tools is None:
                return

        user_message = ChatCompletionUserMessageParam(
            role="user",
            content=f"{model_prompt}{user_input}"
        )
        history: ChatHistory = st.session_state["history"]
        history.messages.append(user_message)

        completions: AsyncCompletions = st.session_state["completions"]
        completion_params = st.session_state.get("completion_params", {})

        loop = True
        while loop:
            try:
                response: ChatCompletion = await with_timeout(
                    completions.create(
                        messages=history.messages,
                        tools=tools,
                        tool_choice="auto",
                        model=str(environ["LLM_MODEL"]),
                        temperature=0,
                        stream=False,
                        extra_body=completion_params.get("extra_body"),
                    ),
                    seconds=3000.0,
                    on_timeout_msg="Generating assistant response took too long."
                )
                if response is None:
                    return
            except Exception as e:
                show_user_error(
                    "A critical error occurred while generating the assistant response.",
                    details=str(e)
                )
                with st.chat_message("assistant"):
                    st.write(
                        "⚠️ A critical error occurred while generating the assistant response.\n\n"
                        f"{e}"
                    )
                return

            try:
                message: ChatCompletionMessage = response.choices[0].message
                content = message.content or ""
                if content:
                    with st.chat_message("assistant"):
                        st.write(content)

                if message.tool_calls:
                    tool_calls: list[ChatCompletionMessageToolCallParam] = [
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.id,
                            type="function",
                            function={
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        )
                        for tool_call in message.tool_calls
                    ]

                    history.add_assistant_message(content=content, tool_calls=tool_calls)

                    for tool_call in tool_calls:
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            args = safe_json_loads(function["arguments"])
                            if args is None:
                                args = {}
                            async with st.session_state["mcp_client"] as mcp_client:
                                tool_message = await with_timeout(
                                    mcp_client.call_tool(name, args),
                                    seconds=3000.0,
                                    on_timeout_msg=f"Calling tool '{name}' took too long."
                                )
                                if tool_message is None:
                                    return
                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while calling tool '{name}'.",
                                details=str(e)
                            )
                            with st.chat_message("assistant"):
                                st.write(f"⚠️ Tool error in '{name}': {e}")
                            return

                        try:
                            _render_tool_output(tool_message, fallback_name=name)

                            parsed_tool = safe_json_loads(tool_message)

                            history.add_tool_message(
                                content=tool_message,
                                tool_call_id=tool_call["id"],
                            )

                            if name == "style_guide_check":
                                try:
                                    report = parsed_tool.get("tool_results", {}).get("report", "")
                                    if report:
                                        with st.chat_message("assistant"):
                                            st.write(report)
                                        history.add_assistant_message(report)
                                    else:
                                        with st.chat_message("assistant"):
                                            st.write("Style guide check completed.")
                                except Exception as e:
                                    show_user_error(
                                        "A critical error occurred while displaying the style guide report.",
                                        details=str(e)
                                    )
                                    with st.chat_message("assistant"):
                                        st.write(f"⚠️ Style guide display error: {e}")
                                    return

                                loop = False

                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while processing tool output for '{name}'.",
                                details=str(e)
                            )
                            with st.chat_message("assistant"):
                                st.write(f"⚠️ Tool output processing error for '{name}': {e}")
                            return
                else:
                    history.add_assistant_message(content)
                    loop = False

            except Exception as e:
                show_user_error(
                    "A critical error occurred while processing the assistant message.",
                    details=str(e)
                )
                with st.chat_message("assistant"):
                    st.write(f"⚠️ Assistant message processing error: {e}")
                return

        user_message["content"] = user_input
        try:
            history.save()
        except Exception as e:
            show_user_error("Saving the conversation failed.", details=str(e))
            with st.chat_message("assistant"):
                st.write(f"⚠️ Saving conversation failed: {e}")
        return

    except Exception as e:
        show_user_error("A critical unexpected error occurred in chat processing.", details=str(e))
        with st.chat_message("assistant"):
            st.write(f"⚠️ Unexpected chat processing error: {e}")
        return
