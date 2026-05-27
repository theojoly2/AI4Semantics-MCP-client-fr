from __future__ import annotations

# Standard library imports
from collections import defaultdict
from typing import Optional, Any, Callable, Awaitable
import asyncio
import logging
from json import loads
from os import environ

# Third-party imports
import streamlit as st

# OpenAI and local application imports
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.resources.chat.completions import AsyncCompletions
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
            st.write(content)


def _build_current_model_prompt(model: Any) -> str:
    if model and "elements" in model:
        return "\n".join([
            "[CURRENT MODEL]",
            str(shorten_json(model)),
            "",
            "[CURRENT USER MESSAGE]",
            "",
        ])

    if model and "ttl" in model:
        return "\n".join([
            "[CURRENT MODEL]",
            str(model["ttl"]),
            "",
            "[CURRENT USER MESSAGE]",
            "",
        ])

    return ""


def _extract_delta_content(delta: Any) -> str:
    content = getattr(delta, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                    elif isinstance(text_value, dict):
                        value = text_value.get("value")
                        if isinstance(value, str):
                            parts.append(value)
                continue

            item_text = getattr(item, "text", None)
            if isinstance(item_text, str):
                parts.append(item_text)
            else:
                value = getattr(item_text, "value", None)
                if isinstance(value, str):
                    parts.append(value)

        return "".join(parts)

    return ""


def _normalize_tool_calls(raw_tool_calls: Any) -> list[ChatCompletionMessageToolCallParam]:
    normalized: list[ChatCompletionMessageToolCallParam] = []

    for tool_call in raw_tool_calls or []:
        if isinstance(tool_call, dict):
            tool_call_id = str(tool_call.get("id") or "")
            function = tool_call.get("function") or {}
            function_name = str(function.get("name") or "")
            function_arguments = str(function.get("arguments") or "{}")
        else:
            tool_call_id = str(getattr(tool_call, "id", "") or "")
            function = getattr(tool_call, "function", None)
            function_name = str(getattr(function, "name", "") or "")
            function_arguments = str(getattr(function, "arguments", "{}") or "{}")

        if not function_name:
            continue

        normalized.append(
            ChatCompletionMessageToolCallParam(
                id=tool_call_id,
                type="function",
                function={
                    "name": function_name,
                    "arguments": function_arguments,
                },
            )
        )

    return normalized


async def _create_completion_streaming(
    completions: AsyncCompletions,
    llm_messages: list[ChatCompletionMessageParam],
    tools: list[Any],
    completion_params: dict[str, Any],
) -> dict[str, Any]:
    stream = await completions.create(
        messages=llm_messages,
        tools=tools,
        tool_choice="auto",
        model=str(environ["LLM_MODEL"]),
        temperature=0,
        stream=True,
        extra_body=completion_params.get("extra_body"),
    )

    assistant_text = ""
    text_placeholder = None
    streamed_any_text = False

    tool_calls_buffer: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
    )

    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        text_piece = _extract_delta_content(delta)
        if text_piece:
            assistant_text += text_piece
            if text_placeholder is None:
                with st.chat_message("assistant"):
                    text_placeholder = st.empty()
            text_placeholder.markdown(assistant_text)
            streamed_any_text = True

        delta_tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in delta_tool_calls:
            idx = getattr(tc, "index", 0) or 0
            entry = tool_calls_buffer[idx]

            tc_id = getattr(tc, "id", None)
            if tc_id:
                entry["id"] = tc_id

            tc_type = getattr(tc, "type", None)
            if tc_type:
                entry["type"] = tc_type

            function = getattr(tc, "function", None)
            if function is not None:
                fname = getattr(function, "name", None)
                fargs = getattr(function, "arguments", None)

                if fname:
                    entry["function"]["name"] += fname
                if fargs:
                    entry["function"]["arguments"] += fargs

    tool_calls: list[ChatCompletionMessageToolCallParam] = []
    for idx in sorted(tool_calls_buffer.keys()):
        entry = tool_calls_buffer[idx]
        if entry["function"]["name"]:
            tool_calls.append(
                ChatCompletionMessageToolCallParam(
                    id=entry["id"] or f"tool_call_{idx}",
                    type="function",
                    function={
                        "name": entry["function"]["name"],
                        "arguments": entry["function"]["arguments"] or "{}",
                    },
                )
            )

    return {
        "content": assistant_text,
        "tool_calls": tool_calls,
        "streamed_any_text": streamed_any_text,
    }

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def set_chatbox_layout() -> None:
    """
    Sets up the chat interface layout in Streamlit, including message display and UI fixes.
    Displays the UI conversation history only (not the LLM summarized memory).
    """
    st.title("Model Bot")

    history = st.session_state.get("history")
    if not history:
        return

    for msg in history.display_messages:
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
async def process_user_input(
    user_input: str | None,
    on_model_mutation: Optional[Callable[[str, Any], Awaitable[None]]] = None,
) -> None:
    """
    Handles user input, generates LLM responses, and processes tool calls in the chat interface.
    Uses two separate histories:
    - display history for what the user sees,
    - LLM history for what is actually sent to the model.
    """
    try:
        if user_input is None:
            return

        history: ChatHistory = st.session_state["history"]

        skip_user_echo = _consume_skip_user_echo_flag()
        if not skip_user_echo:
            with st.chat_message("user"):
                st.write(user_input)

        history.start_new_request(user_input)
        history.add_user_message(user_input)

        model = st.session_state.get("model", {})
        model_prompt = _build_current_model_prompt(model)

        async with st.session_state["mcp_client"] as mcp_client:
            tools = await with_timeout(
                mcp_client.tools(),
                seconds=3000.0,
                on_timeout_msg="listing tools took too long.",
            ) or []
            if tools is None:
                return

        completions: AsyncCompletions = st.session_state["completions"]
        completion_params = st.session_state.get("completion_params", {})

        loop = True
        while loop:
            try:
                llm_messages = history.build_messages_for_llm(
                    current_user_input=user_input,
                    current_model_prompt=model_prompt,
                )

                streamed_response = await with_timeout(
                    _create_completion_streaming(
                        completions=completions,
                        llm_messages=llm_messages,
                        tools=tools,
                        completion_params=completion_params,
                    ),
                    seconds=3000.0,
                    on_timeout_msg="Generating assistant response took too long.",
                )
                if streamed_response is None:
                    return

            except Exception as e:
                show_user_error(
                    "A critical error occurred while generating the assistant response.",
                    details=str(e),
                )
                with st.chat_message("assistant"):
                    st.write(
                        "⚠️ A critical error occurred while generating the assistant response.\n\n"
                        f"{e}"
                    )
                return

            try:
                content = streamed_response.get("content", "") or ""
                tool_calls = _normalize_tool_calls(streamed_response.get("tool_calls", []))
                streamed_any_text = bool(streamed_response.get("streamed_any_text", False))

                if content and not streamed_any_text:
                    with st.chat_message("assistant"):
                        st.write(content)

                if tool_calls:
                    history.add_assistant_message(
                        content=content,
                        tool_calls=tool_calls,
                    )

                    for tool_call in tool_calls:
                        function = tool_call["function"]
                        name = function["name"]
                        raw_arguments_text = function.get("arguments", "{}")

                        try:
                            args = safe_json_loads(raw_arguments_text)
                            if not isinstance(args, dict):
                                args = {}

                            async with st.session_state["mcp_client"] as mcp_client:
                                tool_message = await with_timeout(
                                    mcp_client.call_tool(name, args),
                                    seconds=3000.0,
                                    on_timeout_msg=f"Calling tool '{name}' took too long.",
                                )
                                if tool_message is None:
                                    return

                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while calling tool '{name}'.",
                                details=str(e),
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
                                llm_content=tool_message,
                                tool_name=name,
                                arguments=args,
                                raw_arguments_text=raw_arguments_text,
                            )

                            if on_model_mutation is not None:
                                try:
                                    await on_model_mutation(name, parsed_tool or tool_message)
                                except Exception as e:
                                    logger.exception(
                                        "Model refresh callback failed after tool '%s': %s",
                                        name,
                                        e,
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
                                        details=str(e),
                                    )
                                    with st.chat_message("assistant"):
                                        st.write(f"⚠️ Style guide display error: {e}")
                                    return

                                loop = False
                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while processing tool output for '{name}'.",
                                details=str(e),
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
                    details=str(e),
                )
                with st.chat_message("assistant"):
                    st.write(f"⚠️ Assistant message processing error: {e}")
                return

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
