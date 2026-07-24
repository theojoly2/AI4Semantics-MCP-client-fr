from __future__ import annotations
import time


# Standard library imports
from collections import defaultdict
from typing import Optional, Any, Callable, Awaitable
import asyncio
import logging
from json import loads
from os import environ


# Third-party imports
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


# OpenAI and local application imports
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.resources.chat.completions import AsyncCompletions
from chat_history.chat_history import ChatHistory


# ----------------------------------------------------------------------
# Config & logging
# ----------------------------------------------------------------------
CONTACT_EMAIL = "theo.joly2@developpement-durable.gouv.fr"
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
def _scroll_page_to_bottom() -> None:
    """Scroll the whole page to the bottom immediately after the user sends a message.

    Uses an invisible iframe component so the script is guaranteed to execute even
    after Streamlit reruns.
    """
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
            (function () {
                function scrollParentToBottom() {
                    try {
                        var parentDoc = window.parent.document;
                        var root = parentDoc.documentElement;
                        var body = parentDoc.body;
                        var target = Math.max(
                            body ? body.scrollHeight : 0,
                            root ? root.scrollHeight : 0
                        );
                        window.parent.scrollTo({ top: target, left: 0, behavior: 'auto' });
                        if (root) { root.scrollTop = target; }
                        if (body) { body.scrollTop = target; }
                    } catch (e) {
                        // iframe sandbox may block parent access; fail silently
                    }
                }
                scrollParentToBottom();
                // Retry after a short delay in case Streamlit is still rendering
                setTimeout(scrollParentToBottom, 50);
                setTimeout(scrollParentToBottom, 150);
            })();
        </script>
        """,
        height=0,
        width=0,
    )


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
                f"**Ce que vous pouvez faire maintenant :**\n"
                f"1) Vérifiez vos saisies et corrigez le bug si possible.\n"
                f"2) Relancez l’interface utilisateur.\n"
                f"3) Si l’erreur persiste, contactez l’équipe technique à l’adresse **{CONTACT_EMAIL}**."
            )
            status.update(label="Action required", state="error")
    except Exception:
        st.error(title)
        if details:
            st.write(details)
        st.write(
            f"**Ce que vous pouvez faire maintenant :**\n"
            f"1) Vérifiez vos saisies et corrigez le bug si possible.\n"
            f"2) Relancez l’interface utilisateur.\n"
            f"3) Si l’erreur persiste, contactez l’équipe technique à l’adresse **{CONTACT_EMAIL}**."
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


# ----------------------------------------------------------------------
# Live timeline helpers
# ----------------------------------------------------------------------
def _live_events() -> list[dict[str, Any]]:
    return st.session_state.setdefault("_live_chat_events", [])


def _next_live_event_id() -> int:
    current = int(st.session_state.get("_live_event_seq", 0)) + 1
    st.session_state["_live_event_seq"] = current
    return current


def _clear_live_events() -> None:
    st.session_state["_live_chat_events"] = []
    st.session_state["_live_event_seq"] = 0


def _append_live_event(kind: str, content: str = "", **extra: Any) -> int:
    event_id = _next_live_event_id()
    _live_events().append(
        {
            "id": event_id,
            "kind": kind,
            "content": content,
            **extra,
        }
    )
    return event_id


def _update_live_event(event_id: int, **updates: Any) -> None:
    for event in _live_events():
        if event.get("id") == event_id:
            event.update(updates)
            return


def _has_trailing_thinking_event() -> bool:
    events = _live_events()
    return bool(events and events[-1].get("kind") == "thinking")


def _remove_trailing_thinking_event() -> None:
    events = _live_events()
    if events and events[-1].get("kind") == "thinking":
        events.pop()


def _append_assistant_error_event(message: str) -> None:
    _remove_trailing_thinking_event()
    _append_live_event("assistant", message)


def _render_live_chat_events_in(slot: DeltaGenerator) -> None:
    with slot.container():
        for event in _live_events():
            kind = event.get("kind")
            content = event.get("content", "")

            if kind == "user" and content:
                with st.chat_message("user"):
                    st.write(content)

            elif kind == "assistant":
                with st.chat_message("assistant"):
                    st.write(content)

            elif kind == "tool":
                _render_tool_output(content, fallback_name=event.get("tool_name"))

            elif kind == "thinking":
                with st.chat_message("assistant"):
                    st.markdown(
                        """
                        <div class="chat-thinking-wrap">
                            <span class="chat-thinking-spinner"></span>
                            <span class="chat-thinking-label">Le chatbot réfléchit...</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )




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


def _render_tool_output_in_area(
    content: str,
    fallback_name: Optional[str] = None,
    area: Optional[DeltaGenerator] = None,
) -> None:
    if area is None:
        _render_tool_output(content, fallback_name=fallback_name)
        return

    with area:
        _render_tool_output(content, fallback_name=fallback_name)


async def _create_completion_streaming(
    completions: AsyncCompletions,
    llm_messages: list[ChatCompletionMessageParam],
    tools: list[Any],
    completion_params: dict[str, Any],
    on_assistant_stream_start: Optional[Callable[[], Awaitable[None]]] = None,
    on_assistant_text_update: Optional[Callable[[str], Awaitable[None]]] = None,
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
    pending_text = ""
    streamed_any_text = False
    assistant_stream_started = False
    last_flush = 0.0
    flush_interval = 0.03  # 30 ms

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
            if not assistant_stream_started:
                assistant_stream_started = True
                if on_assistant_stream_start is not None:
                    try:
                        await on_assistant_stream_start()
                    except Exception as e:
                        logger.exception("on_assistant_stream_start callback failed: %s", e)

            pending_text += text_piece

            now = time.perf_counter()
            should_flush = (
                (now - last_flush) >= flush_interval
                or text_piece.endswith((" ", "\n", ".", ",", ":", ";", "!", "?"))
                or len(pending_text) >= 40
            )

            if should_flush:
                assistant_text += pending_text
                pending_text = ""

                if on_assistant_text_update is not None:
                    try:
                        await on_assistant_text_update(assistant_text)
                    except Exception as e:
                        logger.exception("on_assistant_text_update callback failed: %s", e)

                streamed_any_text = True
                last_flush = now

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

    if pending_text:
        assistant_text += pending_text
        pending_text = ""
        if on_assistant_text_update is not None:
            try:
                await on_assistant_text_update(assistant_text)
            except Exception as e:
                logger.exception("on_assistant_text_update callback failed: %s", e)
        streamed_any_text = True

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
    Layout setup for GlossaryAI chat interface, displaying the conversation history.
    """
    history = st.session_state.get("history")
    if history:
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
                bottom: 0.5rem;
                width: 65%;
                z-index: 3;
            }
            .stChatFloatingInputContainer {
                position: fixed !important;
                bottom: 0.5rem !important;
                left: 28% !important;
                width: 67% !important;
                z-index: 3 !important;
            }
            main {
                z-index: 1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Chat processing
# ----------------------------------------------------------------------
async def process_user_input(
    user_input: str | None,
    mcp_client: MCPClient | None = None,
    on_thinking_start: Optional[Callable[[], Awaitable[None]]] = None,
    on_assistant_stream_start: Optional[Callable[[], Awaitable[None]]] = None,
    on_assistant_stream_end: Optional[Callable[[bool], Awaitable[None]]] = None,
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

        async def _call_optional(callback, *args):
            if callback is None:
                return
            try:
                await callback(*args)
            except Exception as e:
                logger.exception("UI callback failed: %s", e)

        live_slot = st.empty()
        _clear_live_events()

        def _rerender_live() -> None:
            _render_live_chat_events_in(live_slot)

        assistant_live_event_id: Optional[int] = None
        history: ChatHistory = st.session_state["history"]

        async def _begin_thinking() -> None:
            await _call_optional(on_thinking_start)
            if not _has_trailing_thinking_event():
                _append_live_event("thinking")
                _rerender_live()

        async def _begin_assistant_stream() -> None:
            nonlocal assistant_live_event_id
            await _call_optional(on_assistant_stream_start)
            _remove_trailing_thinking_event()
            if assistant_live_event_id is None:
                assistant_live_event_id = _append_live_event("assistant", "")
            _rerender_live()

        async def _update_assistant_stream(text: str) -> None:
            nonlocal assistant_live_event_id
            if assistant_live_event_id is None:
                await _begin_assistant_stream()
            if assistant_live_event_id is not None:
                _update_live_event(assistant_live_event_id, content=text)
                _rerender_live()

        async def _end_assistant_stream(still_thinking: bool = False) -> None:
            await _call_optional(on_assistant_stream_end, still_thinking)
            _rerender_live()

        def _commit_non_streamed_assistant_message(content: str) -> None:
            nonlocal assistant_live_event_id
            _remove_trailing_thinking_event()
            assistant_live_event_id = _append_live_event("assistant", content)
            _rerender_live()

        def _commit_tool_message(tool_message: str, tool_name: str) -> None:
            _remove_trailing_thinking_event()
            _append_live_event("tool", tool_message, tool_name=tool_name)
            _rerender_live()

        if not history.user or not history.name:
            show_user_error(
                "Session non définie.",
                details="Veuillez d'abord définir un utilisateur et un nom de session."
            )
            _append_assistant_error_event(
                "⚠️ Session non définie.\n\nVeuillez d'abord définir un utilisateur et un nom de session."
            )
            _rerender_live()
            return

        skip_user_echo = _consume_skip_user_echo_flag()
        if not skip_user_echo:
            _append_live_event("user", user_input)
            _rerender_live()
            _scroll_page_to_bottom()

        history.start_new_request(user_input)
        history.add_user_message(user_input)

        if mcp_client is None:
            mcp_client = st.session_state.get("mcp_client")

        async with mcp_client:
            await _begin_thinking()
            tools = await with_timeout(
                mcp_client.tools(),
                seconds=3000.0,
                on_timeout_msg="listing tools took too long.",
            ) or []
            if tools is None:
                _append_assistant_error_event("⚠️ Tool listing timed out or failed.")
                _rerender_live()
                return

        completions: AsyncCompletions = st.session_state["completions"]
        completion_params = st.session_state.get("completion_params", {})

        loop = True
        while loop:
            assistant_live_event_id = None

            try:
                await _begin_thinking()

                llm_messages = history.build_messages_for_llm(
                    current_user_input=user_input,
                    current_model_prompt="",
                )

                streamed_response = await with_timeout(
                    _create_completion_streaming(
                        completions=completions,
                        llm_messages=llm_messages,
                        tools=tools,
                        completion_params=completion_params,
                        on_assistant_stream_start=_begin_assistant_stream,
                        on_assistant_text_update=_update_assistant_stream,
                    ),
                    seconds=3000.0,
                    on_timeout_msg="Generating assistant response took too long.",
                )
                if streamed_response is None:
                    _append_assistant_error_event("⚠️ Assistant generation timed out or failed.")
                    _rerender_live()
                    return

            except Exception as e:
                show_user_error(
                    "A critical error occurred while generating the assistant response.",
                    details=str(e),
                )
                _append_assistant_error_event(
                    "⚠️ A critical error occurred while generating the assistant response.\n\n"
                    f"{e}"
                )
                _rerender_live()
                return

            try:
                content = streamed_response.get("content", "") or ""
                tool_calls = _normalize_tool_calls(streamed_response.get("tool_calls", []))
                streamed_any_text = bool(streamed_response.get("streamed_any_text", False))

                if streamed_any_text:
                    await _end_assistant_stream(bool(tool_calls))

                if content and not streamed_any_text:
                    await _begin_assistant_stream()
                    _update_live_event(assistant_live_event_id, content=content)
                    _rerender_live()
                    await _end_assistant_stream(bool(tool_calls))

                if not content and not tool_calls:
                    _remove_trailing_thinking_event()
                    _rerender_live()
                    await _end_assistant_stream(False)

                if tool_calls:
                    history.add_assistant_message(
                        content=content,
                        tool_calls=tool_calls,
                    )

                    assistant_live_event_id = None

                    for tool_call in tool_calls:
                        function = tool_call["function"]
                        name = function["name"]
                        raw_arguments_text = function.get("arguments", "{}")

                        await _begin_thinking()

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
                                    _append_assistant_error_event(f"⚠️ Tool '{name}' timed out or failed.")
                                    _rerender_live()
                                    return

                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while calling tool '{name}'.",
                                details=str(e),
                            )
                            _append_assistant_error_event(f"⚠️ Tool error in '{name}': {e}")
                            _rerender_live()
                            return

                        try:
                            _commit_tool_message(tool_message, name)
                            parsed_tool = safe_json_loads(tool_message)

                            history.add_tool_message(
                                content=tool_message,
                                tool_call_id=tool_call["id"],
                                llm_content=tool_message,
                                tool_name=name,
                                arguments=args,
                                raw_arguments_text=raw_arguments_text,
                            )


                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while processing tool output for '{name}'.",
                                details=str(e),
                            )
                            _append_assistant_error_event(
                                f"⚠️ Tool output processing error for '{name}': {e}"
                            )
                            _rerender_live()
                            return

                else:
                    history.add_assistant_message(content)
                    loop = False

            except Exception as e:
                show_user_error(
                    "A critical error occurred while processing the assistant message.",
                    details=str(e),
                )
                _append_assistant_error_event(f"⚠️ Assistant message processing error: {e}")
                _rerender_live()
                return

        try:
            history.save()
            _clear_live_events()
        except Exception as e:
            show_user_error("Saving the conversation failed.", details=str(e))
            _append_assistant_error_event(f"⚠️ Saving conversation failed: {e}")
            _rerender_live()
        return

    except Exception as e:
        show_user_error("A critical unexpected error occurred in chat processing.", details=str(e))
        _append_assistant_error_event(f"⚠️ Unexpected chat processing error: {e}")
        return
