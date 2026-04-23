
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
CONTACT_EMAIL = "emilien.caudron@pwc.com"  # Dummy email per your request
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
    # Persist to session_state for rendering on subsequent runs
    st.session_state["ui_error"] = {
        "title": title,
        "details": details or "",
        "contact_email": CONTACT_EMAIL,
    }

    # Optional immediate transient status (will disappear on rerun)
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
        # Fallback for environments without st.status
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

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def set_chatbox_layout() -> None:
    """
    Sets up the chat interface layout in Streamlit, including message display and UI fixes.
    Displays the conversation history for user, assistant, and tool messages.
    """
    st.title("Model Bot")

    # display history safely
    for msg in st.session_state['history'].messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and content:
            st.chat_message("human").write(content)
        elif role == "assistant" and content:
            st.chat_message("ai").write(content)
        elif role == "tool" and content:
            parsed = safe_json_loads(content)
            if parsed:
                st.json(parsed, expanded=False)
            else:
                st.code(content)

    # Fix CSS (added missing colon)
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

        # Display the user's message in the chat
        st.chat_message("human").write(user_input)

        # Read model context if available
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

        # Gather available tools from the MCP client
        async with st.session_state["mcp_client"] as mcp_client:
            tools = await with_timeout(
                mcp_client.tools(),
                seconds=3000.0,
                on_timeout_msg="listing tools took too long."
            ) or []
            if tools is None:
                # Already surfaced error; stop
                return

        # Add the user message (with model context) to the chat history
        user_message = ChatCompletionUserMessageParam(
            role="user",
            content=f"{model_prompt}{user_input}"
        )
        history: ChatHistory = st.session_state["history"]
        history.messages.append(user_message)

        # Completion client for LLM responses
        completions: AsyncCompletions = st.session_state["completions"]

        # Chat loop: generate a response, call tools, and repeat until no tool call is needed
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
                    ),
                    seconds=3000.0,
                    on_timeout_msg="Generating assistant response took too long."
                )
                if response is None:
                    # Error already shown persistently
                    return
            except Exception as e:
                show_user_error(
                    "A critical error occurred while generating the assistant response.",
                    details=str(e)
                )
                # Mirror the error inline in the chat area for context
                st.chat_message("ai").write(f"⚠️ A critical error occurred while generating the assistant response.\n\n{e}")
                return  # Do not rerun; banner will persist

            try:
                message: ChatCompletionMessage = response.choices[0].message
                content = message.content or ""
                if content:
                    st.chat_message("ai").write(content)

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
                    # Add the assistant message and tool calls to history
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
                                    # Error already shown persistently
                                    return
                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while calling tool '{name}'.",
                                details=str(e)
                            )
                            st.chat_message("ai").write(f"⚠️ Tool error in '{name}': {e}")
                            return

                        try:
                            parsed_tool = safe_json_loads(tool_message)
                            if parsed_tool:
                                st.json(parsed_tool, expanded=2)
                            else:
                                st.code(tool_message)

                            history.add_tool_message(
                                content=tool_message,
                                tool_call_id=tool_call["id"],
                            )

                            if name == 'style_guide_check':
                                try:
                                    report = parsed_tool.get("tool_results", {}).get("report", "")
                                    if report:
                                        st.chat_message("ai").write(report)
                                        history.add_assistant_message(report)
                                    else:
                                        st.chat_message("ai").write("Style guide check completed.")
                                except Exception as e:
                                    show_user_error(
                                        "A critical error occurred while displaying the style guide report.",
                                        details=str(e)
                                    )
                                    st.chat_message("ai").write(f"⚠️ Style guide display error: {e}")
                                    return
                                # End the loop after style guide check
                                loop = False
                        except Exception as e:
                            show_user_error(
                                f"A critical error occurred while processing tool output for '{name}'.",
                                details=str(e)
                            )
                            st.chat_message("ai").write(f"⚠️ Tool output processing error for '{name}': {e}")
                            return
                else:
                    # No tool calls: add the assistant message and exit loop
                    history.add_assistant_message(content)
                    loop = False
            except Exception as e:
                show_user_error(
                    "A critical error occurred while processing the assistant message.",
                    details=str(e)
                )
                st.chat_message("ai").write(f"⚠️ Assistant message processing error: {e}")
                return

        # Remove the model context from the last user message and save the conversation
        user_message["content"] = user_input
        try:
            history.save()
        except Exception as e:
            # Saving failure should not crash the UI; surface guidance persistently
            show_user_error("Saving the conversation failed.", details=str(e))
            st.chat_message("ai").write(f"⚠️ Saving conversation failed: {e}")
        return

    except Exception as e:
        show_user_error("A critical unexpected error occurred in chat processing.", details=str(e))
        st.chat_message("ai").write(f"⚠️ Unexpected chat processing error: {e}")
        return

