
from __future__ import annotations

# Standard library imports
from typing import Any, Optional
from io import BytesIO
import asyncio
import logging
from os import environ

# Third-party imports
import streamlit as st

# Local application imports
from clients import OpenAIClient, MCPClient
from chat_history import ChatHistory
from .data_model_utils import upload_xml, download_xml, visualise
from .chat_logic import set_chatbox_layout, process_user_input, show_user_error

# ----------------------------------------------------------------------
# Config & logging
# ----------------------------------------------------------------------
CONTACT_EMAIL = "emilien.caudron@pwc.com"  # Dummy email per your request
LOGGER_NAME = "data_modelling_chat_tab"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)

# ----------------------------------------------------------------------
# Persistent error banner renderer
# ----------------------------------------------------------------------
def render_persistent_error_banner() -> None:
    """
    Render a persistent error banner based on `st.session_state["ui_error"]`.
    Provides a Dismiss button to clear the error.
    """
    err = st.session_state.get("ui_error")
    if not err:
        return

    with st.container():
        st.error(err.get("title", "An error occurred"))
        details = err.get("details")
        if details:
            st.write(details)
        contact_email = err.get("contact_email", CONTACT_EMAIL)
        st.markdown(
            f"""
**What you can do now:**
1) Review your inputs and correct the bug if possible.  
2) Re-launch the UI.  
3) If the error keeps happening, contact the tech team at **{contact_email}**.
            """
        )
        if st.button("Dismiss error"):
            st.session_state.pop("ui_error", None)
            st.rerun()

# ----------------------------------------------------------------------
# Async timeout helper (uses shared show_user_error)
# ----------------------------------------------------------------------
async def with_timeout(coro, seconds: float = 45.0, on_timeout_msg: str = ""):
    """
    Await a coroutine with a timeout; surface a user message on timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        show_user_error(
            "The operation timed out.",
            details=on_timeout_msg or "The server took too long to respond.",
        )
        return None

def safe_json_loads(text: Optional[str]) -> dict:
    """
    Parse JSON safely, returning {} on failure and logging the error.
    """
    if not text:
        return {}
    try:
        from json import loads
        return loads(text)
    except Exception as e:
        logger.exception("JSON parsing failed: %s", e)
        return {}

# ----------------------------------------------------------------------
# Main tab
# ----------------------------------------------------------------------
async def data_modelling_chat_tab(server:str) -> None:
    """
    Main function for the data modelling chat tab in the Streamlit app.
    Handles user/session management, model upload/visualization, and chat interface.
    Ensures robust error handling and user feedback for all major operations.
    """
    # Layout: three columns for user/session, (spacer), and main chat/model area
    col1, col2, col3 = st.columns([0.15, 0.1, 0.75], gap="small")

    # Render any persisted error banner first (so errors don't disappear)
    render_persistent_error_banner()

    # --- Initialization and session state setup ---
    try:
        # quick env check to avoid late failures
        if not environ.get("LLM_MODEL"):
            show_user_error("Missing LLM_MODEL environment variable.")
            return

        st.session_state["visualise"] = False  # Reset visualisation toggle each run

        # Initialize chat history if not present
        if "history" not in st.session_state:
            st.session_state['history'] = ChatHistory()

        chat_history: ChatHistory = st.session_state["history"]
        st.session_state["user"] = chat_history.user
        st.session_state["name"] = chat_history.name

        # Initialize OpenAI completions client if not present
        if "completions" not in st.session_state:
            try:
                st.session_state["completions"] = OpenAIClient().chat_completions
            except Exception as e:
                show_user_error("LLM client initialization failed.", details=str(e))
                return

        # Initialize MCP client if not present
        if "mcp_client" not in st.session_state:
            try:
                st.session_state["mcp_client"] = MCPClient(st.session_state, server=server)
            except Exception as e:
                show_user_error("MCP client initialization failed.", details=str(e))
                return

        model: dict[str, Any] = {}

        # Load model from server if user and session name are set
        if st.session_state["user"] and st.session_state["name"]:
            try:
                async with st.session_state["mcp_client"] as mcp_client:
                    model = await with_timeout(
                        mcp_client.read_model(),
                        seconds=30.0,
                        on_timeout_msg="Loading the model took too long."
                    ) or {}
            except Exception as e:
                show_user_error("A critical error occurred while loading the model.", details=str(e))
                return

        st.session_state["model"] = model

        # Set root model IDs in session state if available
        if model.get("elements", []):
            root: dict[str, Any] = model["elements"][0]
            st.session_state["ID"] = root.get("ID")
            st.session_state["package"] = root.get("package")
    except Exception as e:
        show_user_error("A critical error occurred during initialization.", details=str(e))
        return

    # --- Column 1: User/session management ---
    with col1:
        try:
            if chat_history.user:
                st.write(f'User: {chat_history.user}')
                if chat_history.name:
                    st.write(f'Session: {chat_history.name}')
                else:
                    # Input for session name if not set
                    chat_history.name = st.text_input(label='Enter Session:')
                    if st.button(label='Set Session', disabled=not bool(chat_history.user)):
                        try:
                            chat_history.save()
                        except Exception as e:
                            show_user_error("A critical error occurred while saving the session.", details=str(e))
                            return
                        st.rerun()

                # Option to reload a different session
                reload_session = st.text_input(
                    label='Session to load:',
                    disabled=not bool(chat_history.user),
                    placeholder="",
                )
                if st.button(label='Load Session', disabled=not bool(chat_history.user)):
                    try:
                        chat_history.load(reload_session)
                    except Exception as e:
                        show_user_error("A critical error occurred while loading the session.", details=str(e))
                        return
                    st.rerun()
            else:
                # Input for user name if not set
                chat_history.user = st.text_input(label='Enter User:')
                if st.button('Set User'):
                    st.rerun()
        except Exception as e:
            show_user_error("A critical error occurred in user/session management.", details=str(e))
            return

    # --- Column 3: Model upload/visualisation + chat ---
    with col3:
        try:
            if all(bool(st.session_state.get(required)) for required in {"user", "name"}):
                model = st.session_state["model"]
                if not model:
                    # File uploader for XML/TTL model files
                    st.session_state["file"] = st.file_uploader(
                        "Upload an XML/TTL document",
                        type=["xml", "ttl"],
                        accept_multiple_files=False,
                        help="Upload your data model as an XML export from EA or a TTL file",
                    )
                    if st.session_state["file"] is not None:
                        try:
                            uploaded_file: BytesIO = st.session_state["file"]
                            st.session_state["model"] = await with_timeout(
                                upload_xml(uploaded_file),
                                seconds=60.0,
                                on_timeout_msg="Uploading/parsing the file took too long."
                            ) or {}
                        except Exception as e:
                            show_user_error("A critical error occurred while uploading the file.", details=str(e))
                            return
                        st.rerun()
                else:
                    try:
                        # local function; keep synchronous
                        download_xml(model)
                        st.session_state["visualise"] = st.checkbox(
                            "Visualise Model",
                            value=st.session_state.get("visualise", False),
                        )
                        if st.session_state.get("visualise", False):
                            if "xmi" in model.keys():
                                # visualise() is synchronous – DO NOT await it
                                visualise(model["xmi"])
                            else:
                                visualise(model)
                    except Exception as e:
                        # Log but avoid crashing UI
                        logger.exception("Downloading/visualising model failed: %s", e)
                        show_user_error("A critical error occurred while downloading or visualising the model.", details=str(e))

            # Set up the chatbox interface
            set_chatbox_layout()

            # Handle user chat input and process with LLM
            try:
                if user_input := st.chat_input(key="xmi_chat_input"):
                    await process_user_input(user_input)
                    # Only rerun after success; errors persist via banner
                    st.rerun()
            except Exception as e:
                show_user_error("A critical error occurred while processing user input.", details=str(e))
                return
        except Exception as e:
            show_user_error("A critical error occurred in the model/visualisation/chat interface.", details=str(e))
            return
