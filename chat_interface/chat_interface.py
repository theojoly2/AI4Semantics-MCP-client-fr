from __future__ import annotations

from typing import Any, Optional
from io import BytesIO
import asyncio
import logging
from os import environ

import streamlit as st

# Local application imports
from clients import OpenAIClient, MCPClient
from chat_history import ChatHistory
from .data_model_utils import upload_xml, download_xml, visualise
from .chat_logic import set_chatbox_layout, process_user_input, show_user_error


# ----------------------------------------------------------------------
# Config & logging
# ----------------------------------------------------------------------
CONTACT_EMAIL = "emilien.caudron@pwc.com"
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
        if st.button("Dismiss error", key="dismiss_error_button"):
            st.session_state.pop("ui_error", None)
            st.rerun()


# ----------------------------------------------------------------------
# Async timeout helper
# ----------------------------------------------------------------------
async def with_timeout(coro, seconds: float = 45.0, on_timeout_msg: str = ""):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        show_user_error(
            "The operation timed out.",
            details=on_timeout_msg or "The server took too long to respond.",
        )
        return None


def safe_json_loads(text: Optional[str]) -> dict:
    if not text:
        return {}
    try:
        from json import loads
        return loads(text)
    except Exception as e:
        logger.exception("JSON parsing failed: %s", e)
        return {}


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _inject_layout_css() -> None:
    st.markdown(
        """
        <style>
            div[data-testid="stMainBlockContainer"] {
                padding-top: 1rem !important;
                padding-bottom: 0rem !important;
            }

            .block-container {
                padding-top: 1.9rem !important;
                padding-bottom: 0rem !important;
            }

            header.stAppHeader {
                background: transparent;
                height: 2.2rem;
                min-height: 2.2rem;
            }

            div[data-testid="stToolbar"] {
                top: 0.15rem !important;
            }

            div[data-testid="stDecoration"] {
                display: none;
            }

            section[data-testid="stSidebar"] .stTextInput,
            section[data-testid="stSidebar"] .stButton,
            section[data-testid="stSidebar"] .stFileUploader {
                width: 100%;
            }

            section[data-testid="stSidebar"] .stButton > button {
                width: 100%;
                min-height: 44px;
            }

            div[data-testid="column"]:has(.model-sticky-anchor) {
                position: sticky;
                top: 2.35rem;
                align-self: flex-start;
            }

            div[data-testid="column"]:has(.chat-scroll-anchor) {
                align-self: flex-start;
                overflow-x: hidden;
                padding-left: 0.5rem;
                padding-right: 0.85rem;
                box-sizing: border-box;
            }

            div[data-testid="stChatInput"] {
                max-width: 100%;
                margin-top: 0rem !important;
                margin-bottom: 0rem !important;
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
            }

            .stChatFloatingInputContainer {
                padding-top: 0.2rem !important;
                padding-bottom: 0.2rem !important;
                background: transparent !important;
            }

            div[data-testid="stChatInput"] textarea,
            div[data-testid="stChatInput"] input {
                padding-top: 0.35rem !important;
                padding-bottom: 0.35rem !important;
            }

            div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] {
                border: none;
                padding-right: 0.25rem;
                box-sizing: border-box;
            }

            .collapsed-canvas-wrap {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                padding-right: 0.75rem;
                gap: 0.05rem;
                position: sticky;
                top: 0.2rem;
                z-index: 5;
                background: white;
            }

            .collapsed-canvas-button {
                margin-bottom: 0 !important;
            }

            .collapsed-canvas-button button {
                min-width: 2.2rem !important;
                width: 2.2rem !important;
                height: 2.2rem !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            .collapsed-canvas-text {
                font-size: 0.85rem;
                color: #7a7a7a;
                line-height: 1.05;
                white-space: normal;
                word-break: keep-all;
                margin-top: -1.5rem;
            }

            button[data-generating="true"] {
                opacity: 0.5 !important;
                cursor: not-allowed !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sync_session_from_history(chat_history: ChatHistory) -> None:
    st.session_state["user"] = chat_history.user
    st.session_state["name"] = chat_history.name


def _toggle_visualisation_panel() -> None:
    st.session_state["panel_collapsed"] = not st.session_state.get("panel_collapsed", False)


def _init_state(server: str) -> ChatHistory:
    if not environ.get("LLM_MODEL"):
        raise RuntimeError("Missing LLM_MODEL environment variable.")

    st.session_state.setdefault("visualise", True)
    st.session_state.setdefault("panel_collapsed", False)
    st.session_state.setdefault("model", {})
    st.session_state.setdefault("_generating", False)
    st.session_state.setdefault("_pending_input", None)

    if "history" not in st.session_state:
        st.session_state["history"] = ChatHistory()

    chat_history: ChatHistory = st.session_state["history"]
    _sync_session_from_history(chat_history)

    if "completions" not in st.session_state:
        st.session_state["completions"] = OpenAIClient().chat_completions

    if "mcp_client" not in st.session_state:
        st.session_state["mcp_client"] = MCPClient(st.session_state, server=server)

    return chat_history


def _render_connection_sidebar(chat_history: ChatHistory) -> None:
    is_generating = st.session_state.get("_generating", False)

    with st.sidebar:
        st.header("Connexion")

        if not chat_history.user:
            user_value = st.text_input(
                "Enter User",
                key="sidebar_user_value",
                placeholder="Votre identifiant",
                disabled=is_generating,
            )
            if st.button(
                "Set User",
                key="sidebar_set_user",
                use_container_width=True,
                disabled=is_generating,
            ):
                user_value = user_value.strip()
                if not user_value:
                    show_user_error("Please enter a user before continuing.")
                    return
                chat_history.user = user_value
                _sync_session_from_history(chat_history)
                st.rerun()

        else:
            st.write(f"**User:** {chat_history.user}")

            if not chat_history.name:
                session_value = st.text_input(
                    "Enter Session",
                    key="sidebar_session_value",
                    placeholder="Nom de session",
                    disabled=is_generating,
                )
                if st.button(
                    "Set Session",
                    key="sidebar_set_session",
                    use_container_width=True,
                    disabled=is_generating,
                ):
                    session_value = session_value.strip()
                    if not session_value:
                        show_user_error("Please enter a session name before continuing.")
                        return
                    chat_history.name = session_value
                    try:
                        chat_history.save()
                    except Exception as e:
                        show_user_error(
                            "A critical error occurred while saving the session.",
                            details=str(e),
                        )
                        return
                    _sync_session_from_history(chat_history)
                    st.rerun()
            else:
                st.write(f"**Session:** {chat_history.name}")

                reload_session = st.text_input(
                    "Session to load",
                    key="sidebar_reload_session",
                    placeholder="Session existante",
                    disabled=is_generating,
                )
                if st.button(
                    "Load Session",
                    key="sidebar_load_session",
                    use_container_width=True,
                    disabled=is_generating,
                ):
                    reload_session = reload_session.strip()
                    if not reload_session:
                        show_user_error("Please enter a session name to load.")
                        return
                    try:
                        chat_history.load(reload_session)
                        _sync_session_from_history(chat_history)
                        st.rerun()
                    except Exception as e:
                        show_user_error(
                            "A critical error occurred while loading the session.",
                            details=str(e),
                        )
                        return

        st.divider()

        toggle_label = (
            "Masquer le canvas"
            if not st.session_state.get("panel_collapsed", False)
            else "Afficher le canvas"
        )
        if st.button(
            "⏳ Génération en cours..." if is_generating else toggle_label,
            key="sidebar_toggle_visualisation",
            use_container_width=True,
            disabled=is_generating,
        ):
            _toggle_visualisation_panel()
            st.rerun()


async def _load_model_if_possible() -> dict[str, Any]:
    existing_model = st.session_state.get("model", {}) or {}
    model: dict[str, Any] = existing_model

    if st.session_state.get("user") and st.session_state.get("name"):
        async with st.session_state["mcp_client"] as mcp_client:
            loaded_model = await with_timeout(
                mcp_client.read_model(),
                seconds=30.0,
                on_timeout_msg="Loading the model took too long.",
            )
            if loaded_model:
                model = loaded_model

    st.session_state["model"] = model

    if model.get("elements"):
        root: dict[str, Any] = model["elements"][0]
        st.session_state["ID"] = root.get("ID")
        st.session_state["package"] = root.get("package")

    return model


async def _render_model_panel() -> None:
    is_generating = st.session_state.get("_generating", False)

    st.markdown("<div class='model-sticky-anchor'></div>", unsafe_allow_html=True)

    if st.session_state.get("panel_collapsed", False):
        st.markdown("<div class='collapsed-canvas-wrap'>", unsafe_allow_html=True)
        st.markdown("<div class='collapsed-canvas-button'>", unsafe_allow_html=True)
        if st.button(
            "⏳" if is_generating else "▶",
            key="inline_toggle_canvas_collapsed",
            disabled=is_generating,
            help="Génération en cours, veuillez patienter..." if is_generating else None,
        ):
            _toggle_visualisation_panel()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='collapsed-canvas-text'>Canvas<br>rétracté.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    toolbar_left, toolbar_right = st.columns([6, 1], gap="small")
    with toolbar_left:
        st.subheader("Visualisation du modèle")
    with toolbar_right:
        if st.button(
            "⏳" if is_generating else "◀",
            key="inline_toggle_canvas_expanded",
            use_container_width=True,
            disabled=is_generating,
            help="Génération en cours, veuillez patienter..." if is_generating else None,
        ):
            _toggle_visualisation_panel()
            st.rerun()

    if not all(bool(st.session_state.get(required)) for required in {"user", "name"}):
        st.info("Définissez d'abord un utilisateur et une session dans la barre latérale.")
        return

    model = st.session_state.get("model", {})

    if not model:
        uploaded_file = st.file_uploader(
            "Upload an XML/TTL document",
            type=["xml", "ttl", "xmi"],
            accept_multiple_files=False,
            help="Upload your data model as an XML export from EA or a TTL file",
            key="model_file_uploader",
            disabled=is_generating,
        )

        if uploaded_file is not None:
            try:
                file_buffer: BytesIO = uploaded_file
                st.session_state["model"] = await with_timeout(
                    upload_xml(file_buffer),
                    seconds=60.0,
                    on_timeout_msg="Uploading/parsing the file took too long.",
                ) or {}
                st.rerun()
            except Exception as e:
                show_user_error(
                    "A critical error occurred while uploading the file.",
                    details=str(e),
                )
    else:
        try:
            if not is_generating:
                download_xml(model)
            else:
                st.button(
                    "⏳ Téléchargement indisponible",
                    disabled=True,
                    help="Génération en cours, veuillez patienter...",
                    key="download_disabled_placeholder",
                )

            if "xmi" in model:
                visualise(model["xmi"])
            else:
                visualise(model)

        except Exception as e:
            logger.exception("Downloading/visualising model failed: %s", e)
            show_user_error(
                "A critical error occurred while downloading or visualising the model.",
                details=str(e),
            )

def _render_page_bottom_guard(height_px: int = 56) -> None:
    st.markdown(
        f"<div style='height: {height_px}px; width: 100%;'></div>",
        unsafe_allow_html=True,
    )


async def _render_chat_panel(user_input: Optional[str] = None) -> None:
    st.markdown("<div class='chat-scroll-anchor'></div>", unsafe_allow_html=True)

    with st.container(height=540, border=False):
        set_chatbox_layout()

        if user_input:
            try:
                await process_user_input(user_input)
            finally:
                st.session_state["_generating"] = False
            st.rerun()


# ----------------------------------------------------------------------
# Main tab
# ----------------------------------------------------------------------
async def data_modelling_chat_tab(server: str) -> None:
    render_persistent_error_banner()
    _inject_layout_css()

    try:
        chat_history = _init_state(server)
    except Exception as e:
        show_user_error("A critical error occurred during initialization.", details=str(e))
        return

    try:
        _render_connection_sidebar(chat_history)
    except Exception as e:
        show_user_error("A critical error occurred in user/session management.", details=str(e))
        return

    try:
        await _load_model_if_possible()
    except Exception as e:
        show_user_error("A critical error occurred while loading the model.", details=str(e))
        return

    try:
        if st.session_state.get("panel_collapsed", False):
            col_model, col_chat = st.columns(
                [0.70, 11.30],
                gap="small",
                vertical_alignment="top",
            )
        else:
            col_model, col_chat = st.columns(
                [7, 5],
                gap="medium",
                vertical_alignment="top",
            )

        with col_model:
            await _render_model_panel()

        with col_chat:
            chat_container = st.container()

        is_generating = st.session_state.get("_generating", False)

        input_left, input_center, input_right = st.columns([1, 6, 1])

        with input_center:
            user_input = st.chat_input(
                "⏳ Génération en cours..." if is_generating else "Your message",
                key="xmi_chat_input",
                width="stretch",
                disabled=is_generating,
            )

        _render_page_bottom_guard(66)

        if user_input and not is_generating:
            st.session_state["_generating"] = True
            st.session_state["_pending_input"] = user_input
            st.rerun()

        with chat_container:
            await _render_chat_panel(
                user_input=st.session_state.pop("_pending_input", None)
            )

    except Exception as e:
        show_user_error(
            "A critical error occurred in the model/visualisation/chat interface.",
            details=str(e),
        )
        return
