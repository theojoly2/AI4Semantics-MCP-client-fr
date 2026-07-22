from __future__ import annotations

from typing import Any, Optional
import asyncio
import logging
from json import loads

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from clients import OpenAIClient, MCPClient
from chat_history import ChatHistory
from .chat_logic import set_chatbox_layout, process_user_input, show_user_error, _render_tool_output


CONTACT_EMAIL = "theo.joly2@developpement-durable.gouv.fr"
LOGGER_NAME = "glossary_chat_tab"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)


def safe_json_loads(text: Optional[str]) -> Any:
    if not text:
        return {}
    try:
        return loads(text)
    except Exception as e:
        logger.exception("JSON parsing failed: %s", e)
        return {}


async def with_timeout(coro, seconds: float = 45.0, on_timeout_msg: str = ""):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        show_user_error(
            "The operation timed out.",
            details=on_timeout_msg or "The server took too long to respond.",
        )
        return None


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
            div[data-testid="stDecoration"] {
                display: none;
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
                width: 100% !important;
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
            button[data-generating="true"] {
                opacity: 0.5 !important;
                cursor: not-allowed !important;
            }
            div[data-testid="stStatusWidget"] {
                display: none !important;
            }
            [data-testid="stToolbar"] button[kind="header"] {
                display: none !important;
            }
            .chat-thinking-wrap {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                color: #6b7280;
                font-size: 0.95rem;
                min-height: 24;
                padding: 0;
            }
            .chat-thinking-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(107, 114, 128, 0.25);
                border-top-color: rgba(107, 114, 128, 0.95);
                border-radius: 50%;
                animation: chat-thinking-spin 0.8s linear infinite;
                transform: translateY(-7.5px);
            }
            .chat-thinking-label {
                display: flex;
                align-items: center;
                line-height: 1;
                margin: -1;
                padding: 0;
                transform: translateY(-7px);
            }
            .chat-thinking-label p,
            .chat-thinking-label span,
            .chat-thinking-label div {
                margin: 0 !important;
                padding: 0 !important;
                line-height: 1 !important;
            }
            @keyframes chat-thinking-spin {
                from { transform: translateY(-7.5px) rotate(0deg); }
                to { transform: translateY(-7.5px) rotate(360deg); }
            }
            .source-card {
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                background-color: #fafafa;
            }
            .source-meta {
                font-size: 0.85rem;
                color: #555;
                margin-bottom: 0.25rem;
            }
            .source-text {
                font-size: 0.95rem;
                color: #111;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clear_generation_state() -> None:
    st.session_state["_generating"] = False
    st.session_state["_pending_input"] = None
    st.session_state["_thinking_visible"] = False
    st.session_state["_assistant_streaming"] = False


def _init_state(server: str) -> ChatHistory:
    st.session_state.setdefault("_generating", False)
    st.session_state.setdefault("_pending_input", None)
    st.session_state.setdefault("_thinking_visible", False)
    st.session_state.setdefault("_assistant_streaming", False)
    st.session_state.setdefault("_live_chat_events", [])
    st.session_state.setdefault("_live_event_seq", 0)
    st.session_state.setdefault("selected_tags", [])
    st.session_state.setdefault("_tags_initialized", False)

    if "mcp_client" not in st.session_state:
        logger.info("Instanciation du MCPClient sur le serveur : %s", server)
        client = MCPClient(st.session_state, server=server)
        st.session_state["mcp_client"] = client

        try:
            if hasattr(client, "initialize"):
                asyncio.run(client.initialize())
            elif hasattr(client, "start"):
                asyncio.run(client.start())
            elif hasattr(client, "connect"):
                asyncio.run(client.connect())
        except Exception as conn_err:
            logger.error("Échec de l'initialisation immédiate du client MCP : %s", conn_err)

    if "history" not in st.session_state:
        st.session_state["history"] = ChatHistory()

    if "completions" not in st.session_state:
        st.session_state["completions"] = OpenAIClient().chat_completions
        st.session_state["completion_params"] = OpenAIClient().completion_params

    chat_history: ChatHistory = st.session_state["history"]
    st.session_state["user"] = chat_history.user
    st.session_state["name"] = chat_history.name

    return chat_history


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("GlossaryAI")

        chat_history: ChatHistory = st.session_state["history"]
        is_generating = st.session_state.get("_generating", False)

        if not chat_history.user:
            user_value = st.text_input(
                "Nom d'utilisateur",
                key="sidebar_user_value",
                placeholder="Votre identifiant",
                disabled=is_generating,
            )
            if st.button(
                "Valider l'utilisateur",
                key="sidebar_set_user",
                use_container_width=True,
                disabled=is_generating,
            ):
                user_value = user_value.strip()
                if not user_value:
                    show_user_error("Veuillez saisir un utilisateur avant de continuer.")
                    return
                _set_active_history(ChatHistory(user=user_value))
                st.session_state["_reset_sidebar_inputs"] = True
                st.rerun()
        else:
            st.write(f"**Utilisateur:** {chat_history.user}")

            if not chat_history.name:
                session_value = st.text_input(
                    "Nom de la session",
                    key="sidebar_session_value",
                    placeholder="Nom de session",
                    disabled=is_generating,
                )
                if st.button(
                    "Valider la session",
                    key="sidebar_set_session",
                    use_container_width=True,
                    disabled=is_generating,
                ):
                    session_value = session_value.strip()
                    if not session_value:
                        show_user_error("Veuillez saisir un nom de session.")
                        return
                    try:
                        _open_or_create_history(chat_history.user, session_value)
                        st.session_state["_reset_sidebar_inputs"] = True
                        st.rerun()
                    except Exception as e:
                        show_user_error(
                            "Une erreur est survenue lors de l'ouverture de la session.",
                            details=str(e),
                        )
                        return
            else:
                st.write(f"**Session:** {chat_history.name}")

                reload_session = st.text_input(
                    "Session à charger",
                    key="sidebar_reload_session",
                    placeholder="Session existante",
                    disabled=is_generating,
                )
                if st.button(
                    "Charger la session",
                    key="sidebar_load_session",
                    use_container_width=True,
                    disabled=is_generating,
                ):
                    reload_session = reload_session.strip()
                    if not reload_session:
                        show_user_error("Veuillez saisir un nom de session à charger.")
                        return
                    try:
                        _open_or_create_history(chat_history.user, reload_session)
                        st.session_state["_reset_sidebar_inputs"] = True
                        st.rerun()
                    except Exception as e:
                        show_user_error(
                            "Une erreur est survenue lors du chargement de la session.",
                            details=str(e),
                        )
                        return

        st.divider()
        st.subheader("Filtres par source")

        available_tags = _fetch_tags()
        if not available_tags:
            st.caption("Aucune source disponible. Vérifiez l'indexation.")
        else:
            if not st.session_state.get("_tags_initialized", False):
                st.session_state["selected_tags"] = [t for t in available_tags]
                st.session_state["_tags_initialized"] = True

            new_selected_tags = []
            for i, tag_name in enumerate(available_tags):
                is_checked = st.checkbox(
                    f"{tag_name}",
                    value=(tag_name in st.session_state.get("selected_tags", [])),
                    key=f"ui_filter_tag_{i}_{tag_name}",
                )
                if is_checked:
                    new_selected_tags.append(tag_name)
            st.session_state["selected_tags"] = new_selected_tags


def _reset_sidebar_widget_values_if_needed() -> None:
    if st.session_state.pop("_reset_sidebar_inputs", False):
        st.session_state.pop("sidebar_session_value", None)
        st.session_state.pop("sidebar_reload_session", None)
        st.session_state.pop("sidebar_user_value", None)


def _set_active_history(history: ChatHistory) -> None:
    st.session_state["history"] = history
    st.session_state["user"] = history.user
    st.session_state["name"] = history.name

    _clear_generation_state()
    st.session_state["_live_chat_events"] = []
    st.session_state["_live_event_seq"] = 0


def _open_or_create_history(user: str, session_name: str) -> ChatHistory:
    existed = ChatHistory.session_exists(user, session_name)
    history = ChatHistory(user=user, name=session_name)

    if not existed:
        history.save()

    _set_active_history(history)
    return history


def _fetch_tags() -> list[str]:
    mcp_client = st.session_state.get("mcp_client")
    if not mcp_client:
        return []

    async def _do_fetch():
        async with mcp_client as wrapper:
            if hasattr(wrapper, "get_available_tags"):
                return await wrapper.get_available_tags()
            return []

    try:
        raw_tags = asyncio.run(_do_fetch())
    except Exception as e:
        logger.warning("Impossible de charger les tags via MCP: %s", e)
        return []

    tags: list[str] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, dict) and "tag" in item:
                tags.append(item["tag"])
            elif isinstance(item, str):
                tags.append(item)
    return tags


def _render_persistent_error_banner() -> None:
    err = st.session_state.get("ui_error")
    if not err:
        return

    with st.container():
        st.error(err.get("title", "Une erreur est survenue"))
        details = err.get("details")
        if details:
            st.write(details)
        contact_email = err.get("contact_email", CONTACT_EMAIL)
        st.markdown(
            f"""
**Ce que vous pouvez faire maintenant :**
1) Vérifiez vos saisies et corrigez le bug si possible.
2) Relancez l'interface utilisateur.
3) Si l'erreur persiste, contactez l'équipe technique à l'adresse **{contact_email}**.
            """
        )
        if st.button("Masquer l'erreur", key="dismiss_error_button"):
            st.session_state.pop("ui_error", None)
            st.rerun()


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
                _clear_generation_state()
            st.rerun()


def data_modelling_chat_tab(server: str) -> None:
    _render_persistent_error_banner()
    _inject_layout_css()

    try:
        chat_history = _init_state(server)
    except Exception as e:
        show_user_error("Une erreur est survenue lors de l'initialisation.", details=str(e))
        return

    _reset_sidebar_widget_values_if_needed()

    try:
        _render_sidebar()
    except Exception as e:
        show_user_error("Une erreur est survenue dans la barre latérale.", details=str(e))
        return

    try:
        is_generating = st.session_state.get("_generating", False)

        with st.container():
            st.title("GlossaryAI")
            st.caption("Assistant vocabulaire, glossaire et textes juridiques")

            filter_spacer, filter_col = st.columns([6.5, 2.5], vertical_alignment="center")
            with filter_col:
                with st.popover("🏷️ Filtres", disabled=is_generating, use_container_width=True):
                    st.write("**Filtrer la recherche :**")
                    available_tags = _fetch_tags()
                    if not available_tags:
                        st.info("Aucune source disponible.")
                    else:
                        if not st.session_state.get("_tags_initialized", False):
                            st.session_state["selected_tags"] = list(available_tags)
                            st.session_state["_tags_initialized"] = True

                        new_selected_tags = []
                        for i, tag_name in enumerate(available_tags):
                            is_checked = st.checkbox(
                                f"{tag_name}",
                                value=(tag_name in st.session_state.get("selected_tags", [])),
                                key=f"top_filter_tag_{i}_{tag_name}",
                            )
                            if is_checked:
                                new_selected_tags.append(tag_name)
                        st.session_state["selected_tags"] = new_selected_tags

        chat_container = st.container()

        input_left, input_center, input_right = st.columns([1, 6, 1])
        with input_center:
            user_input = st.chat_input(
                "Génération en cours..." if is_generating else "Votre message",
                key="glossary_chat_input",
                disabled=is_generating,
            )

        _render_page_bottom_guard(66)

        if user_input and not is_generating:
            st.session_state["_generating"] = True
            st.session_state["_pending_input"] = user_input
            st.rerun()

        with chat_container:
            pending = st.session_state.pop("_pending_input", None)
            if pending:
                asyncio.run(_render_chat_panel(user_input=pending))
            else:
                asyncio.run(_render_chat_panel())

    except Exception as e:
        _clear_generation_state()
        show_user_error(
            "Une erreur est survenue dans l'interface de chat.",
            details=str(e),
        )
        return
