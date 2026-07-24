from __future__ import annotations

from typing import Any, Optional
import asyncio
import logging
from json import loads
from datetime import datetime, timezone

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from clients import OpenAIClient, MCPClient
from chat_history.chat_history import ChatHistory
from .chat_logic import set_chatbox_layout, process_user_input, show_user_error, _render_tool_output


ANONYMOUS_USER = "anonymous"
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


def _inject_scroll_js() -> None:
    """Inject auto-scroll script directly into the chat panel so it runs after every render."""
    st.markdown(
        """
        <script>
            (function () {
                window.__glossaryUserScrolledUp = window.__glossaryUserScrolledUp || false;
                let userHasScrolledUp = window.__glossaryUserScrolledUp;

                function isNearBottom() {
                    const el = document.documentElement;
                    return (el.scrollHeight - el.scrollTop - el.clientHeight) < 200;
                }

                function scrollToBottom() {
                    if (userHasScrolledUp) return;
                    const el = document.documentElement;
                    const target = el.scrollHeight;
                    if (el.scrollTop < target - 10) {
                        window.scrollTo({ top: target, left: 0, behavior: 'auto' });
                        el.scrollTop = target;
                    }
                }

                function onWindowScroll() {
                    const el = document.documentElement;
                    // Detect intentional upward scroll (user wants to read old content)
                    if (el.scrollTop < (window.__glossaryPrevScrollTop || 0) - 20) {
                        userHasScrolledUp = true;
                        window.__glossaryUserScrolledUp = true;
                    } else if (isNearBottom()) {
                        userHasScrolledUp = false;
                        window.__glossaryUserScrolledUp = false;
                    }
                    window.__glossaryPrevScrollTop = el.scrollTop;
                }

                window.removeEventListener('scroll', onWindowScroll);
                window.addEventListener('scroll', onWindowScroll, { passive: true });

                // Run immediately so the page is at the bottom right after the user sends a message
                scrollToBottom();

                // Also hook DOM mutations in case page height changes during generation
                const observer = new MutationObserver(function () {
                    if (!userHasScrolledUp) scrollToBottom();
                });
                observer.observe(document.body, { childList: true, subtree: true });
            })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def _inject_layout_css() -> None:
    st.markdown(
        """
        <style>
            html, body {
                overflow-y: auto !important;
                scroll-behavior: smooth;
            }
            .main .block-container {
                padding-top: 0.25rem !important;
                padding-bottom: 6.5rem !important; /* security space for sticky input */
                max-width: 100% !important;
            }
            div[data-testid="stMainBlockContainer"] {
                padding-top: 0.25rem !important;
                padding-bottom: 0rem !important;
                max-width: 100% !important;
            }
            .block-container {
                padding-top: 0.25rem !important;
                padding-bottom: 6.5rem !important;
            }
            .chat-bottom-guard {
                height: 60px;
                width: 100%;
                flex-shrink: 0;
            }
            header.stAppHeader {
                background: transparent;
                height: 1.8rem !important;
                min-height: 1.8rem !important;
            }
            div[data-testid="stDecoration"] {
                display: none;
            }
            div[data-testid="stChatInput"] {
                max-width: 100%;
                margin-top: 0rem !important;
                margin-bottom: 0rem !important;
                padding-top: 0.25rem !important;
                padding-bottom: 0.25rem !important;
            }
            .stChatFloatingInputContainer {
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                width: 100% !important;
                padding: 0.5rem 1rem 0.75rem 1rem !important;
                background: white !important;
                border-top: 1px solid #e6e6e6;
                z-index: 20 !important;
            }
            div[data-testid="stChatInput"] textarea,
            div[data-testid="stChatInput"] input {
                padding-top: 0.5rem !important;
                padding-bottom: 0.5rem !important;
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
            h1 {
                margin-top: 0rem !important;
                margin-bottom: 0.15rem !important;
                padding-top: 0rem !important;
                line-height: 1.1 !important;
            }
            div[data-testid="stCaptionContainer"] {
                margin-top: 0rem !important;
                margin-bottom: 0.35rem !important;
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


def _anonymous_session_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _ensure_anonymous_history() -> ChatHistory:
    """
    If no user/session is set, silently create an anonymous session named with
    the current UTC date/time so conversations are saved for analysis but not
    easily guessable by end users.
    """
    history: ChatHistory = st.session_state.setdefault("history", ChatHistory())
    if not history.user or not history.name:
        session_name = _anonymous_session_name()
        history = ChatHistory(user=ANONYMOUS_USER, name=session_name)
        history.save()
        st.session_state["history"] = history
    return history


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
        openai_client = OpenAIClient()
        st.session_state["completions"] = openai_client.chat_completions
        st.session_state["completion_params"] = openai_client.completion_params

    chat_history: ChatHistory = st.session_state["history"]

    # If the user has not logged in, silently fall back to an anonymous identity
    # in the background so the chat is usable immediately while still saving history.
    if not chat_history.user:
        chat_history = ChatHistory(user=ANONYMOUS_USER, name=_anonymous_session_name())
        chat_history.save()
        st.session_state["history"] = chat_history

    st.session_state["user"] = chat_history.user
    st.session_state["name"] = chat_history.name

    return chat_history


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("GlossaryAI")

        chat_history: ChatHistory = st.session_state["history"]
        is_generating = st.session_state.get("_generating", False)

        user_value = st.text_input(
            "Nom d'utilisateur",
            value=chat_history.user if chat_history.user != ANONYMOUS_USER else "",
            key="sidebar_user_value",
            placeholder="Votre identifiant",
            disabled=is_generating,
        )
        session_value = st.text_input(
            "Nom de la session",
            value=chat_history.name if chat_history.user != ANONYMOUS_USER else "",
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
            new_user = user_value.strip() or ANONYMOUS_USER
            new_session = session_value.strip() or _anonymous_session_name()
            _open_or_create_history(new_user, new_session)
            st.session_state["_reset_sidebar_inputs"] = True
            st.rerun()
            return

        if chat_history.user != ANONYMOUS_USER:
            st.write(f"**Utilisateur:** {chat_history.user}")
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

        if st.button(
            "Nouvelle session anonyme",
            key="new_anonymous_session",
            use_container_width=True,
            disabled=is_generating,
        ):
            _set_active_history(
                ChatHistory(user=ANONYMOUS_USER, name=_anonymous_session_name())
            )
            st.rerun()

        st.divider()
        st.subheader("Sources")

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
    st.session_state["_tags_initialized"] = False
    st.session_state["selected_tags"] = []


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
    chat_slot = st.container(border=False)
    with chat_slot:
        set_chatbox_layout()

        if user_input:
            try:
                await process_user_input(user_input)
            finally:
                _clear_generation_state()
            st.rerun()

    # Anchor element that the auto-scroll script targets
    st.markdown("<div id='chat-scroll-anchor' style='height:1px; width:100%;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-bottom-guard'></div>", unsafe_allow_html=True)
    _inject_scroll_js()


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

        header_container = st.container()
        with header_container:
            title_col, filter_col = st.columns([8, 1.8], vertical_alignment="top")
            with title_col:
                st.markdown("<h1 style='margin-top:-0.5rem; margin-bottom:-0.3rem; padding-top:0; padding-bottom:0;'>GlossaryAI</h1>", unsafe_allow_html=True)
                st.markdown("<p style='margin-top:0; margin-bottom:-0.5rem; padding-top:0; padding-bottom:0; color:#6b7280; font-size:0.9rem;'>Assistant vocabulaire, glossaire et textes juridiques</p>", unsafe_allow_html=True)
            with filter_col:
                with st.popover("🏷️ Sources", disabled=is_generating, use_container_width=True):
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
        chat_container.markdown("<div style='margin-top:-0.5rem;'></div>", unsafe_allow_html=True)

        with chat_container:
            pending = st.session_state.pop("_pending_input", None)
            if pending:
                asyncio.run(_render_chat_panel(user_input=pending))
            else:
                asyncio.run(_render_chat_panel())

        input_left, input_center, input_right = st.columns([1, 6, 1])
        with input_center:
            user_input = st.chat_input(
                "Génération en cours..." if is_generating else "Votre message",
                key="glossary_chat_input",
                disabled=is_generating,
            )

        if user_input and not is_generating:
            st.session_state["_generating"] = True
            st.session_state["_pending_input"] = user_input
            st.rerun()

        st.markdown("<div class='chat-bottom-guard'></div>", unsafe_allow_html=True)

    except Exception as e:
        _clear_generation_state()
        show_user_error(
            "Une erreur est survenue dans l'interface de chat.",
            details=str(e),
        )
        return
