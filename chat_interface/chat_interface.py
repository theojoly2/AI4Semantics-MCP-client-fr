from __future__ import annotations

from typing import Any, Optional
import asyncio
import logging
from json import loads

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from clients import OpenAIClient, MCPClient
from chat_history import ChatHistory
from .chat_logic import set_chatbox_layout, process_user_input, show_user_error


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


def _fetch_tags(mcp_client: MCPClient) -> list[str]:
    tags_state = st.session_state.setdefault("_tags_state", {
        "initialized": False,
        "tags": [],
        "selected": [],
        "error": None,
    })

    if tags_state["initialized"]:
        return tags_state["selected"]

    async def _load():
        async with mcp_client:
            return await mcp_client.get_available_tags()

    try:
        raw_tags = asyncio.run(with_timeout(_load(), seconds=30.0))
        if raw_tags is None:
            raw_tags = []
    except Exception as e:
        logger.exception("Failed to load tags: %s", e)
        tags_state["error"] = str(e)
        raw_tags = []

    tags = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, dict) and "tag" in item:
                tags.append(item["tag"])
            elif isinstance(item, str):
                tags.append(item)

    tags_state["initialized"] = True
    tags_state["tags"] = tags
    tags_state["selected"] = tags[:]  # Tous les tags sélectionnés par défaut
    st.session_state["selected_tags"] = tags[:]
    return tags_state["selected"]


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("GlossaryAI")

        user = st.text_input("Utilisateur", value=st.session_state.get("user", ""), key="user_input")
        session = st.text_input("Session", value=st.session_state.get("name", "default"), key="session_input")

        if st.button("Démarrer / Recharger", use_container_width=True):
            st.session_state["user"] = user
            st.session_state["name"] = session
            st.session_state["history"] = ChatHistory(user=user, session=session)
            st.session_state["completions"] = None
            st.session_state["messages"] = []
            st.rerun()

        if st.session_state.get("user") and st.session_state.get("name"):
            st.caption(f"Session : {st.session_state['user']} / {st.session_state['name']}")

        st.divider()
        st.subheader("Filtres par source")

        tags_state = st.session_state.get("_tags_state", {"tags": [], "selected": []})
        available_tags = tags_state.get("tags", [])
        selected_tags = tags_state.get("selected", [])

        if available_tags:
            new_selection = st.multiselect(
                "Sources à inclure dans la recherche",
                options=available_tags,
                default=selected_tags,
                key="tag_multiselect",
            )
            tags_state["selected"] = new_selection
            st.session_state["selected_tags"] = new_selection
        else:
            st.caption("Aucune source disponible. Vérifiez l'indexation.")


async def _run_user_input(user_input: str, mcp_client: MCPClient, slot: DeltaGenerator) -> None:
    with slot.container():
        await process_user_input(user_input, mcp_client=mcp_client)


def data_modelling_chat_tab(server: str) -> None:
    _inject_layout_css()
    set_chatbox_layout()

    if "user" not in st.session_state:
        st.session_state["user"] = ""
    if "name" not in st.session_state:
        st.session_state["name"] = "default"
    if "history" not in st.session_state or st.session_state["history"] is None:
        st.session_state["history"] = ChatHistory(
            user=st.session_state["user"],
            name=st.session_state["name"],
        )
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "mcp_client" not in st.session_state or st.session_state["mcp_client"] is None:
        st.session_state["mcp_client"] = MCPClient(
            state=st.session_state,
            server=server,
        )
    if "completions" not in st.session_state or st.session_state["completions"] is None:
        openai_client = OpenAIClient()
        st.session_state["completions"] = openai_client.chat_completions
        st.session_state["completion_params"] = openai_client.completion_params

    mcp_client: MCPClient = st.session_state["mcp_client"]
    _fetch_tags(mcp_client)
    _render_sidebar()

    history: ChatHistory = st.session_state["history"]

    st.title("GlossaryAI")
    st.caption("Assistant vocabulaire, glossaire et textes juridiques")

    # Zone de chat
    chat_slot = st.container()

    # Afficher l'historique
    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with chat_slot.chat_message(role):
            st.write(content)
            if role == "assistant" and "sources" in msg:
                _render_sources(msg["sources"])

    user_input = st.chat_input("Posez votre question (terme, définition, article juridique...)")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        history.add_user_message(user_input)

        with chat_slot.chat_message("user"):
            st.write(user_input)

        with chat_slot.chat_message("assistant"):
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                st.markdown("*GlossaryAI réfléchit...*")

            try:
                asyncio.run(_run_user_input(user_input, mcp_client, st))
            except Exception as e:
                logger.exception("Chat processing failed: %s", e)
                show_user_error("Une erreur est survenue lors du traitement.", details=str(e))
            finally:
                thinking_placeholder.empty()


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for src in sources:
            document = src.get("filename") or src.get("document_name") or "Document inconnu"
            article = src.get("article") or ""
            concept = src.get("concept_uri") or ""
            term = src.get("term") or ""
            score = src.get("score")
            text = src.get("text", "")

            meta_parts = [f"**{document}**"]
            if term:
                meta_parts.append(f"Terme : *{term}*")
            if article:
                meta_parts.append(f"Article : *{article}*")
            if concept:
                meta_parts.append(f"Concept : `{concept}`")
            if score is not None:
                meta_parts.append(f"Score : {score:.3f}")

            meta = " | ".join(meta_parts)
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-meta">{meta}</div>
                    <div class="source-text">{text[:800]}{'...' if len(text) > 800 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
