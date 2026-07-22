# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from json import loads, dumps
from os import environ
from typing import Any, Mapping, Optional, Set


import streamlit as st
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition


from fastmcp import Client

"""
GlossaryAI client MCP wrapper.
"""


CONTACT_EMAIL = "theo.joly2@developpement-durable.gouv.fr"
LOGGER_NAME = "fastmcp_ui"


logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)


def _normalize_list_arg(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"none", "null"}:
            return []
    return []


def _normalize_str_arg(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"none", "null"}:
            return default
        return cleaned
    if value is None:
        return default
    return str(value).strip() or default


def _normalize_int_arg(value: Any, default: int = 10) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"none", "null"}:
            return default
        try:
            return int(cleaned)
        except ValueError:
            return default
    return default


def _normalize_bool_arg(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def safe_json_loads(text: Optional[str]) -> Any:
    if not text:
        return {}
    try:
        return loads(text)
    except Exception as e:
        logger.exception("JSON parsing failed: %s", e)
        return {}


def show_user_error(
    title: str,
    details: Optional[str] = None,
    contact_email: str = CONTACT_EMAIL,
):
    try:
        with st.status(title, expanded=True, state="error") as status:
            if details:
                st.write(details)
            st.write(
                f"**What you can do now:**\n"
                f"1) Review your inputs and correct the bug if possible.\n"
                f"2) Re-launch the UI.\n"
                f"3) If the error keeps happening, contact the tech team at **{contact_email}**."
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
            f"3) If the error keeps happening, contact the tech team at **{contact_email}**."
        )


async def with_timeout(coro, seconds: float = 30.0, *, on_timeout_msg: str = ""):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        show_user_error(
            "The operation timed out.",
            details=on_timeout_msg or "The server took too long to respond.",
        )
        return None


def make_progress_handler():
    placeholder = st.empty()

    async def _handler(progress: float, total: float | None, message: str | None) -> None:
        if total and total > 0:
            pct = (progress / total) * 100
            if pct >= 100:
                placeholder.empty()
            else:
                placeholder.progress(
                    int(pct),
                    text=f"Progress: {pct:.1f}%  {message or ''}".strip(),
                )
        else:
            placeholder.progress(0, text=f"Step {int(progress)}…  {message or ''}".strip())
        logger.debug("Progress: %s / %s — %s", progress, total, message)

    def _clear():
        placeholder.empty()

    return _handler, _clear


async def sampling_handler(messages, params, context) -> str:
    try:
        openai_messages: list[dict[str, str]] = []
        if getattr(params, "systemPrompt", None):
            openai_messages.append({"role": "system", "content": params.systemPrompt})

        for m in messages:
            text = getattr(m.content, "text", str(m.content))
            openai_messages.append({"role": m.role, "content": text})

        if "completions" not in st.session_state or st.session_state["completions"] is None:
            raise RuntimeError("OpenAI AsyncCompletions client is not initialized.")

        llm_model = environ.get("LLM_MODEL")
        if not llm_model:
            raise RuntimeError("Missing LLM_MODEL environment variable.")

        completions: AsyncCompletions = st.session_state["completions"]
        completion_params = st.session_state.get("completion_params", {})

        resp = await with_timeout(
            completions.create(
                model=str(llm_model),
                messages=openai_messages,
                temperature=getattr(params, "temperature", 0.0) or 0.0,
                max_tokens=getattr(params, "maxTokens", 512) or 512,
                stop=getattr(params, "stopSequences", None) or None,
                stream=False,
                extra_body=completion_params.get("extra_body"),
            ),
            seconds=3000.0,
            on_timeout_msg="The LLM did not respond in time. Please try again.",
        )

        if resp is None:
            return ""

        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        logger.exception("sampling_handler failed: %s", e)
        show_user_error("We hit an error while generating a response.", details=str(e))
        return ""


class MCPClient:
    """
    FastMCP client wrapper for GlossaryAI.
    """

    EXPOSED_TOOLS: Set[str] = {
        "retrieve_documents",
        "get_available_tags",
        "plan_workflow_with_tools",
        "resolve_links",
        "compare_concepts",
    }

    def __init__(
        self,
        state: Mapping[str, Any] = {},
        server: str | Any = "https://ai4sem-mcp-server.azurewebsites.net/mcp",
    ) -> None:
        self.state = state
        self.server = server
        self.exit_stack = AsyncExitStack()
        self.client: Client | None = None
        self.tool_results: dict[str, Any] = {}

    async def __aenter__(self) -> "MCPClient":
        try:
            if self.client is None:
                self.client = Client(
                    self.server,
                    sampling_handler=sampling_handler,
                )
            await self.exit_stack.enter_async_context(self.client)
            return self
        except Exception as e:
            logger.exception("__aenter__ failed: %s", e)
            show_user_error("Could not open a session with the MCP server.", details=str(e))
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            logger.exception("__aexit__ failed: %s", e)
            show_user_error("Failed to close the MCP session cleanly.", details=str(e))

    async def _call_with_progress(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float = 3000.0,
    ) -> Any:
        progress_handler, clear_progress = make_progress_handler()

        progress_client = Client(
            self.server,
            sampling_handler=sampling_handler,
            progress_handler=progress_handler,
        )
        try:
            async with progress_client:
                result = await with_timeout(
                    progress_client.call_tool(tool_name, arguments),
                    seconds=timeout,
                    on_timeout_msg=f"Tool '{tool_name}' timed out. Please try again.",
                )
            return result
        finally:
            clear_progress()

    async def tools(self) -> list[ChatCompletionToolParam]:
        assert self.client is not None, "MCP client not initialized"
        try:
            tools = await with_timeout(
                self.client.list_tools(),
                seconds=30.0,
                on_timeout_msg="Listing server tools timed out.",
            )
            if tools is None:
                return []

            exposed: list[ChatCompletionToolParam] = []
            for t in tools:
                if t.name in MCPClient.EXPOSED_TOOLS:
                    exposed.append(
                        ChatCompletionToolParam(
                            type="function",
                            function=FunctionDefinition(
                                name=t.name,
                                description=t.description or "",
                                parameters=t.inputSchema,
                            ),
                        )
                    )
            return exposed
        except Exception as e:
            logger.exception("tools() failed: %s", e)
            show_user_error("Failed to load tools from the server.", details=str(e))
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        logger.info("[CLIENT] call_tool triggered for tool: %s", name)
        logger.info("[CLIENT] Arguments received: %s", arguments)

        payload = {"tool_name": name, "tool_arguments": arguments, "tool_results": ""}

        try:
            if name not in MCPClient.EXPOSED_TOOLS:
                raise ValueError(
                    f"Tool '{name}' is not exposed. Allowed: {MCPClient.EXPOSED_TOOLS}"
                )

            tool_func = getattr(self, f"_{name}", None)
            if not callable(tool_func):
                raise AttributeError(f"No client wrapper implemented for '{name}'")

            payload = await tool_func(payload)

        except Exception as e:
            logger.exception("call_tool failed: %s", e)
            show_user_error(f"Running tool '{name}' failed.", details=str(e))
            payload["tool_results"] = "Error occurred while calling the tool."

        logger.info("[CLIENT] Payload: %s", payload)
        return dumps(payload)

    async def _retrieve_documents(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"

        arguments = payload.get("tool_arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        search_terms = arguments.get("search_terms")
        if not search_terms:
            show_user_error("Missing required argument 'search_terms' for retrieve_documents.")
            payload["tool_results"] = {}
            return payload

        call_args = {
            "search_terms": search_terms,
            "limit": _normalize_int_arg(arguments.get("limit"), default=10),
            "return_full_document": _normalize_bool_arg(
                arguments.get("return_full_document"),
                default=True,
            ),
            "tags": self.state.get("selected_tags", []),
            "document_filter": _normalize_str_arg(arguments.get("document_filter"), default=""),
        }
        if not call_args["document_filter"]:
            call_args.pop("document_filter")

        payload["tool_arguments"] = call_args

        try:
            result = await self._call_with_progress("retrieve_documents", call_args)
            if result is None:
                payload["tool_results"] = {}
                return payload

            content = getattr(result, "content", [])
            if content and getattr(content[0], "type", "") == "text":
                raw_text = getattr(content[0], "text", "") or ""
                parsed = safe_json_loads(raw_text)
                payload["tool_results"] = parsed if parsed is not None else {"raw": raw_text}
            else:
                payload["tool_results"] = {}

        except Exception as e:
            logger.exception("_retrieve_documents failed: %s", e)
            show_user_error("Could not retrieve documents.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _get_available_tags(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"

        try:
            result = await self._call_with_progress("get_available_tags", {})
            if result is None:
                payload["tool_results"] = []
                return payload

            content = getattr(result, "content", [])
            if content and getattr(content[0], "type", "") == "text":
                raw_text = getattr(content[0], "text", "") or ""
                parsed = safe_json_loads(raw_text)
                payload["tool_results"] = parsed if parsed is not None else {"raw": raw_text}
            else:
                payload["tool_results"] = {}

        except Exception as e:
            logger.exception("_get_available_tags failed: %s", e)
            show_user_error("Could not fetch available tags.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _resolve_links(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"

        arguments = payload.get("tool_arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        chunks = arguments.get("chunks")
        if not chunks or not isinstance(chunks, list):
            show_user_error("Missing or invalid required argument 'chunks' for resolve_links.")
            payload["tool_results"] = {}
            return payload

        call_args = {
            "chunks": chunks,
            "max_depth": _normalize_int_arg(arguments.get("max_depth"), default=1),
        }
        payload["tool_arguments"] = call_args

        try:
            result = await self._call_with_progress("resolve_links", call_args)
            if result is None:
                payload["tool_results"] = {}
                return payload

            content = getattr(result, "content", [])
            if content and getattr(content[0], "type", "") == "text":
                raw_text = getattr(content[0], "text", "") or ""
                parsed = safe_json_loads(raw_text)
                payload["tool_results"] = parsed if parsed is not None else {"raw": raw_text}
            else:
                payload["tool_results"] = {}

        except Exception as e:
            logger.exception("_resolve_links failed: %s", e)
            show_user_error("Could not resolve links.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _compare_concepts(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"

        arguments = payload.get("tool_arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        terms = arguments.get("terms")
        if not terms or not isinstance(terms, list):
            show_user_error("Missing or invalid required argument 'terms' for compare_concepts.")
            payload["tool_results"] = {}
            return payload

        call_args = {
            "terms": terms,
            "limit": _normalize_int_arg(arguments.get("limit"), default=5),
        }
        payload["tool_arguments"] = call_args

        try:
            result = await self._call_with_progress("compare_concepts", call_args)
            if result is None:
                payload["tool_results"] = {}
                return payload

            content = getattr(result, "content", [])
            if content and getattr(content[0], "type", "") == "text":
                raw_text = getattr(content[0], "text", "") or ""
                parsed = safe_json_loads(raw_text)
                payload["tool_results"] = parsed if parsed is not None else {"raw": raw_text}
            else:
                payload["tool_results"] = {}

        except Exception as e:
            logger.exception("_compare_concepts failed: %s", e)
            show_user_error("Could not compare concepts.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _plan_workflow_with_tools(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        if "user_question" not in arguments or not arguments["user_question"]:
            show_user_error("Missing 'user_question' for plan_workflow_with_tools.")
            return payload

        call_args = {
            "user": _normalize_str_arg(self.state.get("user"), default=""),
            "name": _normalize_str_arg(self.state.get("name"), default=""),
            "user_question": arguments["user_question"],
            "allowed_executor_tools": sorted(list(self.EXPOSED_TOOLS)),
            "observations": arguments.get("observations") or [],
            "max_steps": arguments.get("max_steps", 5),
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("plan_workflow_with_tools", call_args)
            if res is None:
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None:
                payload["tool_results"] = res.data if isinstance(res.data, dict) else {}
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            payload["tool_results"] = safe_json_loads(text)
        except Exception as e:
            logger.exception("_plan_workflow_with_tools failed: %s", e)
            show_user_error("Could not plan the workflow.", details=str(e))
            payload["tool_results"] = {}

        return payload
