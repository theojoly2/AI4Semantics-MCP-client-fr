# -*- coding: utf-8 -*-
from __future__ import annotations
from time import time

"""
Streamlit app with robust error handling for FastMCP + OpenAI async completions.

What users will see on error:
- A clear, actionable message:
  1) Review inputs and fix the issue if possible
  2) Re-launch the UI
  3) If errors persist, contact the tech team at CONTACT_EMAIL

Replace CONTACT_EMAIL with your real support address when ready.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from json import loads, dumps
from os import environ
from typing import Any, Callable, Mapping, Optional, Set

import streamlit as st
from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from uuid import uuid4

from fastmcp import Client


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CONTACT_EMAIL = "emilien.caudron@pwc.com"
LOGGER_NAME = "fastmcp_ui"

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)


# ----------------------------------------------------------------------
# UI error helpers
# ----------------------------------------------------------------------
def safe_json_loads(text: Optional[str]) -> dict:
    """Parse JSON safely, returning {} on failure and logging the error."""
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
    """
    Show a clear, actionable error in the Streamlit UI.
    """
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
    """Run an async coroutine with a timeout and surface a user message if it times out."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        show_user_error(
            "The operation timed out.",
            details=on_timeout_msg or "The server took too long to respond.",
        )
        return None


# ----------------------------------------------------------------------
# Progress bar factory
#
# The core insight: st.progress / st.empty widgets are positioned in the
# Streamlit DOM at the moment they are *created*, not when they are updated.
# Storing a widget in session_state and reusing it on the next tool call
# causes it to render at its original (old) position — high up in the chat.
#
# Solution: create a fresh st.empty() placeholder right before each tool
# call, build a one-shot progress handler closure around it, and pass that
# closure to the FastMCP Client for that call only.  When the call finishes
# (or errors), clear the placeholder so it disappears cleanly.
# ----------------------------------------------------------------------

def make_progress_handler():
    """
    Create a fresh progress bar placeholder anchored to the *current* DOM
    position (i.e. just below the latest chat message) and return:
      - the async handler to pass to the FastMCP Client
      - a cleanup callable to call after the tool finishes

    Usage (in MCPClient):
        progress_handler, clear_progress = make_progress_handler()
        client = Client(..., progress_handler=progress_handler)
        ...
        clear_progress()
    """
    # st.empty() is created right now, at the current render position.
    # Each call to make_progress_handler() produces a *new* placeholder.
    placeholder = st.empty()

    async def _handler(progress: float, total: float | None, message: str | None) -> None:
        if total and total > 0:
            pct = (progress / total) * 100
            if pct >= 100:
                # Done — wipe the placeholder immediately.
                placeholder.empty()
            else:
                placeholder.progress(
                    int(pct),
                    text=f"Progress: {pct:.1f}%  {message or ''}".strip(),
                )
        else:
            # Indeterminate: just show the raw counter.
            placeholder.progress(0, text=f"Step {int(progress)}…  {message or ''}".strip())

        logger.debug("Progress: %s / %s — %s", progress, total, message)

    def _clear():
        placeholder.empty()

    return _handler, _clear


# ----------------------------------------------------------------------
# Sampling handler (server -> client LLM request)
# ----------------------------------------------------------------------

async def sampling_handler(messages, params, context) -> str:
    """
    Bridges MCP sampling -> OpenAI Chat Completions.
    Called automatically when the server triggers ctx.sample(...).
    """
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

        resp = await with_timeout(
            completions.create(
                model=str(llm_model),
                messages=openai_messages,
                temperature=getattr(params, "temperature", 0.0) or 0.0,
                max_tokens=getattr(params, "maxTokens", 512) or 512,
                stop=getattr(params, "stopSequences", None) or None,
                stream=False,
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


# ----------------------------------------------------------------------
# MCPClient
# ----------------------------------------------------------------------
class MCPClient:
    """
    Prefect FastMCP-compatible client wrapper.

    Key change vs. the original: the FastMCP Client is now instantiated
    fresh for each tool call (inside _call_with_progress) so that its
    progress_handler is bound to a placeholder created at the *current*
    DOM position, ensuring the progress bar always appears below the
    latest chat message.
    """

    EXPOSED_TOOLS: Set[str] = {
        "retrieve_documents",
        "add_class",
        "add_attribute",
        "add_connector",
        "plan_workflow_with_tools",
        "metadata_checker",
        "reuse_check",
        "validator_check",
        "style_guide_check",
    }

    RESERVED_ARGUMENTS: Set[str] = {
        "user", "name", "package", "ID",
    }

    def __init__(
        self,
        state: Mapping[str, Any] = {},
        server: str | Any = "https://ai4sem-mcp-server.azurewebsites.net/mcp",
    ) -> None:
        self.state = state
        self.server = server
        self.exit_stack = AsyncExitStack()
        # self.client is used for non-progress calls (tools(), read_model, etc.)
        self.client: Client | None = None
        self.tool_results: dict[str, Any] = {}

    async def __aenter__(self) -> "MCPClient":
        try:
            if self.client is None:
                # The base client uses no progress handler; progress is injected
                # per-call via _call_with_progress below.
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

    # ------------------------------------------------------------------
    # Core helper: run a tool call with a freshly positioned progress bar
    # ------------------------------------------------------------------
    async def _call_with_progress(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float = 3000.0,
    ) -> Any:
        """
        Create a fresh progress bar placeholder at the current DOM position,
        open a short-lived FastMCP Client session with that handler, execute
        the tool call, then clean up the placeholder.

        This guarantees the progress bar always appears directly below the
        latest chat message, regardless of how many tool calls have been made.
        """
        progress_handler, clear_progress = make_progress_handler()

        # Open a *new* Client session with the freshly created progress handler.
        # This is inexpensive — FastMCP sessions are lightweight HTTP connections.
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
            # Always clear the placeholder, even if the call raised.
            clear_progress()

    # ------------------------------------------------------------------
    # Tool exposure for OpenAI
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Generic tool dispatcher
    # ------------------------------------------------------------------
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        logger.info("[CLIENT] call_tool triggered for tool: %s", name)
        logger.info("[CLIENT] Arguments received: %s", arguments)

        payload = {"tool_name": name, "tool_arguments": arguments, "tool_results": ""}

        try:
            if name not in MCPClient.EXPOSED_TOOLS:
                raise ValueError(f"Tool '{name}' is not exposed. Allowed: {MCPClient.EXPOSED_TOOLS}")

            tool_func: Optional[Callable[[dict[str, Any]], Any]] = getattr(self, f"_{name}", None)
            if not callable(tool_func):
                raise AttributeError(f"No client wrapper implemented for '{name}'")

            payload = await tool_func(payload)

        except Exception as e:
            logger.exception("call_tool failed: %s", e)
            show_user_error(f"Running tool '{name}' failed.", details=str(e))
            payload["tool_results"] = "Error occurred while calling the tool."

        logger.info("[CLIENT] Payload: %s", payload)
        return dumps(payload)

    # ------------------------------------------------------------------
    # Tool wrappers — all now delegate the actual call to _call_with_progress
    # ------------------------------------------------------------------

    async def _retrieve_documents(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        if "search_terms" not in arguments or not arguments["search_terms"]:
            show_user_error("Missing required argument 'search_terms' for retrieve_documents.")
            return payload

        call_args = {
            "search_terms": arguments["search_terms"],
            "vocabularies": arguments.get("vocabularies", []),
            "limit": arguments.get("limit", 10),
        }
        payload["tool_arguments"] = call_args

        try:
            result = await self._call_with_progress("retrieve_documents", call_args)
            if result is None:
                payload["tool_results"] = {}
                return payload

            content = getattr(result, "content", [])
            if content and getattr(content[0], "type", "") == "text":
                payload["tool_results"] = safe_json_loads(getattr(content[0], "text", "")) or {}
            else:
                payload["tool_results"] = {}
        except Exception as e:
            logger.exception("_retrieve_documents failed: %s", e)
            show_user_error("Could not retrieve documents.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _add_class(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        required = {"title", "definition", "usage_note"}
        missing = [arg for arg in required if arg not in arguments or not arguments[arg]]
        if missing:
            show_user_error(f"Missing required arguments for add_class: {missing}.")
            return payload

        call_args = {
            "user": self.state.get("user"),
            "name": self.state.get("name"),
            "package": self.state.get("package", ""),
            "ID": self._generate_id(),
            **{k: arguments[k] for k in required},
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("add_class", call_args)
            if res is None:
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None:
                payload["tool_results"] = res.data if isinstance(res.data, dict) else {}
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            payload["tool_results"] = safe_json_loads(text)
        except Exception as e:
            logger.exception("_add_class failed: %s", e)
            show_user_error("Could not add the class.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _add_attribute(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        required = {"class_name", "attr_label", "attr_definition", "attr_uri"}
        missing = [arg for arg in required if arg not in arguments or not arguments[arg]]
        if missing:
            show_user_error(f"Missing required arguments for add_attribute: {missing}.")
            return payload

        call_args = {
            "user": self.state.get("user"),
            "name": self.state.get("name"),
            "class_name": arguments["class_name"],
            "attr_label": arguments["attr_label"],
            "attr_definition": arguments["attr_definition"],
            "attr_uri": arguments["attr_uri"] or f"http://example.com/{arguments['attr_label']}",
            "attr_usage_note": arguments.get("attr_usage_note", ""),
            "attr_type": arguments.get("attr_type", ""),
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("add_attribute", call_args)
            if res is None:
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None:
                payload["tool_results"] = res.data if isinstance(res.data, dict) else {}
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            payload["tool_results"] = safe_json_loads(text)
        except Exception as e:
            logger.exception("_add_attribute failed: %s", e)
            show_user_error("Could not add the attribute.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _add_connector(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        required = {"source_name", "target_name", "rel_label", "rel_definition", "rel_uri", "relationship"}
        missing = [arg for arg in required if arg not in arguments or not arguments[arg]]
        if missing:
            show_user_error(f"Missing required arguments for add_connector: {missing}.")
            return payload

        call_args = {
            "user": self.state.get("user"),
            "name": self.state.get("name"),
            "source_name": arguments["source_name"],
            "target_name": arguments["target_name"],
            "rel_label": arguments["rel_label"],
            "rel_definition": arguments["rel_definition"],
            "rel_uri": arguments["rel_uri"] or f"http://example.com/{arguments['rel_label']}",
            "relationship": arguments["relationship"],
            "rb": arguments.get("rb"),
            "rt": arguments.get("rt"),
            "rel_usage_note": arguments.get("rel_usage_note", ""),
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("add_connector", call_args)
            if res is None:
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None:
                payload["tool_results"] = res.data if isinstance(res.data, dict) else {}
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            payload["tool_results"] = safe_json_loads(text)
        except Exception as e:
            logger.exception("_add_connector failed: %s", e)
            show_user_error("Could not add the connector.", details=str(e))
            payload["tool_results"] = {}

        return payload

    async def _plan_workflow_with_tools(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        if "user_question" not in arguments or not arguments["user_question"]:
            show_user_error("Missing 'user_question' for plan_workflow_with_tools.")
            return payload

        call_args = {
            "user": self.state.get("user") or "",
            "name": self.state.get("name") or "",
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

    async def _metadata_checker(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        call_args = {
            "user": self.state.get("user"),
            "name": self.state.get("name"),
            "target_names": arguments.get("target_names") or [],
            "check_instruction": arguments.get("check_instruction") or "",
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("metadata_checker", call_args)
            logger.info("metadata_checker results: %s", res)

            if res is None:
                self.tool_results["metadata_checker"] = {}
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None and isinstance(res.data, dict):
                self.tool_results["metadata_checker"] = res.data
                payload["tool_results"] = res.data
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            parsed = safe_json_loads(text)
            self.tool_results["metadata_checker"] = parsed
            payload["tool_results"] = parsed
        except Exception as e:
            logger.exception("_metadata_checker failed: %s", e)
            show_user_error("Metadata check failed.", details=str(e))
            self.tool_results["metadata_checker"] = {}
            payload["tool_results"] = {}

        return payload

    async def _reuse_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        call_args = {
            "user": self.state.get("user"),
            "name": self.state.get("name"),
            "vocabularies": arguments.get("vocabularies") or [],
            "n_documents": arguments.get("n_documents", 10),
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("reuse_check", call_args)
            if res is None:
                self.tool_results["reuse_check"] = {}
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None and isinstance(res.data, dict):
                self.tool_results["reuse_check"] = res.data
                payload["tool_results"] = res.data
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            parsed = safe_json_loads(text)
            self.tool_results["reuse_check"] = parsed
            payload["tool_results"] = parsed
        except Exception as e:
            logger.exception("_reuse_check failed: %s", e)
            show_user_error("Reuse check failed.", details=str(e))
            self.tool_results["reuse_check"] = {}
            payload["tool_results"] = {}

        return payload

    async def _validator_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        call_args = {
            "user": self.state.get("user"),
            "name": self.state.get("name"),
            "validation_server": arguments.get(
                "validation_server",
                "https://www.itb.ec.europa.eu/shacl/semicstyleguide/api/validate",
            ),
            "output_format": "text/turtle",
            "validation_version": arguments.get("validation_version", "owl"),
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("validator_check", call_args)
            if res is None:
                self.tool_results["validator_check"] = {}
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None and isinstance(res.data, dict):
                self.tool_results["validator_check"] = res.data
                payload["tool_results"] = res.data
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            parsed = safe_json_loads(text)
            self.tool_results["validator_check"] = parsed
            payload["tool_results"] = parsed
        except Exception as e:
            logger.exception("_validator_check failed: %s", e)
            show_user_error("Validator check failed.", details=str(e))
            self.tool_results["validator_check"] = {}
            payload["tool_results"] = {}

        return payload

    async def _style_guide_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        arguments = payload.get("tool_arguments", {})

        validator_check = self.tool_results.get("validator_check") or arguments.get("validator_check", {})
        metadata_checks = self.tool_results.get("metadata_checker") or arguments.get("metadata_checker", {})
        reuse_checks    = self.tool_results.get("reuse_check")      or arguments.get("reuse_check", {})

        call_args = {
            "validator_check": validator_check or {},
            "metadata_checks": metadata_checks or {},
            "reuse_checks":    reuse_checks    or {},
        }
        payload["tool_arguments"] = call_args

        try:
            res = await self._call_with_progress("style_guide_check", call_args)
            if res is None:
                payload["tool_results"] = {}
                return payload

            if getattr(res, "data", None) is not None and isinstance(res.data, dict):
                payload["tool_results"] = res.data
                return payload

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            payload["tool_results"] = safe_json_loads(text)
        except Exception as e:
            logger.exception("_style_guide_check failed: %s", e)
            show_user_error("Style guide check failed.", details=str(e))
            payload["tool_results"] = {}

        return payload

    # ------------------------------------------------------------------
    # Direct UI helpers (not exposed to the LLM)
    # ------------------------------------------------------------------

    async def upload_model(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        model_payload = arguments.get("model")
        if not model_payload:
            show_user_error("No 'model' payload provided to upload_model.")
            return {}

        try:
            res = await with_timeout(
                self.client.call_tool(
                    "upload_model",
                    {
                        "user": self.state.get("user"),
                        "name": self.state.get("name"),
                        "model": model_payload,
                    },
                ),
                seconds=3000.0,
                on_timeout_msg="Uploading model timed out. Please try again.",
            )
            if res is None:
                return {}

            if getattr(res, "data", None) is not None and isinstance(res.data, dict):
                return res.data

            text = "".join(getattr(b, "text", "") for b in (res.content or []))
            return safe_json_loads(text)
        except Exception as e:
            logger.exception("upload_model failed: %s", e)
            show_user_error("Uploading the model failed.", details=str(e))
            return {}

    async def read_model(self) -> dict[str, Any]:
        assert self.client is not None, "MCP client not initialized"
        user = self.state.get("user")
        name = self.state.get("name")
        if not user or not name:
            show_user_error("Missing 'user' or 'name' in client state for read_model.")
            return {}

        try:
            contents = await with_timeout(
                self.client.read_resource(f"resource://model/{user}/{name}"),
                seconds=3000.0,
                on_timeout_msg="Reading model timed out. Please try again.",
            )
            if not contents:
                return {}

            text = getattr(contents[0], "text", None)
            return safe_json_loads(text)
        except Exception as e:
            logger.exception("read_model failed: %s", e)
            show_user_error("Reading the model failed.", details=str(e))
            return {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_id() -> str:
        return f"EAID_{str(uuid4()).upper().replace('-', '_')}"