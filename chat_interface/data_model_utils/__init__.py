from __future__ import annotations

# Standard library imports
from typing import Any, Optional, Literal
from pathlib import Path
from io import BytesIO
import asyncio
import logging

# Streamlit UI imports
import streamlit as st

# Local application imports
from clients import MCPClient
from .import_ttl import ttl_to_json
from .import_xml import xml_to_json
from .export_xml import json_to_xml
from .visualisation import get_image_bytes
from .export_ttl import jsonld_to_ttl_bytes


# ----------------------------------------------------------------------
# Config & logging
# ----------------------------------------------------------------------
CONTACT_EMAIL = "emilien.caudron@pwc.com"
LOGGER_NAME = "model_utils"
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
    Clear, actionable error message in the Streamlit UI.
    """
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


async def with_timeout(coro, seconds: float = 60.0, on_timeout_msg: str = ""):
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


# ----------------------------------------------------------------------
# File type detection
# ----------------------------------------------------------------------
def _detect_file_type(
    first_bytes: bytes,
    filename: Optional[str] = None,
) -> Optional[Literal["xml", "ttl", "xmi"]]:
    """
    Detect file type from filename first, then from content.
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix == ".ttl":
            return "ttl"
        if suffix == ".xmi":
            return "xmi"
        if suffix == ".xml":
            return "xml"

    sniff = first_bytes[:512].lstrip()

    if sniff.startswith(b"<") or sniff.startswith(b"<?xml"):
        return "xml"

    turtle_markers = (
        b"@prefix",
        b"@base",
        b"PREFIX ",
        b"BASE ",
        b"prefix ",
        b"base ",
    )
    if any(marker in sniff for marker in turtle_markers):
        return "ttl"

    return None


# ----------------------------------------------------------------------
# Export helpers
# ----------------------------------------------------------------------
def _get_model_name(default: str = "export") -> str:
    value = (st.session_state.get("name") or default).strip()
    return value or default


def _build_ttl_bytes(json_data: dict[str, Any]) -> bytes:
    """
    Build TTL bytes from a model if possible.
    Preference:
    1) ttl_raw
    2) ttl as JSON-LD transformed via jsonld_to_ttl_bytes
    """
    ttl_raw = json_data.get("ttl_raw")

    if isinstance(ttl_raw, bytes) and ttl_raw:
        return ttl_raw

    if isinstance(ttl_raw, str) and ttl_raw.strip():
        return ttl_raw.encode("utf-8")

    ttl_json = json_data.get("ttl")
    if ttl_json:
        ttl_bytes = jsonld_to_ttl_bytes(ttl_json)
        if isinstance(ttl_bytes, str):
            ttl_bytes = ttl_bytes.encode("utf-8")
        return ttl_bytes or b""

    logger.warning(
        "TTL export unavailable: missing ttl_raw and ttl. Available keys=%s",
        list(json_data.keys()),
    )
    return b""


def _build_xmi_bytes(json_data: dict[str, Any]) -> bytes:
    """
    Build XMI/XML bytes from a model if possible.
    Preference:
    1) json_data['xmi']
    2) json_data itself if it already looks like an XMI-like JSON model
    """
    if isinstance(json_data.get("xmi"), dict):
        export_source = json_data["xmi"]
    elif "elements" in json_data or "connectors" in json_data:
        export_source = {
            "elements": json_data.get("elements", []),
            "connectors": json_data.get("connectors", []),
        }
    else:
        logger.warning(
            "XMI export unavailable: missing xmi/elements/connectors. Available keys=%s",
            list(json_data.keys()),
        )
        return b""

    bytes_data = json_to_xml(export_source)
    if isinstance(bytes_data, str):
        bytes_data = bytes_data.encode("utf-8")
    return bytes_data or b""


# ----------------------------------------------------------------------
# Main upload & conversion entrypoint
# ----------------------------------------------------------------------
async def upload_xml(uploaded_file: BytesIO) -> dict[str, Any]:
    """
    Imports an XML/XMI or TTL file uploaded by the user and converts it to
    a JSON-compatible dictionary.

    If XML/XMI, adds a 'Generated' package for further modifications, and uploads
    to the MCP server.

    If TTL, preserves ttl_raw so that TTL export remains available later.
    """
    try:
        if "mcp_client" not in st.session_state or st.session_state["mcp_client"] is None:
            show_user_error("MCP client is not initialized in session.")
            return {}

        mcp_client: MCPClient = st.session_state["mcp_client"]

        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        filename = getattr(uploaded_file, "name", None)
        kind = _detect_file_type(file_bytes, filename)
        if kind is None:
            show_user_error(
                "Unsupported file format.",
                "Please upload an XMI/XML or TTL file."
            )
            return {}

        if kind in {"xml", "xmi"}:
            try:
                json_data = xml_to_json(BytesIO(file_bytes))
            except Exception as e:
                show_user_error("Failed to parse the XML/XMI file.", details=str(e))
                return {}

            try:
                elements = json_data.get("elements", [])
                if not elements:
                    show_user_error("Parsed XML has no elements.", "Ensure the XMI version is supported.")
                    return {}

                root_model_id = elements[0].get("ID")
                if not root_model_id:
                    show_user_error("Parsed XML root element is missing an ID.")
                    return {}

                st.session_state["package"] = root_model_id
                st.session_state["ID"] = mcp_client._generate_id()

                elements.append({
                    "name": "Generated",
                    "ID": st.session_state["ID"],
                    "type": "uml:Package",
                    "package": st.session_state["package"],
                    "tags": [],
                })
            except Exception as e:
                show_user_error("Could not append the 'Generated' package.", details=str(e))
                return {}

            json_data["source_format"] = "xmi"

        else:  # kind == "ttl"
            try:
                json_data = ttl_to_json(BytesIO(file_bytes))
            except Exception as e:
                show_user_error("Failed to parse the TTL file.", details=str(e))
                return {}

            json_data["source_format"] = "ttl"
            json_data["ttl_raw"] = file_bytes.decode("utf-8", errors="replace")

        try:
            async with mcp_client:
                model = await with_timeout(
                    mcp_client.upload_model({"model": json_data}),
                    seconds=60.0,
                    on_timeout_msg="Uploading the model took too long."
                )
                if model is None:
                    return {}
        except Exception as e:
            show_user_error("Uploading the model to the server failed.", details=str(e))
            return {}

        model = model or {}

        # Reinject source info if the server did not preserve it
        if kind == "ttl":
            model.setdefault("source_format", "ttl")
            model.setdefault("ttl_raw", json_data.get("ttl_raw", ""))

            if json_data.get("ttl") and not model.get("ttl"):
                model["ttl"] = json_data["ttl"]

        else:
            model.setdefault("source_format", "xmi")

        st.session_state["model"] = model
        return model

    except Exception as e:
        logger.exception("upload_xml failed: %s", e)
        show_user_error("A critical error occurred while uploading the file.", details=str(e))
        return {}


# ----------------------------------------------------------------------
# Download buttons
# ----------------------------------------------------------------------
def download_xml(json_data: dict[str, Any], disabled: bool = False) -> None:
    """
    Render two independent download buttons:
    - Exporter en TTL
    - Exporter en XMI

    A failure in one export must not prevent the other from rendering.
    """
    model_name = _get_model_name("export")

    ttl_bytes = b""
    xmi_bytes = b""
    ttl_error: Optional[str] = None
    xmi_error: Optional[str] = None

    try:
        ttl_bytes = _build_ttl_bytes(json_data)
    except Exception as e:
        logger.exception("TTL export preparation failed: %s", e)
        ttl_error = str(e)

    try:
        xmi_bytes = _build_xmi_bytes(json_data)
    except Exception as e:
        logger.exception("XMI export preparation failed: %s", e)
        xmi_error = str(e)

    col_ttl, col_xmi = st.columns(2, gap="small")

    with col_ttl:
        if disabled:
            st.button(
                "⏳ Export TTL indisponible",
                disabled=True,
                use_container_width=True,
                key="download_ttl_disabled_generating",
            )
        elif ttl_bytes:
            st.download_button(
                label="Exporter en TTL",
                data=ttl_bytes,
                file_name=f"{model_name}.ttl",
                mime="text/turtle",
                use_container_width=True,
                key="download_ttl_button",
            )
        else:
            st.button(
                "Export TTL indisponible",
                disabled=True,
                use_container_width=True,
                key="download_ttl_disabled_empty",
            )
            if ttl_error:
                st.caption(f"Erreur TTL : {ttl_error}")

    with col_xmi:
        if disabled:
            st.button(
                "⏳ Export XMI indisponible",
                disabled=True,
                use_container_width=True,
                key="download_xmi_disabled_generating",
            )
        elif xmi_bytes:
            st.download_button(
                label="Exporter en XMI",
                data=xmi_bytes,
                file_name=f"{model_name}.xmi",
                mime="application/xml",
                use_container_width=True,
                key="download_xmi_button",
            )
        else:
            st.button(
                "Export XMI indisponible",
                disabled=True,
                use_container_width=True,
                key="download_xmi_disabled_empty",
            )
            if xmi_error:
                st.caption(f"Erreur XMI : {xmi_error}")


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
def visualise(json_data: dict[str, Any]) -> None:
    """
    Visualizes a JSON model as an image in the Streamlit UI.
    """
    try:
        image_bytes = get_image_bytes(json_data)
        if not image_bytes:
            show_user_error("No image could be generated from the model.")
            return
        st.image(image_bytes)
    except Exception as e:
        show_user_error("Visualisation failed.", details=str(e))
