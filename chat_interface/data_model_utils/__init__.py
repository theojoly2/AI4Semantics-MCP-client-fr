
from __future__ import annotations

# Standard library imports
from typing import Any, Optional, Literal
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
CONTACT_EMAIL = "emilien.caudron@pwc.com"  # Dummy email per your request
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
def _detect_file_type(first_bytes: bytes) -> Optional[Literal["xml", "ttl", "xmi"]]:
    """
    Heuristically detect file type from initial bytes.
    """
    sniff = first_bytes[:256].lstrip()
    # XML/XMI often starts with '<' or '<?xml'
    if sniff.startswith(b"<") or sniff.startswith(b"<?xml"):
        return "xml"
    # Turtle hints
    turtle_markers = (b"@prefix", b"@base", b"PREFIX ", b"BASE ", b"@prefix ")
    if any(m in sniff for m in turtle_markers):
        return "ttl"
    return None


# ----------------------------------------------------------------------
# Main upload & conversion entrypoint
# ----------------------------------------------------------------------
async def upload_xml(uploaded_file: BytesIO) -> dict[str, Any]:
    """
    Imports an XML or TTL file uploaded by the user and converts it to a JSON-compatible dictionary.
    If XML/XMI, adds a 'Generated' package for further modifications, and uploads to the MCP server.

    Args:
        uploaded_file: The uploaded XML or TTL file (as a BytesIO-like object).
    Returns:
        The server-confirmed model (dict), or {} on error.
    """
    try:
        # Ensure MCP client is available
        if "mcp_client" not in st.session_state or st.session_state["mcp_client"] is None:
            show_user_error("MCP client is not initialized in session.")
            return {}

        mcp_client: MCPClient = st.session_state["mcp_client"]

        # Read bytes & reset pointer
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        # Detect type
        kind = _detect_file_type(file_bytes)
        if kind is None:
            show_user_error("Unsupported file format.",
                            "Please upload an XMI/XML or TTL file.")
            return {}

        # Parse based on type
        if kind == "xml":
            try:
                json_data = xml_to_json(BytesIO(file_bytes))
            except Exception as e:
                show_user_error("Failed to parse the XML/XMI file.", details=str(e))
                return {}

            # Add a Generated package to allow further edits in UI
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

        else:  # kind == "ttl"
            try:
                json_data = ttl_to_json(BytesIO(file_bytes))
            except Exception as e:
                show_user_error("Failed to parse the TTL file.", details=str(e))
                return {}

        # Upload the model to the MCP server
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

        return model or {}

    except Exception as e:
        logger.exception("upload_xml failed: %s", e)
        show_user_error("A critical error occurred while uploading the file.", details=str(e))
        return {}


def download_xml(json_data: dict[str, Any]) -> None:
    """
    Converts a JSON model to XML and provides a download button in the Streamlit UI.
    """
    try:
        if "elements" in json_data.keys():
            bytes_data = json_to_xml(json_data)  # expected to return bytes or str
            if isinstance(bytes_data, str):
                bytes_data = bytes_data.encode("utf-8")

            st.download_button(
                label="Download model",
                data=bytes_data or b"",
                file_name="export.xml",
                mime="application/xml",
            )
        elif "ttl" in json_data.keys():
            ttl_content = jsonld_to_ttl_bytes(json_data.get("ttl", ""))

            st.download_button(
                label="Download model",
                data=ttl_content or b"",
                file_name="export.ttl",
                mime="text/turtle",
            )
        else: 
            raise show_user_error("No downloadable model found in the provided data.")
    except Exception as e:
        show_user_error("Failed to generate download for XML.", details=str(e))


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
