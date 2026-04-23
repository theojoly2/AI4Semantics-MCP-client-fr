
from __future__ import annotations

import re
from typing import Any, Optional
from xml.etree.ElementTree import Element, parse

# =====================================================================
#  Dynamic XMI Namespace Detection
# =====================================================================

def detect_xmi_namespace(root: Element) -> str:
    """
    Dynamically detect the XMI namespace used by the file.
    Supports all XMI versions: 1.x, 2.0, 2.1, 2.4, 2.4.1, 2.5, 2.5.1, etc.

    Returns:
        e.g. "{http://schema.omg.org/spec/XMI/2.1}"
    """
    # ------- Try attributes like xmlns:xmi="..."
    for k, v in root.attrib.items():
        if "xmlns" in k.lower() and "xmi" in k.lower():
            return "{" + v.rstrip("/") + "}"

    # ------- Many XMI roots look like <XMI xmi.version="2.5" ...>
    if root.tag.startswith("{") and "XMI" in root.tag:
        ns = root.tag.split("}")[0].lstrip("{")
        return "{" + ns.rstrip("/") + "}"

    # ------- Regex fallback scanning raw XML text
    raw = Element.__str__(root)
    matches = re.findall(r"http[s]?://[^\"' >]+XMI[^\"' >]+", raw)
    if matches:
        return "{" + matches[0].rstrip("/") + "}"

    # ------- Final fallback (wildcard namespace)
    return "{http://schema.omg.org/spec/XMI}"


# This variable is dynamically overwritten by xml_to_json()
NS_XMI = "{http://schema.omg.org/spec/XMI}"


# =====================================================================
#  Helper functions (safe wrappers)
# =====================================================================

def _attr(elem: Optional[Element], name: str, default=None):
    return elem.get(name) if elem is not None else default

def _find(elem: Optional[Element], path: str) -> Optional[Element]:
    return elem.find(path) if elem is not None else None

def _findall(elem: Optional[Element], path: str) -> list[Element]:
    return elem.findall(path) if elem is not None else []


# =====================================================================
#  ELEMENT EXTRACTORS
# =====================================================================

def _get_package(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name"),
        "ID": _attr(elem, f"{NS_XMI}idref"),
        "type": _attr(elem, f"{NS_XMI}type"),
        "package": _attr(_find(elem, "model"), "package"),
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
    }


def _get_class(elem: Element) -> dict[str, Any]:
    class_dict = {
        "name": _attr(elem, "name"),
        "ID": _attr(elem, f"{NS_XMI}idref"),
        "type": _attr(elem, f"{NS_XMI}type"),
        "package": _attr(_find(elem, "model"), "package"),
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
        "attributes": [],
    }

    for attribute in _findall(_find(elem, "attributes"), "attribute"):

        attr_dict = {
            "name": _attr(attribute, "name"),
            "type": next(
                (_attr(prop, "type") for prop in _findall(attribute, "properties")),
                None,
            ),
            "lower_bounds": next(
                (_attr(b, "lower") for b in _findall(attribute, "bounds")),
                None,
            ),
            "upper_bounds": next(
                (_attr(b, "upper") for b in _findall(attribute, "bounds")),
                None,
            ),
            "tags_attribute": [
                {"name": _attr(t, "name"), "value": _attr(t, "value")}
                for t in _findall(_find(attribute, "tags"), "tag")
            ],
        }

        class_dict["attributes"].append(attr_dict)

    return class_dict


def _get_datatype(elem: Element) -> dict[str, Any]:
    dt_dict = {
        "name": _attr(elem, "name"),
        "ID": _attr(elem, f"{NS_XMI}idref"),
        "type": _attr(elem, f"{NS_XMI}type"),
        "package": _attr(_find(elem, "model"), "package"),
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
        "attributes": [],
    }

    for attribute in _findall(_find(elem, "attributes"), "attribute"):

        attr_dict = {
            "name": _attr(attribute, "name"),
            "type": next(
                (_attr(prop, "type") for prop in _findall(attribute, "properties")),
                None,
            ),
            "lower_bounds": next(
                (_attr(b, "lower") for b in _findall(attribute, "bounds")),
                None,
            ),
            "upper_bounds": next(
                (_attr(b, "upper") for b in _findall(attribute, "bounds")),
                None,
            ),
            "tags_attribute": [
                {"name": _attr(t, "name"), "value": _attr(t, "value")}
                for t in _findall(_find(attribute, "tags"), "tag")
            ],
        }

        dt_dict["attributes"].append(attr_dict)

    return dt_dict


def _get_enumeration(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name"),
        "ID": _attr(elem, f"{NS_XMI}idref"),
        "type": _attr(elem, f"{NS_XMI}type"),
        "package": _attr(_find(elem, "model"), "package"),
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
        "categories": [
            _attr(a, "name")
            for a in _findall(_find(elem, "attributes"), "attribute")
        ],
    }


# =====================================================================
#  CONNECTOR EXTRACTORS
# =====================================================================

def _get_connector(connector: Element) -> dict[str, Any]:
    labels = _find(connector, "labels")

    return {
        "source_name": _attr(_find(_find(connector, "source"), "model"), "name"),
        "target_name": _attr(_find(_find(connector, "target"), "model"), "name"),
        "relationship": _attr(_find(connector, "properties"), "ea_type"),

        # label positions
        "lb": _attr(labels, "lb"),
        "lt": _attr(labels, "lt"),
        "rb": _attr(labels, "rb"),
        "rt": _attr(labels, "rt"),

        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(connector, "tags"), "tag")
        ],

        "tags_source": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(_find(connector, "source"), "tags"), "tag")
        ],

        "tags_target": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(_find(connector, "target"), "tags"), "tag")
        ],
    }


def _get_connectors(root: Element) -> list[dict[str, Any]]:
    return [
        _get_connector(conn)
        for conn in root.iter("connector")
    ]


# =====================================================================
#  ELEMENT & CONNECTOR COLLECTION
# =====================================================================

def _get_elements(root: Element) -> list[dict[str, Any]]:
    """
    Extract UML packages, classes, datatypes, and enumerations
    from any XMI version or vendor (EA, MagicDraw, Cameo, etc.).
    """

    # Many EA exports: root[2][0].iter("element")
    # We allow fallback to ANY 'element' tag
    try:
        elems = root[2][0].iter("element")
    except Exception:
        elems = root.iter("element")

    elements: list[dict[str, Any]] = []

    for elem in elems:
        t = _attr(elem, f"{NS_XMI}type") or ""

        if t == "uml:Package":
            elements.append(_get_package(elem))
        elif t == "uml:Class":
            elements.append(_get_class(elem))
        elif t == "uml:DataType":
            elements.append(_get_datatype(elem))
        elif t == "uml:Enumeration":
            elements.append(_get_enumeration(elem))
        else:
            # Unknown element types are simply skipped (safe behavior)
            continue

    return elements


# =====================================================================
#  MAIN ENTRYPOINT — FULL PARSER
# =====================================================================

def xml_to_json(bytes_data) -> dict[str, Any]:
    """
    Convert any XML/XMI file into a JSON-compatible UML model.
    Supports all XMI versions and EA/MagicDraw/Cameo/Papyrus variations.
    """
    global NS_XMI

    try:
        tree = parse(bytes_data)
        root = tree.getroot()

        # --- Detect & apply the namespace ----
        NS_XMI = detect_xmi_namespace(root)

        # --- Extract data ----
        elements = _get_elements(root)
        connectors = _get_connectors(root)

        if not elements:
            raise ValueError(
                "No UML elements were detected. This XMI variant may require a custom parser."
            )

        return {
            "elements": elements,
            "connectors": connectors,
        }

    except Exception as e:
        print(f"[ERROR] XMI parsing failed: {e}")
        raise
