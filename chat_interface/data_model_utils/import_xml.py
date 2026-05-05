from __future__ import annotations

import re
from typing import Any, Optional
from xml.etree.ElementTree import Element, parse


# =====================================================================
#  Dynamic XMI Namespace Detection
# =====================================================================


def detect_namespaces(root: Element) -> dict[str, str]:
    """
    Dynamically detect XMI and UML namespaces used by the file.
    Supports both XMI-wrapped and direct UML formats.
    
    Returns:
        dict with keys 'xmi' and 'uml', e.g.:
        {
            'xmi': '{http://schema.omg.org/spec/XMI/2.1}',
            'uml': '{http://schema.omg.org/spec/UML/2.1}'
        }
    """
    ns = {'xmi': '', 'uml': ''}
    
    # Scan all xmlns declarations in root attributes
    for k, v in root.attrib.items():
        if 'xmlns' in k.lower():
            if 'xmi' in k.lower():
                ns['xmi'] = '{' + v.rstrip('/') + '}'
            elif 'uml' in k.lower():
                ns['uml'] = '{' + v.rstrip('/') + '}'
    
    # Fallback: extract from tag itself
    if root.tag.startswith('{'):
        tag_ns = root.tag.split('}')[0].lstrip('{')
        if 'XMI' in root.tag:
            ns['xmi'] = '{' + tag_ns.rstrip('/') + '}'
        elif 'Model' in root.tag:
            ns['uml'] = '{' + tag_ns.rstrip('/') + '}'
    
    # Default fallbacks
    if not ns['xmi']:
        ns['xmi'] = '{http://schema.omg.org/spec/XMI/2.1}'
    if not ns['uml']:
        ns['uml'] = '{http://schema.omg.org/spec/UML/2.1}'
    
    return ns


# Global namespace variables (overwritten dynamically)
NS_XMI = "{http://schema.omg.org/spec/XMI/2.1}"
NS_UML = "{http://schema.omg.org/spec/UML/2.1}"

# Global parent map for tree traversal
_PARENT_MAP = {}


# =====================================================================
#  Helper functions (safe wrappers)
# =====================================================================


def _attr(elem: Optional[Element], name: str, default=""):
    """Get attribute, trying both with and without namespace prefix. Returns empty string by default."""
    if elem is None:
        return default
    
    # Try exact match first
    val = elem.get(name)
    if val is not None:
        return val
    
    # Try without namespace prefix (for xmi:id, xmi:type, etc.)
    simple_name = name.split('}')[-1] if '}' in name else name
    for k, v in elem.attrib.items():
        if k.split('}')[-1] == simple_name:
            return v
    
    return default


def _find(elem: Optional[Element], path: str) -> Optional[Element]:
    return elem.find(path) if elem is not None else None


def _findall(elem: Optional[Element], path: str) -> list[Element]:
    return elem.findall(path) if elem is not None else []


def _build_parent_map(root: Element):
    """Build a map {child: parent} for the entire tree."""
    global _PARENT_MAP
    _PARENT_MAP = {c: p for p in root.iter() for c in p}


def _find_parent_package(elem: Element) -> str:
    """Find the parent package name by traversing up the tree. Returns empty string if not found."""
    current = elem
    while current is not None:
        current = _PARENT_MAP.get(current)
        if current is not None and 'Package' in (current.tag or ''):
            name = _attr(current, "name")
            return name if name else ""
    return ""


# =====================================================================
#  UML Model Navigation (handles both formats)
# =====================================================================


def _find_uml_model(root: Element) -> Optional[Element]:
    """
    Find the UML Model element, whether:
    - Wrapped: <xmi:XMI><uml:Model>
    - Direct: <uml:Model> as root
    """
    # Direct UML root
    if 'Model' in root.tag:
        return root
    
    # XMI-wrapped: search for uml:Model child
    for child in root:
        if 'Model' in child.tag:
            return child
    
    return None


def _get_packaged_elements(model: Element) -> list[Element]:
    """
    Extract all packagedElement nodes (UML 2.x standard structure).
    Works recursively for nested packages.
    """
    elements = []
    
    for child in model:
        if 'packagedElement' in child.tag:
            elements.append(child)
            # Recurse into nested packages
            elements.extend(_get_packaged_elements(child))
    
    return elements


# =====================================================================
#  ELEMENT EXTRACTORS (UML 2.x standard structure)
# =====================================================================


def _extract_uml_class(elem: Element) -> dict[str, Any]:
    """Extract UML Class with ownedAttribute structure."""
    package_name = _find_parent_package(elem)
    
    class_dict = {
        "name": _attr(elem, "name") or "UnnamedClass",
        "ID": _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Class",
        "visibility": _attr(elem, "visibility") or "public",
        "package": package_name,
        "attributes": [],
        "tags": []
    }
    
    # Extract attributes (ownedAttribute)
    for attr in elem.findall('.//{*}ownedAttribute'):
        attr_name = _attr(attr, "name")
        if not attr_name:
            continue  # Skip attributes without name
        
        attr_dict = {
            "name": attr_name,
            "visibility": _attr(attr, "visibility") or "public",
            "type": "",
            "lower_bounds": _attr(_find(attr, './/{*}lowerValue'), "value") or "0",
            "upper_bounds": _attr(_find(attr, './/{*}upperValue'), "value") or "1",
            "tags_attribute": []
        }
        
        # Extract type reference
        type_elem = _find(attr, './/{*}type')
        if type_elem is not None:
            type_href = _attr(type_elem, "href")
            if type_href and "PrimitiveTypes.xmi#" in type_href:
                # Extract primitive type name (e.g., "Integer", "String")
                attr_dict["type"] = type_href.split('#')[-1]
            else:
                attr_dict["type"] = type_href or _attr(type_elem, f"{NS_XMI}idref") or ""
        
        class_dict["attributes"].append(attr_dict)
    
    return class_dict


def _extract_uml_package(elem: Element) -> dict[str, Any]:
    """Extract UML Package."""
    parent_package = _find_parent_package(elem)
    
    return {
        "name": _attr(elem, "name") or "UnnamedPackage",
        "ID": _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Package",
        "visibility": _attr(elem, "visibility") or "public",
        "package": parent_package,
        "tags": []
    }


def _extract_uml_datatype(elem: Element) -> dict[str, Any]:
    """Extract UML DataType (same structure as Class)."""
    dt = _extract_uml_class(elem)
    dt["type"] = "uml:DataType"
    return dt


def _extract_uml_enumeration(elem: Element) -> dict[str, Any]:
    """Extract UML Enumeration with ownedLiteral."""
    package_name = _find_parent_package(elem)
    
    enum_dict = {
        "name": _attr(elem, "name") or "UnnamedEnumeration",
        "ID": _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or "",
        "type": "uml:Enumeration",
        "visibility": _attr(elem, "visibility") or "public",
        "package": package_name,
        "categories": [],
        "tags": []
    }
    
    # Extract literals
    for literal in elem.findall('.//{*}ownedLiteral'):
        lit_name = _attr(literal, "name")
        if lit_name:
            enum_dict["categories"].append(lit_name)
    
    return enum_dict


# =====================================================================
#  EA Legacy Format Extractors
# =====================================================================


def _get_package(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name") or "UnnamedPackage",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Package",
        "package": _attr(_find(elem, "model"), "package") or "",
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
    }


def _get_class(elem: Element) -> dict[str, Any]:
    class_dict = {
        "name": _attr(elem, "name") or "UnnamedClass",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Class",
        "package": _attr(_find(elem, "model"), "package") or "",
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
        "attributes": [],
    }

    for attribute in _findall(_find(elem, "attributes"), "attribute"):
        attr_name = _attr(attribute, "name")
        if not attr_name:
            continue
        
        attr_type = ""
        for prop in _findall(attribute, "properties"):
            t = _attr(prop, "type")
            if t:
                attr_type = t
                break
        
        lower_bound = ""
        upper_bound = ""
        for b in _findall(attribute, "bounds"):
            lower_bound = _attr(b, "lower") or "0"
            upper_bound = _attr(b, "upper") or "1"
            break
        
        attr_dict = {
            "name": attr_name,
            "type": attr_type,
            "lower_bounds": lower_bound,
            "upper_bounds": upper_bound,
            "tags_attribute": [
                {"name": _attr(t, "name"), "value": _attr(t, "value")}
                for t in _findall(_find(attribute, "tags"), "tag")
            ],
        }
        class_dict["attributes"].append(attr_dict)

    return class_dict


def _get_datatype(elem: Element) -> dict[str, Any]:
    dt_dict = {
        "name": _attr(elem, "name") or "UnnamedDataType",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:DataType",
        "package": _attr(_find(elem, "model"), "package") or "",
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
        "attributes": [],
    }

    for attribute in _findall(_find(elem, "attributes"), "attribute"):
        attr_name = _attr(attribute, "name")
        if not attr_name:
            continue
        
        attr_type = ""
        for prop in _findall(attribute, "properties"):
            t = _attr(prop, "type")
            if t:
                attr_type = t
                break
        
        lower_bound = ""
        upper_bound = ""
        for b in _findall(attribute, "bounds"):
            lower_bound = _attr(b, "lower") or "0"
            upper_bound = _attr(b, "upper") or "1"
            break
        
        attr_dict = {
            "name": attr_name,
            "type": attr_type,
            "lower_bounds": lower_bound,
            "upper_bounds": upper_bound,
            "tags_attribute": [
                {"name": _attr(t, "name"), "value": _attr(t, "value")}
                for t in _findall(_find(attribute, "tags"), "tag")
            ],
        }
        dt_dict["attributes"].append(attr_dict)

    return dt_dict


def _get_enumeration(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name") or "UnnamedEnumeration",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Enumeration",
        "package": _attr(_find(elem, "model"), "package") or "",
        "tags": [
            {"name": _attr(tag, "name"), "value": _attr(tag, "value")}
            for tag in _findall(_find(elem, "tags"), "tag")
        ],
        "categories": [
            _attr(a, "name")
            for a in _findall(_find(elem, "attributes"), "attribute")
            if _attr(a, "name")
        ],
    }


# =====================================================================
#  CONNECTOR EXTRACTORS
# =====================================================================


def _get_connector(connector: Element) -> dict[str, Any]:
    labels = _find(connector, "labels")

    return {
        "source_name": _attr(_find(_find(connector, "source"), "model"), "name") or "",
        "target_name": _attr(_find(_find(connector, "target"), "model"), "name") or "",
        "relationship": _attr(_find(connector, "properties"), "ea_type") or "",
        "lb": _attr(labels, "lb") or "",
        "lt": _attr(labels, "lt") or "",
        "rb": _attr(labels, "rb") or "",
        "rt": _attr(labels, "rt") or "",
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
    return [_get_connector(conn) for conn in root.iter("connector")]


# =====================================================================
#  ELEMENT COLLECTION (unified for both formats)
# =====================================================================


def _get_elements(root: Element) -> list[dict[str, Any]]:
    """
    Extract UML elements from BOTH formats:
    - EA XMI: <extension extender="Enterprise Architect"><elements><element>
    - Standard UML: <uml:Model><packagedElement xmi:type="uml:Class">
    """
    elements: list[dict[str, Any]] = []
    
    # ===== FORMAT 1: Enterprise Architect XMI (legacy structure) =====
    try:
        ea_elements = root[2][0].iter("element")
        for elem in ea_elements:
            t = _attr(elem, f"{NS_XMI}type")
            
            if t == "uml:Package":
                elements.append(_get_package(elem))
            elif t == "uml:Class":
                elements.append(_get_class(elem))
            elif t == "uml:DataType":
                elements.append(_get_datatype(elem))
            elif t == "uml:Enumeration":
                elements.append(_get_enumeration(elem))
        
        if elements:
            return elements
    except (IndexError, AttributeError):
        pass
    
    # ===== FORMAT 2: Standard UML 2.x (packagedElement structure) =====
    model = _find_uml_model(root)
    if model is None:
        raise ValueError("No UML Model found in XMI file")
    
    packaged_elements = _get_packaged_elements(model)
    
    for elem in packaged_elements:
        elem_type = _attr(elem, f"{NS_XMI}type")
        
        if "Package" in elem_type:
            elements.append(_extract_uml_package(elem))
        elif "Class" in elem_type:
            elements.append(_extract_uml_class(elem))
        elif "DataType" in elem_type:
            elements.append(_extract_uml_datatype(elem))
        elif "Enumeration" in elem_type:
            elements.append(_extract_uml_enumeration(elem))
    
    return elements


# =====================================================================
#  MAIN ENTRYPOINT — FULL PARSER
# =====================================================================


def xml_to_json(bytes_data) -> dict[str, Any]:
    """
    Convert any XML/XMI file into a JSON-compatible UML model.
    Supports:
    - XMI-wrapped format: <xmi:XMI><uml:Model>
    - Direct UML format: <uml:Model> as root
    - All XMI versions (1.x, 2.0, 2.1, 2.4, 2.5)
    - Enterprise Architect, Modelio, MagicDraw, Papyrus
    """
    global NS_XMI, NS_UML

    try:
        tree = parse(bytes_data)
        root = tree.getroot()
        
        # Build parent map for package resolution
        _build_parent_map(root)

        # Detect namespaces
        namespaces = detect_namespaces(root)
        NS_XMI = namespaces['xmi']
        NS_UML = namespaces['uml']

        # Extract data
        elements = _get_elements(root)
        connectors = _get_connectors(root)

        if not elements:
            raise ValueError(
                "No UML elements detected. File may be empty or use an unsupported XMI variant."
            )

        return {
            "elements": elements,
            "connectors": connectors,
        }

    except Exception as e:
        print(f"[ERROR] XMI parsing failed: {e}")
        raise
