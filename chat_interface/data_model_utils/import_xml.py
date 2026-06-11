from __future__ import annotations

import io
import re
from collections import defaultdict
from typing import Any, Optional
from xml.etree.ElementTree import Element, iterparse, parse

# =====================================================================
#  Namespaces & Default Configurations
# =====================================================================

DEFAULT_XMI_NS = "{http://schema.omg.org/spec/XMI/2.1}"
DEFAULT_UML_NS = "{http://schema.omg.org/spec/UML/2.1}"

NS_XMI = DEFAULT_XMI_NS
NS_UML = DEFAULT_UML_NS

_PARENT_MAP: dict[Element, Element] = {}


# =====================================================================
#  Generic Utilities & Safe Extractors
# =====================================================================

def _local_name(value: str) -> str:
    """Helper to strip namespace prefixes and curly braces."""
    if not value:
        return ""
    if value.startswith("{"):
        return value.split("}", 1)[1]
    if ":" in value:
        return value.split(":", 1)[1]
    return value


def detect_namespaces(source: Any, root: Optional[Element] = None) -> dict[str, str]:
    """Detect default namespaces for standard UML2 parsers."""
    ns = {
        "xmi": DEFAULT_XMI_NS,
        "uml": DEFAULT_UML_NS,
    }
    stream = source
    reset_pos: Optional[int] = None
    should_reset = False

    try:
        if isinstance(source, (bytes, bytearray)):
            stream = io.BytesIO(source)
        elif hasattr(source, "seek") and hasattr(source, "tell"):
            reset_pos = source.tell()
            source.seek(0)
            should_reset = True

        for _, node in iterparse(stream, events=("start-ns",)):
            prefix, uri = node
            uri = (uri or "").rstrip("/")
            if prefix == "xmi" and uri:
                ns["xmi"] = "{" + uri + "}"
            elif prefix == "uml" and uri:
                ns["uml"] = "{" + uri + "}"
    except Exception:
        pass
    finally:
        if should_reset and hasattr(source, "seek") and reset_pos is not None:
            source.seek(reset_pos)

    if root is not None:
        root_local = _local_name(root.tag)
        if root_local == "XMI" and root.tag.startswith("{"):
            root_ns = root.tag.split("}", 1)[0].lstrip("{").rstrip("/")
            if root_ns:
                ns["xmi"] = "{" + root_ns + "}"
        elif root_local == "Model" and root.tag.startswith("{"):
            root_ns = root.tag.split("}", 1)[0].lstrip("{").rstrip("/")
            if root_ns:
                ns["uml"] = "{" + root_ns + "}"

    return ns


def _clean_type_name(raw_type: str) -> str:
    """
    Nettoie les noms de types propriétaires d'Enterprise Architect (ex: EAJava_String, EASQL_Real)
    pour revenir sur des types primitifs UML standards.
    """
    if not raw_type:
        return ""

    cleaned = raw_type
    match = re.match(r"^EA(Java|SQL|Oracle|C\+\+|C#|Python|VB|Delphi|PHP)_(.*)", raw_type, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(2)

    lower_clean = cleaned.lower()
    if lower_clean in ("string", "str", "text", "varchar", "char"):
        return "String"
    if lower_clean in ("int", "integer", "long", "short"):
        return "Integer"
    if lower_clean in ("real", "float", "double", "decimal", "number"):
        return "Real"
    if lower_clean in ("bool", "boolean"):
        return "Boolean"
    if lower_clean in ("date", "datetime", "timestamp"):
        return "DateTime"

    return cleaned


def _get_xmi_type(elem: Optional[Element]) -> str:
    """Safely extract the UML structural type."""
    if elem is None:
        return ""
    val = elem.get(f"{NS_XMI}type") or elem.get("xmi:type")
    if val:
        return val
    val = elem.get("type")
    if val and val.startswith("uml:"):
        return val
    for k, v in elem.attrib.items():
        if (k.endswith("}type") or k.endswith(":type")) and v.startswith("uml:"):
            return v
    return ""


def _get_xmi_id(elem: Optional[Element]) -> str:
    """Safely extract the element ID."""
    if elem is None:
        return ""
    val = elem.get(f"{NS_XMI}id") or elem.get("xmi:id") or elem.get("id")
    if val:
        return val
    for k, v in elem.attrib.items():
        if k.endswith("}id") or k.endswith(":id"):
            return v
    return ""


def _attr(elem: Optional[Element], name: str, default: str = "") -> str:
    if elem is None:
        return default
    val = elem.get(name)
    if val is not None:
        return val
    wanted = _local_name(name)
    for k, v in elem.attrib.items():
        if _local_name(k) == wanted:
            return v
    return default


def _children(elem: Optional[Element], local_name: str) -> list[Element]:
    if elem is None:
        return []
    return [child for child in list(elem) if _local_name(child.tag) == local_name]


def _child(elem: Optional[Element], local_name: str) -> Optional[Element]:
    if elem is None:
        return None
    for child in list(elem):
        if _local_name(child.tag) == local_name:
            return child
    return None


def _build_parent_map(root: Element) -> None:
    global _PARENT_MAP
    _PARENT_MAP = {child: parent for parent in root.iter() for child in parent}


def _find_parent_class_id(elem: Element) -> str:
    current = elem
    while current is not None:
        current = _PARENT_MAP.get(current)
        if current is None:
            return ""
        if _get_xmi_type(current) in ("uml:Class", "uml:DataType"):
            return _get_xmi_id(current)
    return ""


def _parse_semantic_body(text: str) -> list[dict[str, str]]:
    text = (text or "").strip()
    if not text:
        return []

    # Sécurisation : forcer les sauts de ligne sur les mots clés sémantiques.
    text = re.sub(
        r"(?i)(?<!\n)\b(URI:|Label:|Definition:|Usage note:|Usage_note:|Usagenote:|Referenced:|Connector ID:|Connector_id:)",
        r"\n\1",
        text
    )

    mapping = {
        "uri": "uri",
        "label": "label-en",
        "definition": "definition-en",
        "usage note": "usageNote-en",
        "usage_note": "usageNote-en",
        "usagenote": "usageNote-en",
        "referenced": "referenced",
        "connector id": "connector_id",
        "connector_id": "connector_id",
    }

    tags: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        norm_key = key.strip().lower()
        norm_key = re.sub(r"\s+", " ", norm_key)
        norm_key = mapping.get(norm_key, "")
        if not norm_key:
            continue
        clean_value = value.strip()
        md_link = re.match(r"^\[([^\]]+)\]\(([^)]+)\)$", clean_value)
        if md_link and norm_key in {"uri", "label-en"}:
            clean_value = md_link.group(2) if norm_key == "uri" else md_link.group(1)
        tags.append({"name": norm_key, "value": clean_value})
    return tags


def _is_ea_xml(root: Element) -> bool:
    """
    Detect if the XML structure is a direct Enterprise Architect export.
    """
    for elem in root.iter():
        if _local_name(elem.tag) == "Extension" and elem.get("extender") == "Enterprise Architect":
            return True
    return False


# =====================================================================
#  STRATEGY A: STANDARD UML2 & PAPYRUS PARSER (CLASSIQUE)
# =====================================================================

def _tag_value(tags: list[dict[str, Any]] | None, name: str) -> str:
    name_lower = name.lower()
    for tag in tags or []:
        if (tag.get("name") or "").strip().lower() == name_lower:
            return (tag.get("value") or "").strip()
    return ""


def _comment_tags(elem: Optional[Element]) -> list[dict[str, str]]:
    if elem is None:
        return []
    comments = _children(elem, "ownedComment")
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for comment in comments:
        body = _child(comment, "body")
        body_text = (body.text if body is not None else "") or ""
        for tag in _parse_semantic_body(body_text):
            key = (tag["name"], tag["value"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(tag)
    return merged


def _is_package_element(elem: Optional[Element]) -> bool:
    if elem is None:
        return False
    elem_type = _get_xmi_type(elem)
    if "Package" in elem_type:
        return True
    if _local_name(elem.tag) == "packagedElement" and "Package" in elem_type:
        return True
    return False


def _find_parent_package_id(elem: Element) -> str:
    current = elem
    while current is not None:
        current = _PARENT_MAP.get(current)
        if current is None:
            return ""
        if _is_package_element(current):
            return _get_xmi_id(current)
    return ""


def _find_uml_model(root: Element) -> Optional[Element]:
    if _local_name(root.tag) == "Model":
        return root
    for elem in root.iter():
        if _local_name(elem.tag) == "Model":
            return elem
    return None


def _get_packaged_elements(container: Optional[Element]) -> list[Element]:
    if container is None:
        return []
    out: list[Element] = []
    for child in list(container):
        if _local_name(child.tag) == "packagedElement":
            out.append(child)
            out.extend(_get_packaged_elements(child))
    return out


def _normalize_href_or_ref(value: str) -> str:
    if not value:
        return ""
    if "#" in value:
        return value.split("#")[-1]
    return value


def _extract_type_reference(elem: Optional[Element]) -> str:
    """Strictly extracts the underlying data/class type, avoiding xmi:type confusion."""
    if elem is None:
        return ""

    direct_type = elem.get("type")
    if direct_type and direct_type not in ("uml:Property", "uml:Attribute", "uml:Port", "uml:Association"):
        return _normalize_href_or_ref(direct_type)

    type_elem = _child(elem, "type")
    if type_elem is not None:
        href = type_elem.get("href")
        if href:
            return _normalize_href_or_ref(href)
        return (
            _attr(type_elem, f"{NS_XMI}idref")
            or type_elem.get("idref")
            or _get_xmi_id(type_elem)
            or ""
        )

    for candidate in ("datatype", "classifier"):
        val = elem.get(candidate)
        if val:
            return _normalize_href_or_ref(val)

    return ""


def _extract_multiplicity(elem: Optional[Element]) -> str:
    if elem is None:
        return ""
    lower = _attr(_child(elem, "lowerValue"), "value")
    upper = _attr(_child(elem, "upperValue"), "value")
    if not lower and not upper:
        return ""
    lower = lower or "1"
    upper = upper or "1"
    if lower == upper:
        return lower
    return f"{lower}..{upper}"


def _extract_standard_attribute(attribute: Element) -> Optional[dict[str, Any]]:
    attr_name = _attr(attribute, "name")
    if not attr_name:
        return None

    attr_type = _extract_type_reference(attribute)
    if "PrimitiveTypes.xmi#" in attr_type:
        attr_type = attr_type.split("#")[-1]

    attr_type = _clean_type_name(attr_type)
    tags = _comment_tags(attribute)

    return {
        "name": attr_name,
        "visibility": _attr(attribute, "visibility") or "public",
        "type": attr_type,
        "uri": _tag_value(tags, "uri") or None,
        "lower_bounds": _attr(_child(attribute, "lowerValue"), "value"),
        "upper_bounds": _attr(_child(attribute, "upperValue"), "value"),
        "tags_attribute": tags,
    }


def _extract_uml_package(elem: Element) -> dict[str, Any]:
    tags = _comment_tags(elem)
    return {
        "name": _attr(elem, "name") or "UnnamedPackage",
        "ID": _get_xmi_id(elem),
        "uri": _tag_value(tags, "uri") or None,
        "type": _get_xmi_type(elem) or "uml:Package",
        "visibility": _attr(elem, "visibility") or "public",
        "package": _find_parent_package_id(elem),
        "tags": tags,
    }


def _extract_uml_class(elem: Element) -> dict[str, Any]:
    tags = _comment_tags(elem)
    class_dict = {
        "name": _attr(elem, "name") or "UnnamedClass",
        "ID": _get_xmi_id(elem),
        "uri": _tag_value(tags, "uri") or None,
        "type": _get_xmi_type(elem) or "uml:Class",
        "visibility": _attr(elem, "visibility") or "public",
        "package": _find_parent_package_id(elem),
        "attributes": [],
        "tags": tags,
    }

    for attr in _children(elem, "ownedAttribute"):
        attr_dict = _extract_standard_attribute(attr)
        if attr_dict is not None:
            class_dict["attributes"].append(attr_dict)

    return class_dict


def _extract_uml_datatype(elem: Element) -> dict[str, Any]:
    dt = _extract_uml_class(elem)
    dt["type"] = "uml:DataType"
    return dt


def _extract_uml_enumeration(elem: Element) -> dict[str, Any]:
    tags = _comment_tags(elem)
    enum_dict = {
        "name": _attr(elem, "name") or "UnnamedEnumeration",
        "ID": _get_xmi_id(elem),
        "uri": _tag_value(tags, "uri") or None,
        "type": "uml:Enumeration",
        "visibility": _attr(elem, "visibility") or "public",
        "package": _find_parent_package_id(elem),
        "categories": [],
        "tags": tags,
    }
    for literal in _children(elem, "ownedLiteral"):
        lit_name = _attr(literal, "name")
        if lit_name:
            enum_dict["categories"].append(lit_name)
    return enum_dict


def _get_standard_elements(root: Element) -> list[dict[str, Any]]:
    model = _find_uml_model(root)
    if model is None:
        return []

    elements: list[dict[str, Any]] = []
    for elem in _get_packaged_elements(model):
        elem_type = _get_xmi_type(elem)

        if "Package" in elem_type:
            elements.append(_extract_uml_package(elem))
        elif "Class" in elem_type:
            elements.append(_extract_uml_class(elem))
        elif "DataType" in elem_type:
            elements.append(_extract_uml_datatype(elem))
        elif "Enumeration" in elem_type:
            elements.append(_extract_uml_enumeration(elem))

    return elements


def _get_standard_connectors(
    root: Element,
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model = _find_uml_model(root)
    if model is None:
        return []

    element_by_id, elements_by_name = _build_element_indexes(elements)
    property_index = _build_property_index(root)

    connectors: list[dict[str, Any]] = []

    for elem in _get_packaged_elements(model):
        elem_type = _get_xmi_type(elem)
        if elem_type and "Association" in elem_type:
            conn = _get_standard_association_connector(
                elem, element_by_id, elements_by_name, property_index
            )
            if conn is not None:
                connectors.append(conn)

    connectors.extend(
        _get_standard_generalization_connectors(model, element_by_id, elements_by_name)
    )

    return _dedupe_connectors(connectors)


def _build_element_indexes(elements: list[dict[str, Any]]):
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for elem in elements:
        elem_id = elem.get("ID") or ""
        elem_name = elem.get("name") or ""
        if elem_id:
            by_id[elem_id] = elem
        if elem_name:
            by_name[elem_name].append(elem)
    return by_id, by_name


def _resolve_endpoint(
    *,
    ref_id: str = "",
    ref_name: str = "",
    element_by_id: Optional[dict[str, dict[str, Any]]] = None,
    elements_by_name: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> tuple[str, str]:
    element_by_id = element_by_id or {}
    elements_by_name = elements_by_name or {}

    if ref_id and ref_id in element_by_id:
        elem = element_by_id[ref_id]
        return elem.get("ID", ref_id), elem.get("name", ref_name)
    if ref_name:
        matches = elements_by_name.get(ref_name, [])
        if matches:
            elem = matches[0]
            return elem.get("ID", ""), elem.get("name", ref_name)
    if ref_id:
        return ref_id, ref_name
    return "", ref_name


def _dedupe_connectors(connectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for conn in connectors:
        relation_name = (
            conn.get("name")
            or _tag_value(conn.get("tags_target", []), "label-en")
            or _tag_value(conn.get("tags", []), "label-en")
            or ""
        )
        relation_uri = (
            conn.get("uri")
            or _tag_value(conn.get("tags_target", []), "uri")
            or _tag_value(conn.get("tags", []), "uri")
            or ""
        )
        key = (
            conn.get("source_id", ""),
            conn.get("target_id", ""),
            conn.get("source_name", ""),
            conn.get("target_name", ""),
            conn.get("relationship", ""),
            relation_name,
            relation_uri,
        )
        if key in seen:
            continue
        seen.add(key)

        # Sécurisation : on remonte l'URI trouvée à la racine pour garantir l'export et l'accès modèle
        if not conn.get("uri") and relation_uri:
            conn["uri"] = relation_uri
        if not conn.get("name") and relation_name:
            conn["name"] = relation_name

        out.append(conn)
    return out


def _build_property_index(root: Element) -> dict[str, Element]:
    prop_index: dict[str, Element] = {}
    for elem in root.iter():
        local = _local_name(elem.tag)
        if local in {"ownedEnd", "ownedAttribute", "property"}:
            elem_id = _get_xmi_id(elem)
            if elem_id:
                prop_index[elem_id] = elem
    return prop_index


def _association_relationship(assoc: Element, end_1: Element, end_2: Element) -> str:
    aggregations = {
        assoc.get("aggregation") or "",
        end_1.get("aggregation") or "",
        end_2.get("aggregation") or "",
    }
    if "composite" in aggregations:
        return "Composition"
    if "shared" in aggregations:
        return "Aggregation"
    return "Association"


def _association_endpoints(
    assoc: Element,
    property_index: dict[str, Element],
) -> tuple[Optional[Element], Optional[Element]]:
    owned_ends = _children(assoc, "ownedEnd")
    if len(owned_ends) >= 2:
        return owned_ends[0], owned_ends[1]

    member_end_ids = (assoc.get("memberEnd") or _attr(assoc, "memberEnd") or "").split()
    member_ends = [property_index[mid] for mid in member_end_ids if mid in property_index]
    if len(member_ends) >= 2:
        return member_ends[0], member_ends[1]

    return None, None


def _get_standard_association_connector(
    assoc: Element,
    element_by_id: dict[str, dict[str, Any]],
    elements_by_name: dict[str, list[dict[str, Any]]],
    property_index: dict[str, Element],
) -> Optional[dict[str, Any]]:
    end_1, end_2 = _association_endpoints(assoc, property_index)
    if end_1 is None or end_2 is None:
        return None

    t1 = _extract_type_reference(end_1) or _find_parent_class_id(end_1)
    t2 = _extract_type_reference(end_2) or _find_parent_class_id(end_2)

    source_id, source_name = _resolve_endpoint(
        ref_id=t2, element_by_id=element_by_id, elements_by_name=elements_by_name
    )
    target_id, target_name = _resolve_endpoint(
        ref_id=t1, element_by_id=element_by_id, elements_by_name=elements_by_name
    )

    if not source_id and not target_id:
        return None

    assoc_tags = _comment_tags(assoc)
    connector_id = _tag_value(assoc_tags, "connector_id")
    relation_name = _tag_value(assoc_tags, "label-en") or _attr(assoc, "name") or ""
    relation_uri = _tag_value(assoc_tags, "uri")

    tags = []
    if connector_id:
        tags.append({"name": "connector_id", "value": connector_id})

    target_tags = list(assoc_tags)
    if connector_id and not _tag_value(target_tags, "connector_id"):
        target_tags.append({"name": "connector_id", "value": connector_id})

    return {
        "connector_id": connector_id,
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_name,
        "target_name": target_name,
        "relationship": _association_relationship(assoc, end_1, end_2),
        "name": relation_name,
        "uri": relation_uri,
        "lb": _extract_multiplicity(end_2),
        "lt": _attr(end_2, "name") or "",
        "rb": _extract_multiplicity(end_1),
        "rt": _attr(end_1, "name") or "",
        "tags": tags,
        "tags_source": [],
        "tags_target": target_tags,
    }


def _get_standard_generalization_connectors(
    model: Optional[Element],
    element_by_id: dict[str, dict[str, Any]],
    elements_by_name: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if model is None:
        return []

    connectors: list[dict[str, Any]] = []
    for elem in _get_packaged_elements(model):
        elem_type = _get_xmi_type(elem)
        if not elem_type or not any(t in elem_type for t in ("Class", "DataType", "Enumeration")):
            continue

        source_id = _get_xmi_id(elem)
        source_name = _attr(elem, "name") or ""

        for gen in _children(elem, "generalization"):
            general_id = _attr(gen, "general") or _attr(gen, f"{NS_XMI}idref") or _attr(gen, "idref")
            target_id, target_name = _resolve_endpoint(
                ref_id=general_id,
                ref_name="",
                element_by_id=element_by_id,
                elements_by_name=elements_by_name,
            )
            connectors.append({
                "source_id": source_id,
                "target_id": target_id,
                "source_name": source_name,
                "target_name": target_name,
                "relationship": "Generalization",
                "name": "subClassOf",
                "uri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                "lb": "",
                "lt": "",
                "rb": "",
                "rt": "",
                "tags": [],
                "tags_source": [],
                "tags_target": [],
            })
    return connectors


# =====================================================================
#  STRATEGY B: ENTERPRISE ARCHITECT DIRECT XML/XMI PARSER
# =====================================================================

def _ea_attr(elem: Optional[Element], name: str, default=None):
    if elem is None:
        return default
    val = elem.get(name)
    if val is not None:
        return val
    wanted = _local_name(name)
    for k, v in elem.attrib.items():
        if _local_name(k) == wanted:
            return v
    return default


def _ea_find(elem: Optional[Element], path: str) -> Optional[Element]:
    if elem is None:
        return None
    path_parts = path.split("/")
    curr = elem
    for part in path_parts:
        found = None
        for child in list(curr):
            if _local_name(child.tag) == part:
                found = child
                break
        if found is None:
            return None
        curr = found
    return curr


def _ea_findall(elem: Optional[Element], path: str) -> list[Element]:
    if elem is None:
        return []
    path_parts = path.split("/")
    curr_elements = [elem]
    for part in path_parts:
        next_elements = []
        for curr in curr_elements:
            next_elements.extend([child for child in list(curr) if _local_name(child.tag) == part])
        curr_elements = next_elements
    return curr_elements


def _ea_extract_semantic_tags(elem: Optional[Element], is_class_level: bool = False) -> list[dict[str, str]]:
    """Extract tags dynamically from standard tags OR hidden EA documentation fields."""
    if elem is None:
        return []

    tags: list[dict[str, str]] = []
    for t in _ea_findall(_ea_find(elem, "tags"), "tag"):
        n = _ea_attr(t, "name")
        v = _ea_attr(t, "value")
        if n:
            tags.append({"name": n, "value": v})

    doc_text = ""
    if is_class_level:
        props = _ea_find(elem, "properties")
        if props is not None:
            doc_text = _ea_attr(props, "documentation") or ""
    else:
        doc = _ea_find(elem, "documentation")
        if doc is not None:
            doc_text = _ea_attr(doc, "value") or ""

    if doc_text:
        parsed = _parse_semantic_body(doc_text)
        existing_names = {t["name"].lower() for t in tags}
        for pt in parsed:
            if pt["name"].lower() not in existing_names:
                tags.append(pt)

    return tags


def _ea_get_package(elem: Element) -> dict[str, Any]:
    tags = _ea_extract_semantic_tags(elem, is_class_level=True)
    return {
        "name": _ea_attr(elem, "name"),
        "ID": _ea_attr(elem, f"{NS_XMI}idref") or _ea_attr(elem, "idref") or _ea_attr(elem, "ea_localid"),
        "uri": _tag_value(tags, "uri") or None,
        "type": _ea_attr(elem, f"{NS_XMI}type") or _ea_attr(elem, "type"),
        "package": _ea_attr(_ea_find(elem, "model"), "package"),
        "tags": tags,
    }


def _ea_get_class(elem: Element) -> dict[str, Any]:
    tags = _ea_extract_semantic_tags(elem, is_class_level=True)
    class_dict = {
        "name": _ea_attr(elem, "name"),
        "ID": _ea_attr(elem, f"{NS_XMI}idref") or _ea_attr(elem, "idref") or _ea_attr(elem, "ea_localid"),
        "uri": _tag_value(tags, "uri") or None,
        "type": _ea_attr(elem, f"{NS_XMI}type") or _ea_attr(elem, "type"),
        "package": _ea_attr(_ea_find(elem, "model"), "package"),
        "tags": tags,
        "attributes": [],
    }
    for attribute in _ea_findall(_ea_find(elem, "attributes"), "attribute"):
        attr_type = _ea_attr(attribute, "type")
        if not attr_type:
            prop = _ea_find(attribute, "properties")
            if prop is not None:
                attr_type = _ea_attr(prop, "type")

        # Nettoyage des types EA
        attr_type = _clean_type_name(attr_type or "")
        attr_tags = _ea_extract_semantic_tags(attribute, is_class_level=False)

        attr_dict = {
            "name": _ea_attr(attribute, "name"),
            "type": attr_type,
            "uri": _tag_value(attr_tags, "uri") or None,
            "lower_bounds": next((_ea_attr(b, "lower") for b in _ea_findall(attribute, "bounds")), None),
            "upper_bounds": next((_ea_attr(b, "upper") for b in _ea_findall(attribute, "bounds")), None),
            "tags_attribute": attr_tags,
        }
        class_dict["attributes"].append(attr_dict)
    return class_dict


def _ea_get_datatype(elem: Element) -> dict[str, Any]:
    dt_dict = _ea_get_class(elem)
    dt_dict["type"] = "uml:DataType"
    return dt_dict


def _ea_get_enumeration(elem: Element) -> dict[str, Any]:
    tags = _ea_extract_semantic_tags(elem, is_class_level=True)
    return {
        "name": _ea_attr(elem, "name"),
        "ID": _ea_attr(elem, f"{NS_XMI}idref") or _ea_attr(elem, "idref") or _ea_attr(elem, "ea_localid"),
        "uri": _tag_value(tags, "uri") or None,
        "type": _ea_attr(elem, f"{NS_XMI}type") or _ea_attr(elem, "type"),
        "package": _ea_attr(_ea_find(elem, "model"), "package"),
        "tags": tags,
        "categories": [
            _ea_attr(a, "name")
            for a in _ea_findall(_ea_find(elem, "attributes"), "attribute")
            if _ea_attr(a, "name")
        ],
    }


def _ea_get_connector(connector: Element) -> dict[str, Any]:
    labels = _ea_find(connector, "labels")
    source = _ea_find(connector, "source")
    target = _ea_find(connector, "target")

    source_id = ""
    target_id = ""
    source_name = ""
    target_name = ""

    if source is not None:
        source_id = _ea_attr(source, f"{NS_XMI}idref") or _ea_attr(source, "idref")
        source_model = _ea_find(source, "model")
        if not source_id and source_model is not None:
            source_id = _ea_attr(source_model, "ea_localid") or ""
        source_name = _ea_attr(source_model, "name") if source_model is not None else ""

    if target is not None:
        target_id = _ea_attr(target, f"{NS_XMI}idref") or _ea_attr(target, "idref")
        target_model = _ea_find(target, "model")
        if not target_id and target_model is not None:
            target_id = _ea_attr(target_model, "ea_localid") or ""
        target_name = _ea_attr(target_model, "name") if target_model is not None else ""

    doc_tags = _ea_extract_semantic_tags(connector, is_class_level=False)
    source_tags = _ea_extract_semantic_tags(source, is_class_level=False) if source is not None else []
    target_tags = _ea_extract_semantic_tags(target, is_class_level=False) if target is not None else []

    existing_tgt_names = {t["name"].lower() for t in target_tags}
    for dt in doc_tags:
        if dt["name"].lower() not in existing_tgt_names:
            target_tags.append(dt)

    connector_id = _ea_attr(connector, f"{NS_XMI}idref") or _ea_attr(connector, "idref")
    if not connector_id:
        connector_id = _tag_value(target_tags, "connector_id")

    relation_uri = _tag_value(doc_tags, "uri") or _tag_value(target_tags, "uri")

    # Securité supplémentaire si la regex a manqué l'URI dans doc_tags à cause d'un formatage bizarre
    if not relation_uri:
        raw_doc = _ea_attr(_ea_find(connector, "documentation"), "value") or ""
        match = re.search(r"(?i)URI:\s*(http[^\s\n]+)", raw_doc)
        if match:
            relation_uri = match.group(1).strip()
            target_tags.append({"name": "uri", "value": relation_uri})
            doc_tags.append({"name": "uri", "value": relation_uri})

    source_type_node = _ea_find(source, "type") if source is not None else None
    target_type_node = _ea_find(target, "type") if target is not None else None
    source_role_node = _ea_find(source, "role") if source is not None else None
    target_role_node = _ea_find(target, "role") if target is not None else None

    # Extraction logique prioritaire
    logical_lb = _ea_attr(source_type_node, "multiplicity")
    logical_rb = _ea_attr(target_type_node, "multiplicity")
    logical_lt = _ea_attr(source_role_node, "name")
    logical_rt = _ea_attr(target_role_node, "name")

    relation_name = _tag_value(doc_tags, "label-en") or _ea_attr(connector, "name") or ""

    return {
        "connector_id": connector_id,
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_name,
        "target_name": target_name,
        "relationship": _ea_attr(_ea_find(connector, "properties"), "ea_type"),
        "name": relation_name,
        "uri": relation_uri,
        # Si la donnée logique est vide, on se rabat sur la donnée visuelle
        "lb": logical_lb if logical_lb else (_ea_attr(labels, "lb") or ""),
        "lt": logical_lt if logical_lt else (_ea_attr(labels, "lt") or ""),
        "rb": logical_rb if logical_rb else (_ea_attr(labels, "rb") or ""),
        "rt": logical_rt if logical_rt else (_ea_attr(labels, "rt") or ""),
        "tags": doc_tags,
        "tags_source": source_tags,
        "tags_target": target_tags,
    }


def _ea_get_connectors(root: Element) -> list[dict[str, Any]]:
    return [
        _ea_get_connector(conn)
        for conn in root.iter()
        if _local_name(conn.tag) == "connector"
    ]


def _ea_get_elements_direct(root: Element) -> list[dict[str, Any]]:
    elems = []
    try:
        if len(root) > 2 and len(root[2]) > 0:
            elems = list(root[2][0].iter())
    except Exception:
        pass
    if not elems:
        elems = list(root.iter())

    element_nodes = [e for e in elems if _local_name(e.tag) == "element"]
    elements: list[dict[str, Any]] = []

    for elem in element_nodes:
        t = _ea_attr(elem, f"{NS_XMI}type") or _ea_attr(elem, "type") or ""
        if t == "uml:Package":
            elements.append(_ea_get_package(elem))
        elif t == "uml:Class":
            elements.append(_ea_get_class(elem))
        elif t == "uml:DataType":
            elements.append(_ea_get_datatype(elem))
        elif t == "uml:Enumeration":
            elements.append(_ea_get_enumeration(elem))
    return elements


# =====================================================================
#  MAIN ENTRYPOINT — AUTO-ROUTING MULTI-FORMAT PARSER
# =====================================================================

def xml_to_json(bytes_data) -> dict[str, Any]:
    """
    Convert any XML/XMI/UML2 file into a JSON-compatible model.
    Robustly handles Standard XMI (Papyrus/Cameo) AND EA Proprietary exports.
    """
    global NS_XMI, NS_UML

    source = bytes_data
    if isinstance(bytes_data, (bytes, bytearray)):
        source = io.BytesIO(bytes_data)

    try:
        namespaces = detect_namespaces(source)
        NS_XMI = namespaces["xmi"]
        NS_UML = namespaces["uml"]

        if hasattr(source, "seek"):
            source.seek(0)

        tree = parse(source)
        root = tree.getroot()

        _build_parent_map(root)

        if _is_ea_xml(root):
            print("[INFO] EA-proprietary direct XML structure detected. Routing to Enterprise Architect parser...")
            elements = _ea_get_elements_direct(root)
            connectors = _ea_get_connectors(root)
        else:
            print("[INFO] Standard UML2 layout detected. Routing to STANDARD CLASSIQUE parser...")
            elements = _get_standard_elements(root)
            connectors = _get_standard_connectors(root, elements)

        if not elements:
            raise ValueError(
                "No UML elements were detected. The XML/XMI variant may require a custom parser."
            )

        return {
            "elements": elements,
            "connectors": _dedupe_connectors(connectors),
        }

    except Exception as e:
        print(f"[ERROR] XML/XMI parsing failed: {e}")
        raise
