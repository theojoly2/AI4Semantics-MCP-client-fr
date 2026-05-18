from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional
from xml.etree.ElementTree import Element, parse, iterparse
import io


DEFAULT_XMI_NS = "{http://schema.omg.org/spec/XMI/2.1}"
DEFAULT_UML_NS = "{http://schema.omg.org/spec/UML/2.1}"

NS_XMI = DEFAULT_XMI_NS
NS_UML = DEFAULT_UML_NS

_PARENT_MAP: dict[Element, Element] = {}


def _local_name(value: str) -> str:
    if not value:
        return ""
    if value.startswith("{"):
        return value.split("}", 1)[1]
    if ":" in value:
        return value.split(":", 1)[1]
    return value


def detect_namespaces(source: Any, root: Optional[Element] = None) -> dict[str, str]:
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


def _extract_tags(tags_elem: Optional[Element]) -> list[dict[str, str]]:
    if tags_elem is None:
        return []

    out: list[dict[str, str]] = []
    for tag in _children(tags_elem, "tag"):
        try:
            out.append({
                "name": _attr(tag, "name"),
                "value": _attr(tag, "value"),
            })
        except Exception:
            continue
    return out


def _tag_value(tags: list[dict[str, Any]] | None, name: str) -> str:
    for tag in tags or []:
        if (tag.get("name") or "").strip() == name:
            return (tag.get("value") or "").strip()
    return ""


def _connector_name(
    connector_tags: list[dict[str, Any]] | None,
    source_tags: list[dict[str, Any]] | None,
    target_tags: list[dict[str, Any]] | None,
    fallback: str = "",
) -> str:
    return (
        _tag_value(connector_tags, "label-en")
        or _tag_value(target_tags, "label-en")
        or _tag_value(source_tags, "label-en")
        or fallback
    )


def _connector_uri(
    connector_tags: list[dict[str, Any]] | None,
    source_tags: list[dict[str, Any]] | None,
    target_tags: list[dict[str, Any]] | None,
) -> str:
    return (
        _tag_value(connector_tags, "uri")
        or _tag_value(target_tags, "uri")
        or _tag_value(source_tags, "uri")
    )


def _is_package_element(elem: Optional[Element]) -> bool:
    if elem is None:
        return False

    elem_type = _attr(elem, f"{NS_XMI}type") or _attr(elem, "type")
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
            return _attr(current, f"{NS_XMI}id") or _attr(current, "id") or ""
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
    if elem is None:
        return ""

    direct_type = _attr(elem, "type")
    if direct_type:
        return _normalize_href_or_ref(direct_type)

    type_elem = _child(elem, "type")
    if type_elem is None:
        return ""

    href = _attr(type_elem, "href")
    if href:
        return _normalize_href_or_ref(href)

    return (
        _attr(type_elem, f"{NS_XMI}idref")
        or _attr(type_elem, "idref")
        or _attr(type_elem, f"{NS_XMI}id")
        or _attr(type_elem, "id")
        or ""
    )


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


def _extract_ea_attribute(attribute: Element) -> Optional[dict[str, Any]]:
    attr_name = _attr(attribute, "name")
    if not attr_name:
        return None

    attr_type = ""
    for prop in _children(attribute, "properties"):
        t = _attr(prop, "type")
        if t:
            attr_type = t
            break

    lower_bound = ""
    upper_bound = ""
    for bounds in _children(attribute, "bounds"):
        lower_bound = _attr(bounds, "lower")
        upper_bound = _attr(bounds, "upper")
        break

    return {
        "name": attr_name,
        "type": attr_type,
        "lower_bounds": lower_bound,
        "upper_bounds": upper_bound,
        "tags_attribute": _extract_tags(_child(attribute, "tags")),
    }


def _extract_standard_attribute(attribute: Element) -> Optional[dict[str, Any]]:
    attr_name = _attr(attribute, "name")
    if not attr_name:
        return None

    attr_type = _extract_type_reference(attribute)
    if "PrimitiveTypes.xmi#" in attr_type:
        attr_type = attr_type.split("#")[-1]

    return {
        "name": attr_name,
        "visibility": _attr(attribute, "visibility") or "public",
        "type": attr_type,
        "lower_bounds": _attr(_child(attribute, "lowerValue"), "value"),
        "upper_bounds": _attr(_child(attribute, "upperValue"), "value"),
        "tags_attribute": [],
    }


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
            conn.get("lb", ""),
            conn.get("lt", ""),
            conn.get("rb", ""),
            conn.get("rt", ""),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(conn)

    return out


def _extract_uml_package(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name") or "UnnamedPackage",
        "ID": _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Package",
        "visibility": _attr(elem, "visibility") or "public",
        "package": _find_parent_package_id(elem),
        "tags": [],
    }


def _extract_uml_class(elem: Element) -> dict[str, Any]:
    class_dict = {
        "name": _attr(elem, "name") or "UnnamedClass",
        "ID": _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Class",
        "visibility": _attr(elem, "visibility") or "public",
        "package": _find_parent_package_id(elem),
        "attributes": [],
        "tags": [],
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
    enum_dict = {
        "name": _attr(elem, "name") or "UnnamedEnumeration",
        "ID": _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or "",
        "type": "uml:Enumeration",
        "visibility": _attr(elem, "visibility") or "public",
        "package": _find_parent_package_id(elem),
        "categories": [],
        "tags": [],
    }

    for literal in _children(elem, "ownedLiteral"):
        lit_name = _attr(literal, "name")
        if lit_name:
            enum_dict["categories"].append(lit_name)

    return enum_dict


def _find_ea_extension(root: Element) -> Optional[Element]:
    for elem in root.iter():
        if _local_name(elem.tag) != "Extension":
            continue
        extender = _attr(elem, "extender")
        if not extender or "Enterprise Architect" in extender:
            return elem
    return None


def _get_package(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name") or "UnnamedPackage",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Package",
        "package": _attr(_child(elem, "model"), "package") or "",
        "tags": _extract_tags(_child(elem, "tags")),
    }


def _get_class(elem: Element) -> dict[str, Any]:
    class_dict = {
        "name": _attr(elem, "name") or "UnnamedClass",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Class",
        "package": _attr(_child(elem, "model"), "package") or "",
        "tags": _extract_tags(_child(elem, "tags")),
        "attributes": [],
    }

    for attribute in _children(_child(elem, "attributes"), "attribute"):
        attr_dict = _extract_ea_attribute(attribute)
        if attr_dict is not None:
            class_dict["attributes"].append(attr_dict)

    return class_dict


def _get_datatype(elem: Element) -> dict[str, Any]:
    dt_dict = {
        "name": _attr(elem, "name") or "UnnamedDataType",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:DataType",
        "package": _attr(_child(elem, "model"), "package") or "",
        "tags": _extract_tags(_child(elem, "tags")),
        "attributes": [],
    }

    for attribute in _children(_child(elem, "attributes"), "attribute"):
        attr_dict = _extract_ea_attribute(attribute)
        if attr_dict is not None:
            dt_dict["attributes"].append(attr_dict)

    return dt_dict


def _get_enumeration(elem: Element) -> dict[str, Any]:
    return {
        "name": _attr(elem, "name") or "UnnamedEnumeration",
        "ID": _attr(elem, f"{NS_XMI}idref") or "",
        "type": _attr(elem, f"{NS_XMI}type") or "uml:Enumeration",
        "package": _attr(_child(elem, "model"), "package") or "",
        "tags": _extract_tags(_child(elem, "tags")),
        "categories": [
            _attr(a, "name")
            for a in _children(_child(elem, "attributes"), "attribute")
            if _attr(a, "name")
        ],
    }


def _get_ea_elements(root: Element) -> list[dict[str, Any]]:
    ext = _find_ea_extension(root)
    if ext is None:
        return []

    elements_container = _child(ext, "elements")
    if elements_container is None:
        return []

    elements: list[dict[str, Any]] = []

    for elem in _children(elements_container, "element"):
        elem_type = _attr(elem, f"{NS_XMI}type")

        if elem_type == "uml:Package":
            elements.append(_get_package(elem))
        elif elem_type == "uml:Class":
            elements.append(_get_class(elem))
        elif elem_type == "uml:DataType":
            elements.append(_get_datatype(elem))
        elif elem_type == "uml:Enumeration":
            elements.append(_get_enumeration(elem))

    return elements


def _build_property_index(root: Element) -> dict[str, Element]:
    prop_index: dict[str, Element] = {}

    for elem in root.iter():
        local = _local_name(elem.tag)
        if local not in {"ownedEnd", "ownedAttribute"}:
            continue

        elem_id = _attr(elem, f"{NS_XMI}id") or _attr(elem, "id")
        if elem_id:
            prop_index[elem_id] = elem

    return prop_index


def _association_relationship(assoc: Element, end_1: Element, end_2: Element) -> str:
    aggregations = {
        _attr(assoc, "aggregation"),
        _attr(end_1, "aggregation"),
        _attr(end_2, "aggregation"),
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

    member_end_ids = (_attr(assoc, "memberEnd") or "").split()
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

    src_type_ref = _extract_type_reference(end_1)
    tgt_type_ref = _extract_type_reference(end_2)

    source_id, source_name = _resolve_endpoint(
        ref_id=src_type_ref,
        ref_name="",
        element_by_id=element_by_id,
        elements_by_name=elements_by_name,
    )
    target_id, target_name = _resolve_endpoint(
        ref_id=tgt_type_ref,
        ref_name="",
        element_by_id=element_by_id,
        elements_by_name=elements_by_name,
    )

    return {
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_name,
        "target_name": target_name,
        "relationship": _association_relationship(assoc, end_1, end_2),
        "name": _attr(assoc, "name") or "",
        "uri": "",
        "lb": _extract_multiplicity(end_1),
        "lt": _attr(end_1, "name") or "",
        "rb": _extract_multiplicity(end_2),
        "rt": _attr(end_2, "name") or "",
        "tags": [],
        "tags_source": [],
        "tags_target": [],
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
        elem_type = _attr(elem, f"{NS_XMI}type")
        if not elem_type or not any(t in elem_type for t in ("Class", "DataType", "Enumeration")):
            continue

        source_id = _attr(elem, f"{NS_XMI}id") or _attr(elem, "id") or ""
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
                "uri": "",
                "lb": "",
                "lt": "",
                "rb": "",
                "rt": "",
                "tags": [],
                "tags_source": [],
                "tags_target": [],
            })

    return connectors


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
        elem_type = _attr(elem, f"{NS_XMI}type")
        if elem_type and "Association" in elem_type:
            conn = _get_standard_association_connector(
                elem,
                element_by_id,
                elements_by_name,
                property_index,
            )
            if conn is not None:
                connectors.append(conn)

    connectors.extend(
        _get_standard_generalization_connectors(model, element_by_id, elements_by_name)
    )

    return _dedupe_connectors(connectors)


def _get_connector(
    connector: Element,
    element_by_id: dict[str, dict[str, Any]],
    elements_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    labels = _child(connector, "labels")

    source = _child(connector, "source")
    target = _child(connector, "target")

    source_model = _child(source, "model")
    target_model = _child(target, "model")

    raw_source_name = _attr(source_model, "name") or ""
    raw_target_name = _attr(target_model, "name") or ""

    raw_source_id = (
        _attr(source_model, f"{NS_XMI}idref")
        or _attr(source_model, "idref")
        or _attr(source_model, f"{NS_XMI}id")
        or _attr(source_model, "id")
        or ""
    )
    raw_target_id = (
        _attr(target_model, f"{NS_XMI}idref")
        or _attr(target_model, "idref")
        or _attr(target_model, f"{NS_XMI}id")
        or _attr(target_model, "id")
        or ""
    )

    source_id, source_name = _resolve_endpoint(
        ref_id=raw_source_id,
        ref_name=raw_source_name,
        element_by_id=element_by_id,
        elements_by_name=elements_by_name,
    )
    target_id, target_name = _resolve_endpoint(
        ref_id=raw_target_id,
        ref_name=raw_target_name,
        element_by_id=element_by_id,
        elements_by_name=elements_by_name,
    )

    connector_tags = _extract_tags(_child(connector, "tags"))
    source_tags = _extract_tags(_child(source, "tags"))
    target_tags = _extract_tags(_child(target, "tags"))

    relation_name = _connector_name(
        connector_tags=connector_tags,
        source_tags=source_tags,
        target_tags=target_tags,
        fallback="",
    )
    relation_uri = _connector_uri(
        connector_tags=connector_tags,
        source_tags=source_tags,
        target_tags=target_tags,
    )

    return {
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_name,
        "target_name": target_name,
        "relationship": _attr(_child(connector, "properties"), "ea_type") or "",
        "name": relation_name,
        "uri": relation_uri,
        "lb": _attr(labels, "lb") or "",
        "lt": _attr(labels, "lt") or "",
        "rb": _attr(labels, "rb") or "",
        "rt": _attr(labels, "rt") or "",
        "tags": connector_tags,
        "tags_source": source_tags,
        "tags_target": target_tags,
    }


def _get_ea_connectors(
    root: Element,
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ext = _find_ea_extension(root)
    if ext is None:
        return []

    connectors_container = _child(ext, "connectors")
    if connectors_container is None:
        return []

    element_by_id, elements_by_name = _build_element_indexes(elements)

    connectors = [
        _get_connector(conn, element_by_id, elements_by_name)
        for conn in _children(connectors_container, "connector")
    ]

    return _dedupe_connectors(connectors)


def _get_standard_elements(root: Element) -> list[dict[str, Any]]:
    model = _find_uml_model(root)
    if model is None:
        raise ValueError("No UML Model found in XMI file")

    elements: list[dict[str, Any]] = []

    for elem in _get_packaged_elements(model):
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


def _get_elements(root: Element) -> list[dict[str, Any]]:
    ea_elements = _get_ea_elements(root)
    if ea_elements:
        return ea_elements

    return _get_standard_elements(root)


def xml_to_json(bytes_data) -> dict[str, Any]:
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

        elements = _get_elements(root)
        if not elements:
            raise ValueError(
                "No UML elements detected. File may be empty or use an unsupported XMI variant."
            )

        connectors = _get_ea_connectors(root, elements)
        if not connectors:
            connectors = _get_standard_connectors(root, elements)

        return {
            "elements": elements,
            "connectors": connectors,
        }

    except Exception as e:
        print(f"[ERROR] XMI parsing failed: {e}")
        raise
