from __future__ import annotations

from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring, register_namespace


XMI_NS = "http://schema.omg.org/spec/XMI/2.1"
UML_NS = "http://schema.omg.org/spec/UML/2.1"
PRIMITIVE_TYPES_HREF = "http://www.omg.org/spec/UML/20131001/PrimitiveTypes.xmi"

VALID_VISIBILITIES = {"public", "private", "protected", "package"}


def _xmi_attr(name: str) -> str:
    return f"{{{XMI_NS}}}{name}"


def _is_non_empty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _safe_name(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _canonical_name(value: Any) -> str:
    return str(value or "").strip().casefold()


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_primitive_type(value: str | None) -> str | None:
    if not value:
        return None

    raw = value.strip()
    lowered = raw.lower()

    mapping = {
        "string": "String",
        "str": "String",
        "text": "String",
        "char": "String",
        "varchar": "String",
        "integer": "Integer",
        "int": "Integer",
        "long": "Integer",
        "short": "Integer",
        "boolean": "Boolean",
        "bool": "Boolean",
        "real": "Real",
        "float": "Real",
        "double": "Real",
        "decimal": "Real",
        "number": "Real",
        "unlimitednatural": "UnlimitedNatural",
        "date": "String",
        "datetime": "String",
        "time": "String",
        "uri": "String",
    }

    normalized = mapping.get(lowered, raw)
    if normalized in {"String", "Integer", "Boolean", "Real", "UnlimitedNatural"}:
        return normalized
    return None


def _normalize_multiplicity_token(value: str | None) -> str | None:
    if value is None:
        return None

    token = str(value).strip()
    if not token:
        return None

    lowered = token.casefold()
    if lowered in {"n", "m", "many", "multiple", "*"}:
        return "*"

    return token


def _parse_multiplicity(raw: Any) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None

    text = str(raw).strip()
    if not text:
        return None, None

    if text == "*":
        return "0", "*"

    if ".." in text:
        lower_raw, upper_raw = text.split("..", 1)
        lower = _normalize_multiplicity_token(lower_raw.strip())
        upper = _normalize_multiplicity_token(upper_raw.strip())
    else:
        token = _normalize_multiplicity_token(text)
        if token == "*":
            return "0", "*"
        return token, token

    if lower == "*":
        lower = "0"
    if upper == "*" and lower is None:
        lower = "0"

    return lower, upper


def _tag_value(tags: list[dict[str, Any]] | None, *keys: str) -> str | None:
    wanted = {k.casefold() for k in keys}
    for tag in _safe_list(tags):
        if not isinstance(tag, dict):
            continue
        name = str(tag.get("name", "")).strip().casefold()
        value = str(tag.get("value", "")).strip()
        if name in wanted and value:
            return value
    return None


def _merge_connector_tags(connector_dict: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for bucket_name in ("tags", "tags_target", "tags_source"):
        for tag in _safe_list(connector_dict.get(bucket_name)):
            if not isinstance(tag, dict):
                continue

            name = str(tag.get("name", "")).strip()
            value = str(tag.get("value", "")).strip()
            if not name or not value:
                continue

            key = (name.casefold(), value)
            if key in seen:
                continue

            seen.add(key)
            merged.append({"name": name, "value": value})

    return merged


def _tag_lines(tags: list[dict[str, Any]] | None) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for tag in _safe_list(tags):
        if not isinstance(tag, dict):
            continue

        name = str(tag.get("name", "")).strip()
        value = str(tag.get("value", "")).strip()
        if not name or not value:
            continue

        lowered = name.casefold()
        if lowered in {"uri", "semanticuri", "semantic_uri"}:
            label = "URI"
        elif lowered in {"definition-en", "definition", "description"}:
            label = "Definition"
        elif lowered in {"label-en", "label"}:
            label = "Label"
        elif lowered in {"usagenote-en", "usagenote"}:
            label = "Usage note"
        elif lowered == "referenced":
            label = "Referenced"
        elif lowered in {"connectorid", "connector_id"}:
            label = "Connector ID"
        else:
            label = name

        item = (label, value)
        if item not in seen:
            seen.add(item)
            lines.append(f"{label}: {value}")

    return lines


def _add_comment(parent: Element, text: str | None,
                 comment_id: str,
                 annotated_id: str | None = None) -> None:
    if not _is_non_empty(text):
        return

    owned_comment = SubElement(parent, "ownedComment")
    owned_comment.set(_xmi_attr("id"), comment_id)
    owned_comment.set(_xmi_attr("type"), "uml:Comment")
    if annotated_id:
        owned_comment.set("annotatedElement", annotated_id)

    body = SubElement(owned_comment, "body")
    body.text = str(text).strip()


def _build_comment_text_from_tags(tags: list[dict[str, Any]] | None) -> str | None:
    lines = _tag_lines(tags)
    return "\n".join(lines) if lines else None


def _connector_semantic_uri(connector_dict: dict[str, Any]) -> str | None:
    for key in ("semantic_uri", "semanticuri", "uri"):
        value = connector_dict.get(key)
        if _is_non_empty(value):
            return str(value).strip()

    merged = _merge_connector_tags(connector_dict)
    return _tag_value(merged, "semantic_uri", "semanticuri", "uri")


def _connector_id(connector_dict: dict[str, Any], index: int) -> str:
    for key in ("ID", "connector_id", "connectorid"):
        value = connector_dict.get(key)
        if _is_non_empty(value):
            return str(value).strip()

    merged = _merge_connector_tags(connector_dict)
    from_tags = _tag_value(merged, "connectorid", "connector_id")
    if from_tags:
        return from_tags

    return f"assoc_{index}"


def _build_connector_comment(connector_dict: dict[str, Any]) -> str | None:
    merged = _merge_connector_tags(connector_dict)
    text = _build_comment_text_from_tags(merged)
    if text:
        return text

    lines: list[str] = []

    semantic_uri = _connector_semantic_uri(connector_dict)
    if semantic_uri:
        lines.append(f"URI: {semantic_uri}")

    label = connector_dict.get("name") or connector_dict.get("label")
    if _is_non_empty(label):
        lines.append(f"Label: {str(label).strip()}")

    return "\n".join(lines) if lines else None


def _set_visibility(target: Element, raw_visibility: Any, default: str | None = None) -> None:
    visibility = str(raw_visibility or "").strip()
    if not visibility and default:
        visibility = default
    if visibility in VALID_VISIBILITIES:
        target.set("visibility", visibility)


def _set_bool_attr_if_present(target: Element, attr_name: str, raw_value: Any) -> None:
    value = _as_bool(raw_value)
    if value is not None:
        target.set(attr_name, "true" if value else "false")


def _set_common_classifier_attrs(element: Element, element_dict: dict[str, Any]) -> None:
    _set_visibility(element, element_dict.get("visibility"), default="public")
    _set_bool_attr_if_present(element, "isAbstract", element_dict.get("isAbstract"))
    _set_bool_attr_if_present(element, "isLeaf", element_dict.get("isLeaf"))
    _set_bool_attr_if_present(element, "isFinalSpecialization", element_dict.get("isFinalSpecialization"))


def _set_property_attrs(prop: Element, raw: dict[str, Any] | None = None, *, default_visibility: str | None = None) -> None:
    raw = raw or {}
    _set_visibility(prop, raw.get("visibility"), default=default_visibility)

    is_ordered = _as_bool(raw.get("isOrdered"))
    is_unique = _as_bool(raw.get("isUnique"))

    prop.set("isOrdered", "true" if is_ordered is True else "false")
    prop.set("isUnique", "false" if is_unique is False else "true")

    _set_bool_attr_if_present(prop, "isReadOnly", raw.get("isReadOnly"))
    _set_bool_attr_if_present(prop, "isDerived", raw.get("isDerived"))
    _set_bool_attr_if_present(prop, "isStatic", raw.get("isStatic"))


def _resolve_classifier_type_id(
    raw_type: Any,
    classifier_ids_by_id: dict[str, str],
    classifier_ids_by_name: dict[str, str],
) -> tuple[str | None, str | None]:
    if not _is_non_empty(raw_type):
        return None, None

    text = str(raw_type).strip()

    primitive = _normalize_primitive_type(text)
    if primitive:
        return None, primitive

    if text in classifier_ids_by_id:
        return classifier_ids_by_id[text], None

    by_name = classifier_ids_by_name.get(_canonical_name(text))
    if by_name:
        return by_name, None

    return None, None


def _add_value_spec(parent: Element, tag_name: str, value_id: str, raw_value: Any, primitive_hint: str | None = None) -> None:
    if not _is_non_empty(raw_value):
        return

    value = str(raw_value).strip()
    primitive = _normalize_primitive_type(primitive_hint)

    xmi_type = "uml:LiteralString"
    normalized_value = value

    if primitive == "Boolean" and value.casefold() in {"true", "false"}:
        xmi_type = "uml:LiteralBoolean"
        normalized_value = value.casefold()
    elif primitive in {"Integer", "UnlimitedNatural"} and value.isdigit():
        xmi_type = "uml:LiteralInteger" if primitive == "Integer" else "uml:LiteralUnlimitedNatural"
    elif value.isdigit():
        xmi_type = "uml:LiteralInteger"

    node = SubElement(parent, tag_name)
    node.set(_xmi_attr("id"), value_id)
    node.set(_xmi_attr("type"), xmi_type)
    node.set("value", normalized_value)


def _add_multiplicity(prop: Element, raw: Any, value_id_prefix: str) -> None:
    lower_raw, upper_raw = _parse_multiplicity(raw)

    if lower_raw is not None:
        lower = SubElement(prop, "lowerValue")
        lower.set(_xmi_attr("type"), "uml:LiteralInteger")
        lower.set(_xmi_attr("id"), f"{value_id_prefix}_lower")
        lower.set("value", lower_raw)

    if upper_raw is not None:
        upper = SubElement(prop, "upperValue")
        upper.set(
            _xmi_attr("type"),
            "uml:LiteralUnlimitedNatural" if upper_raw == "*" else "uml:LiteralInteger",
        )
        upper.set(_xmi_attr("id"), f"{value_id_prefix}_upper")
        upper.set("value", upper_raw)


def _add_attribute(
    parent: Element,
    attribute_dict: dict[str, Any],
    owner_id: str,
    index: int,
    classifier_ids_by_id: dict[str, str],
    classifier_ids_by_name: dict[str, str],
) -> None:
    attr = SubElement(parent, "ownedAttribute")
    attr_id = f"{owner_id}_attr_{index}"
    attr.set(_xmi_attr("id"), attr_id)
    attr.set(_xmi_attr("type"), "uml:Property")
    attr.set("name", _safe_name(attribute_dict.get("name"), f"attribute_{index}"))

    _set_property_attrs(attr, attribute_dict, default_visibility="private")

    type_id, primitive_type = _resolve_classifier_type_id(
        attribute_dict.get("type"),
        classifier_ids_by_id=classifier_ids_by_id,
        classifier_ids_by_name=classifier_ids_by_name,
    )

    if primitive_type:
        type_elem = SubElement(attr, "type")
        type_elem.set(_xmi_attr("type"), "uml:PrimitiveType")
        type_elem.set("href", f"{PRIMITIVE_TYPES_HREF}#{primitive_type}")
    elif type_id:
        attr.set("type", type_id)

    lower = attribute_dict.get("lower_bounds")
    upper = attribute_dict.get("upper_bounds")
    if _is_non_empty(lower) and _is_non_empty(upper):
        multiplicity = f"{str(lower).strip()}..{str(upper).strip()}"
        _add_multiplicity(attr, multiplicity, attr_id)
    elif _is_non_empty(lower):
        _add_multiplicity(attr, f"{str(lower).strip()}..{str(lower).strip()}", attr_id)
    elif _is_non_empty(upper):
        _add_multiplicity(attr, f"{str(upper).strip()}..{str(upper).strip()}", attr_id)

    _add_value_spec(
        attr,
        "defaultValue",
        f"{attr_id}_default",
        attribute_dict.get("default"),
        primitive_hint=primitive_type,
    )

    attr_comment = _build_comment_text_from_tags(_safe_list(attribute_dict.get("tags_attribute")))
    _add_comment(
        attr,
        attr_comment,
        f"{attr_id}_comment",
        annotated_id=attr_id,
        )


def _add_class_like(
    parent: Element,
    element_dict: dict[str, Any],
    uml_type: str,
    classifier_ids_by_id: dict[str, str],
    classifier_ids_by_name: dict[str, str],
) -> Element:
    element_id = _safe_name(element_dict.get("ID"), f"{uml_type}_id")
    element = SubElement(parent, "packagedElement")
    element.set(_xmi_attr("id"), element_id)
    element.set(_xmi_attr("type"), uml_type)
    element.set("name", _safe_name(element_dict.get("name"), element_id))

    _set_common_classifier_attrs(element, element_dict)
    _add_comment(
        element,
        _build_comment_text_from_tags(_safe_list(element_dict.get("tags"))),
        f"{element_id}_comment",
        annotated_id=element_id,
    )

    for index, attribute_dict in enumerate(_safe_list(element_dict.get("attributes")), start=1):
        if isinstance(attribute_dict, dict):
            _add_attribute(
                element,
                attribute_dict,
                element_id,
                index,
                classifier_ids_by_id=classifier_ids_by_id,
                classifier_ids_by_name=classifier_ids_by_name,
            )

    return element


def _add_enumeration(parent: Element, element_dict: dict[str, Any]) -> Element:
    element_id = _safe_name(element_dict.get("ID"), "enumeration_id")

    enum_elem = SubElement(parent, "packagedElement")
    enum_elem.set(_xmi_attr("id"), element_id)
    enum_elem.set(_xmi_attr("type"), "uml:Enumeration")
    enum_elem.set("name", _safe_name(element_dict.get("name"), element_id))

    _set_common_classifier_attrs(enum_elem, element_dict)
    _add_comment(
        enum_elem,
        _build_comment_text_from_tags(_safe_list(element_dict.get("tags"))),
        f"{element_id}_comment",
        annotated_id=element_id,
    )

    categories = element_dict.get("categories")
    if not isinstance(categories, list):
        categories = []

    if not categories:
        categories = [
            str(attr.get("name")).strip()
            for attr in _safe_list(element_dict.get("attributes"))
            if isinstance(attr, dict) and _is_non_empty(attr.get("name"))
        ]

    for index, literal_name in enumerate(categories, start=1):
        literal = SubElement(enum_elem, "ownedLiteral")
        literal.set(_xmi_attr("id"), f"{element_id}_lit_{index}")
        literal.set(_xmi_attr("type"), "uml:EnumerationLiteral")
        literal.set("name", _safe_name(literal_name, f"literal_{index}"))

    return enum_elem


def _add_package(parent: Element, element_dict: dict[str, Any]) -> Element:
    element_id = _safe_name(element_dict.get("ID"), "package_id")

    package = SubElement(parent, "packagedElement")
    package.set(_xmi_attr("id"), element_id)
    package.set(_xmi_attr("type"), "uml:Package")
    package.set("name", _safe_name(element_dict.get("name"), element_id))

    _set_visibility(package, element_dict.get("visibility"), default="public")
    _add_comment(
        package,
        _build_comment_text_from_tags(_safe_list(element_dict.get("tags"))),
        f"{element_id}_comment",
        annotated_id=element_id,
    )
    return package


def _relationship_kind(connector_dict: dict[str, Any]) -> str:
    return str(connector_dict.get("relationship", "")).strip().casefold() or "association"


def _association_name(connector_dict: dict[str, Any], index: int) -> str:
    candidates = [
        connector_dict.get("name"),
        connector_dict.get("label"),
        connector_dict.get("rt"),
        connector_dict.get("lt"),
    ]
    for candidate in candidates:
        if _is_non_empty(candidate):
            return str(candidate).strip()
    return f"Association_{index}"


def _connector_source_name(connector_dict: dict[str, Any]) -> str:
    return _safe_name(
        connector_dict.get("source_name")
        or connector_dict.get("source")
        or connector_dict.get("sourceName"),
        "",
    )


def _connector_target_name(connector_dict: dict[str, Any]) -> str:
    return _safe_name(
        connector_dict.get("target_name")
        or connector_dict.get("target")
        or connector_dict.get("targetName"),
        "",
    )


def _connector_source_id(
    connector_dict: dict[str, Any],
    classifier_ids_by_name: dict[str, str],
) -> str | None:
    for key in ("source_id", "sourceid", "sourceId"):
        value = connector_dict.get(key)
        if _is_non_empty(value):
            return str(value).strip()

    source_name = _connector_source_name(connector_dict)
    if source_name:
        return classifier_ids_by_name.get(_canonical_name(source_name))

    return None


def _connector_target_id(
    connector_dict: dict[str, Any],
    classifier_ids_by_name: dict[str, str],
) -> str | None:
    for key in ("target_id", "targetid", "targetId"):
        value = connector_dict.get(key)
        if _is_non_empty(value):
            return str(value).strip()

    target_name = _connector_target_name(connector_dict)
    if target_name:
        return classifier_ids_by_name.get(_canonical_name(target_name))

    return None


def _pick_association_parent(
    model: Element,
    connector_dict: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
    package_elements_by_id: dict[str, Element],
    classifier_ids_by_name: dict[str, str],
) -> Element:
    source_id = _connector_source_id(connector_dict, classifier_ids_by_name)
    target_id = _connector_target_id(connector_dict, classifier_ids_by_name)

    source_pkg = elements_by_id.get(source_id, {}).get("package") if source_id else None
    target_pkg = elements_by_id.get(target_id, {}).get("package") if target_id else None

    if _is_non_empty(source_pkg) and source_pkg == target_pkg and str(source_pkg) in package_elements_by_id:
        return package_elements_by_id[str(source_pkg)]

    if _is_non_empty(source_pkg) and str(source_pkg) in package_elements_by_id:
        return package_elements_by_id[str(source_pkg)]

    if _is_non_empty(target_pkg) and str(target_pkg) in package_elements_by_id:
        return package_elements_by_id[str(target_pkg)]

    return model


def _add_generalization(
    source_element: Element,
    connector_dict: dict[str, Any],
    general_id: str,
    target_id: str,
) -> None:
    gen = SubElement(source_element, "generalization")
    gen.set(_xmi_attr("id"), general_id)
    gen.set(_xmi_attr("type"), "uml:Generalization")
    gen.set("general", target_id)

    _add_comment(
        gen,
        _build_connector_comment(connector_dict),
        f"{general_id}_comment",
        annotated_id=general_id,
    )


def _end_role_name(raw_role: Any, fallback_classifier_name: str, fallback_index: int) -> str:
    if _is_non_empty(raw_role):
        return str(raw_role).strip()
    if fallback_classifier_name:
        name = fallback_classifier_name[:1].lower() + fallback_classifier_name[1:]
        return name or f"end_{fallback_index}"
    return f"end_{fallback_index}"


def _add_association_owned_end(
    association: Element,
    end_id: str,
    classifier_id: str,
    role_name: str | None,
    multiplicity_raw: Any,
    *,
    association_id: str,
    visibility: str | None = None,
    aggregation: str | None = None,
    property_flags: dict[str, Any] | None = None,
) -> Element:
    end = SubElement(association, "ownedEnd")
    end.set(_xmi_attr("id"), end_id)
    end.set(_xmi_attr("type"), "uml:Property")
    end.set("type", classifier_id)
    end.set("association", association_id)

    if _is_non_empty(role_name):
        end.set("name", str(role_name).strip())

    _set_property_attrs(
        end,
        {**(property_flags or {}), "visibility": visibility},
        default_visibility="public",
    )

    if aggregation in {"none", "shared", "composite"}:
        end.set("aggregation", aggregation)

    _add_multiplicity(end, multiplicity_raw, end_id)
    return end


def _add_association(
    parent: Element,
    connector_dict: dict[str, Any],
    classifier_ids_by_name: dict[str, str],
    classifier_xml_by_id: dict[str, Element],
    classifier_names_by_id: dict[str, str],
    index: int,
) -> None:
    source_id = _connector_source_id(connector_dict, classifier_ids_by_name)
    target_id = _connector_target_id(connector_dict, classifier_ids_by_name)

    if not source_id or not target_id:
        return

    source_xml = classifier_xml_by_id.get(source_id)
    target_xml = classifier_xml_by_id.get(target_id)
    if source_xml is None or target_xml is None:
        return

    assoc_id = _connector_id(connector_dict, index)
    relationship = _relationship_kind(connector_dict)

    assoc = SubElement(parent, "packagedElement")
    assoc.set(_xmi_attr("id"), assoc_id)
    assoc.set(_xmi_attr("type"), "uml:Association")
    assoc.set("name", _association_name(connector_dict, index))
    _set_visibility(assoc, connector_dict.get("visibility"), default="public")

    source_end_id = f"{assoc_id}_source_end"
    target_end_id = f"{assoc_id}_target_end"
    assoc.set("memberEnd", f"{source_end_id} {target_end_id}")

    source_role = _end_role_name(
        connector_dict.get("lt"),
        classifier_names_by_id.get(source_id, _connector_source_name(connector_dict)),
        1,
    )
    target_role = _end_role_name(
        connector_dict.get("rt"),
        classifier_names_by_id.get(target_id, _connector_target_name(connector_dict)),
        2,
    )

    source_flags = {
        "isOrdered": connector_dict.get("lb_isOrdered"),
        "isUnique": connector_dict.get("lb_isUnique"),
        "isReadOnly": connector_dict.get("lb_isReadOnly"),
        "isDerived": connector_dict.get("lb_isDerived"),
        "isStatic": connector_dict.get("lb_isStatic"),
    }
    target_flags = {
        "isOrdered": connector_dict.get("rb_isOrdered"),
        "isUnique": connector_dict.get("rb_isUnique"),
        "isReadOnly": connector_dict.get("rb_isReadOnly"),
        "isDerived": connector_dict.get("rb_isDerived"),
        "isStatic": connector_dict.get("rb_isStatic"),
    }

    aggregation_value = "none"
    if relationship == "aggregation":
        aggregation_value = "shared"
    elif relationship == "composition":
        aggregation_value = "composite"

    _add_association_owned_end(
        association=assoc,
        end_id=source_end_id,
        classifier_id=source_id,
        role_name=source_role,
        multiplicity_raw=connector_dict.get("lb"),
        association_id=assoc_id,
        visibility=connector_dict.get("lt_visibility"),
        aggregation="none",
        property_flags=source_flags,
    )

    _add_association_owned_end(
        association=assoc,
        end_id=target_end_id,
        classifier_id=target_id,
        role_name=target_role,
        multiplicity_raw=connector_dict.get("rb"),
        association_id=assoc_id,
        visibility=connector_dict.get("rt_visibility"),
        aggregation=aggregation_value,
        property_flags=target_flags,
    )

    _add_comment(
        assoc,
        _build_connector_comment(connector_dict),
        f"{assoc_id}_comment",
        annotated_id=assoc_id,
    )


def json_to_xml(json_data: dict[str, Any]) -> bytes:
    register_namespace("xmi", XMI_NS)
    register_namespace("uml", UML_NS)

    root = Element(_xmi_attr("XMI"))
    root.set(_xmi_attr("version"), "2.1")

    documentation = SubElement(root, _xmi_attr("Documentation"))
    documentation.set("exporter", "Custom UML 2 XMI Exporter")
    documentation.set("exporterVersion", "3.0")

    model = SubElement(root, f"{{{UML_NS}}}Model")
    model.set(_xmi_attr("id"), _safe_name(json_data.get("model_id"), "model_1"))
    model.set(_xmi_attr("type"), "uml:Model")
    model.set("name", _safe_name(json_data.get("model_name"), "UML_Model"))
    model.set("visibility", "public")

    elements = _safe_list(json_data.get("elements"))
    connectors = _safe_list(json_data.get("connectors"))

    package_elements_by_id: dict[str, Element] = {}
    elements_by_id: dict[str, dict[str, Any]] = {}
    classifier_ids_by_id: dict[str, str] = {}
    classifier_ids_by_name: dict[str, str] = {}
    classifier_xml_by_id: dict[str, Element] = {}
    classifier_names_by_id: dict[str, str] = {}

    for element_dict in elements:
        if not isinstance(element_dict, dict):
            continue

        element_id = _safe_name(element_dict.get("ID"), "")
        element_type = str(element_dict.get("type", "")).strip()
        element_name = _safe_name(element_dict.get("name"), element_id)

        if element_id:
            elements_by_id[element_id] = element_dict

        if element_type in {"uml:Class", "uml:DataType", "uml:Enumeration"} and element_id:
            classifier_ids_by_id[element_id] = element_id
            classifier_names_by_id[element_id] = element_name
            if element_name:
                classifier_ids_by_name[_canonical_name(element_name)] = element_id

    remaining = [e for e in elements if isinstance(e, dict)]
    max_passes = max(1, len(remaining) * 2)
    current_pass = 0

    while remaining and current_pass < max_passes:
        current_pass += 1
        next_remaining: list[dict[str, Any]] = []
        progressed = False

        for element_dict in remaining:
            element_type = str(element_dict.get("type", "")).strip()
            element_id = _safe_name(element_dict.get("ID"), "")
            package_id = element_dict.get("package")

            parent = model
            if _is_non_empty(package_id):
                parent = package_elements_by_id.get(str(package_id))
                if parent is None:
                    next_remaining.append(element_dict)
                    continue

            if element_type == "uml:Package":
                pkg = _add_package(parent, element_dict)
                if element_id:
                    package_elements_by_id[element_id] = pkg
                progressed = True

            elif element_type == "uml:Class":
                cls = _add_class_like(
                    parent,
                    element_dict,
                    "uml:Class",
                    classifier_ids_by_id=classifier_ids_by_id,
                    classifier_ids_by_name=classifier_ids_by_name,
                )
                if element_id:
                    classifier_xml_by_id[element_id] = cls
                progressed = True

            elif element_type == "uml:DataType":
                dt = _add_class_like(
                    parent,
                    element_dict,
                    "uml:DataType",
                    classifier_ids_by_id=classifier_ids_by_id,
                    classifier_ids_by_name=classifier_ids_by_name,
                )
                if element_id:
                    classifier_xml_by_id[element_id] = dt
                progressed = True

            elif element_type == "uml:Enumeration":
                enum = _add_enumeration(parent, element_dict)
                if element_id:
                    classifier_xml_by_id[element_id] = enum
                progressed = True

            else:
                next_remaining.append(element_dict)

        if not progressed:
            for element_dict in next_remaining:
                element_type = str(element_dict.get("type", "")).strip()
                element_id = _safe_name(element_dict.get("ID"), "")

                if element_type == "uml:Package":
                    pkg = _add_package(model, element_dict)
                    if element_id:
                        package_elements_by_id[element_id] = pkg

                elif element_type == "uml:Class":
                    cls = _add_class_like(
                        model,
                        element_dict,
                        "uml:Class",
                        classifier_ids_by_id=classifier_ids_by_id,
                        classifier_ids_by_name=classifier_ids_by_name,
                    )
                    if element_id:
                        classifier_xml_by_id[element_id] = cls

                elif element_type == "uml:DataType":
                    dt = _add_class_like(
                        model,
                        element_dict,
                        "uml:DataType",
                        classifier_ids_by_id=classifier_ids_by_id,
                        classifier_ids_by_name=classifier_ids_by_name,
                    )
                    if element_id:
                        classifier_xml_by_id[element_id] = dt

                elif element_type == "uml:Enumeration":
                    enum = _add_enumeration(model, element_dict)
                    if element_id:
                        classifier_xml_by_id[element_id] = enum
            break

        remaining = next_remaining

    for index, connector_dict in enumerate(connectors, start=1):
        if not isinstance(connector_dict, dict):
            continue

        relationship = _relationship_kind(connector_dict)
        source_id = _connector_source_id(connector_dict, classifier_ids_by_name)
        target_id = _connector_target_id(connector_dict, classifier_ids_by_name)

        if not source_id or not target_id:
            continue

        if relationship == "generalization":
            source_element = classifier_xml_by_id.get(source_id)
            if source_element is None:
                continue
            _add_generalization(
                source_element=source_element,
                connector_dict=connector_dict,
                general_id=_connector_id(connector_dict, index),
                target_id=target_id,
            )
            continue

        if relationship in {"association", "aggregation", "composition"}:
            assoc_parent = _pick_association_parent(
                model=model,
                connector_dict=connector_dict,
                elements_by_id=elements_by_id,
                package_elements_by_id=package_elements_by_id,
                classifier_ids_by_name=classifier_ids_by_name,
            )
            _add_association(
                parent=assoc_parent,
                connector_dict=connector_dict,
                classifier_ids_by_name=classifier_ids_by_name,
                classifier_xml_by_id=classifier_xml_by_id,
                classifier_names_by_id=classifier_names_by_id,
                index=index,
            )

    return tostring(root, encoding="utf-8", xml_declaration=True)
