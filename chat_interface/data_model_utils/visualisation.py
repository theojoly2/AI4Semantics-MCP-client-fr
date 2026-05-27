from typing import Any
from io import BytesIO
import json
import re
from collections import defaultdict
import plantuml
from plantuml import PlantUML, PlantUMLConnectionError


class FixedPlantUMLHTTPError(PlantUMLConnectionError):
    def __init__(self, response, content, *args, **kwdargs):
        self.response = response
        self.content = content
        message = "%d: %s" % (self.response.status, self.response.reason)
        self.message = message
        super().__init__(message, *args, **kwdargs)


plantuml.PlantUMLHTTPError = FixedPlantUMLHTTPError


def _alias_from_id(element_id: str) -> str:
    return "cls_" + re.sub(r"[^A-Za-z0-9_]", "_", str(element_id))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _strip_visibility(value: str) -> str:
    value = _safe_text(value)
    if value[:1] in {"+", "-", "#", "~"}:
        return value[1:].strip()
    return value


def _escape_display(value: str) -> str:
    return _safe_text(value).replace('"', '\\"')


def _get_tag_value(tags: list[dict[str, Any]] | None, tag_name: str) -> str:
    for tag in tags or []:
        if tag.get("name") == tag_name:
            return _safe_text(tag.get("value"))
    return ""


def _get_center_label(connector: dict[str, Any]) -> str:
    # On privilégie un label central pour libérer les extrémités
    # afin d'y afficher les cardinalités quand elles existent.
    for key in ("name", "label", "relation_name"):
        if connector.get(key):
            return _safe_text(connector.get(key))

    rt = _strip_visibility(connector.get("rt"))
    if rt:
        return rt

    label_en = _get_tag_value(connector.get("tags_target"), "label-en")
    if label_en:
        return label_en

    return ""


def _resolve_alias(
    ref_id: str,
    ref_name: str,
    id_to_alias: dict[str, str],
    name_to_aliases: dict[str, list[str]],
    endpoint_kind: str,
    debug: bool = False,
) -> str:
    if ref_id:
        alias = id_to_alias.get(ref_id)
        if alias:
            return alias
        if debug:
            print(f"[WARN] {endpoint_kind}_id introuvable: {ref_id} -> fallback sur le nom")

    matches = name_to_aliases.get(ref_name, [])

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        chosen = matches[0]
        if debug:
            print(
                f"[WARN] {endpoint_kind}_name ambigu: '{ref_name}' "
                f"-> fallback sur le premier alias déclaré: {chosen}"
            )
        return chosen

    synthetic = f'{endpoint_kind}_missing_' + re.sub(r"[^A-Za-z0-9_]", "_", ref_name or "unknown")
    if debug:
        print(
            f"[WARN] {endpoint_kind}_name introuvable: '{ref_name}' "
            f"-> création d'un placeholder: {synthetic}"
        )
    return synthetic


def get_image_bytes(
    source_json: dict[str, list[dict[str, Any]]],
    debug: bool = False,
) -> BytesIO:
    if debug:
        print("SOURCE JSON:")
        print(json.dumps(source_json, indent=2, ensure_ascii=False))

    elements = source_json.get("elements", [])
    connectors = source_json.get("connectors", [])

    plantuml_lines: list[str] = []
    plantuml_lines.append("@startuml")
    plantuml_lines.append("left to right direction")
    plantuml_lines.append("hide empty members")

    classes: list[dict[str, Any]] = []

    for element in elements:
        if element.get("type") == "uml:Class":
            classes.append(element)

    id_to_alias: dict[str, str] = {}
    name_to_aliases: dict[str, list[str]] = defaultdict(list)

    for element in classes:
        element_id = _safe_text(element.get("ID"))
        element_name = _safe_text(element.get("name"))
        alias = _alias_from_id(element_id or element_name)

        id_to_alias[element_id] = alias
        name_to_aliases[element_name].append(alias)

    def render_class(alias: str, element: dict[str, Any]) -> list[str]:
        display_name = _escape_display(_safe_text(element.get("name")))
        attributes = element.get("attributes", []) or []

        lines: list[str] = [f'class "{display_name}" as {alias} {{']
        for attr in attributes:
            attr_name = _safe_text(attr.get("name"))
            attr_type = _safe_text(attr.get("type")) or "Any"
            lines.append(f"  {attr_name}: {attr_type}")
        lines.append("}")
        return lines

    # Groupement sans encadrement
    plantuml_lines.append("together {")
    for element in classes:
        alias = id_to_alias[_safe_text(element.get("ID"))]
        plantuml_lines.extend(render_class(alias, element))
    plantuml_lines.append("}")

    placeholders: set[str] = set()

    for connector in connectors:
        source_id = _safe_text(connector.get("source_id"))
        target_id = _safe_text(connector.get("target_id"))
        source_name = _safe_text(connector.get("source_name"))
        target_name = _safe_text(connector.get("target_name"))
        relationship = _safe_text(connector.get("relationship"))

        source_alias = _resolve_alias(
            source_id, source_name, id_to_alias, name_to_aliases, "source", debug=debug
        )
        target_alias = _resolve_alias(
            target_id, target_name, id_to_alias, name_to_aliases, "target", debug=debug
        )

        if source_alias.startswith("source_missing_") and source_alias not in placeholders:
            plantuml_lines.append(f'class "{_escape_display(source_name)}" as {source_alias}')
            placeholders.add(source_alias)

        if target_alias.startswith("target_missing_") and target_alias not in placeholders:
            plantuml_lines.append(f'class "{_escape_display(target_name)}" as {target_alias}')
            placeholders.add(target_alias)

        # Cardinalités : uniquement depuis lt / rt
        left_card = _safe_text(connector.get("lt"))
        right_card = _safe_text(connector.get("rt"))

        # Nom de relation au centre, pour ne pas bloquer l'affichage des cardinalités
        center_label = _get_center_label(connector)

        if relationship == "Association":
            arrow = "--"
        elif relationship == "Generalization":
            arrow = "<|--"
        elif relationship == "Aggregation":
            arrow = "o--"
        elif relationship == "Composition":
            arrow = "*--"
        else:
            arrow = "--"

        left_txt = f' "{_escape_display(left_card)}"' if left_card else ""
        right_txt = f' "{_escape_display(right_card)}"' if right_card else ""
        center_txt = f" : {_escape_display(center_label)}" if center_label else ""

        plantuml_lines.append(
            f"{source_alias}{left_txt} {arrow}{right_txt} {target_alias}{center_txt}"
        )

    plantuml_lines.append("@enduml")
    plantuml_text = "\n".join(plantuml_lines)

    if debug:
        print("\nPLANTUML:")
        print(plantuml_text)

    server = PlantUML(url="https://www.plantuml.com/plantuml/img/")
    image_bytes = server.processes(plantuml_text)
    return BytesIO(image_bytes)
