from typing import (
    Any,
)
from plantuml import PlantUML
from io import (
    BytesIO,
)


# takes in the json data and returns the image bytes of the uml diagram visual
def get_image_bytes(
    source_json: dict[str, list[dict[str, Any]]],
) -> BytesIO:
    plantuml_text: str = ""

    elements = source_json["elements"]
    connectors = source_json["connectors"]

    title: str = "uml_diagram"
    plantuml_text += f"@startuml {title} \n"

    for element in elements:
        element["name"] = element["name"].replace("-", "_")
        element["name"] = element["name"].replace(" ", "_")
        element["name"] = element["name"].replace(":", "_")
        if element["type"] == "uml:Class":
            name = element["name"]
            element_type = element["type"]

            attribute_name: list[str] = []
            attribute_type: list[str] = []
            if element.get("attributes"):
                for i in range(0, len(element["attributes"])):
                    attribute_name.append(element["attributes"][i]["name"])
                    attribute_type.append(element["attributes"][i]["type"])

                for i in range(0, len(attribute_name)):
                    if i == 0:
                        plantuml_text += f" class {name} {{\n"

                    plantuml_text += f" {attribute_name[i]}: {attribute_type[i]} \n"
                    if i == len(attribute_name) - 1:
                        plantuml_text += " }\n"

            else:
                plantuml_text += f" class {name} \n"

    for connector in connectors:
        connector["source_name"] = connector["source_name"].replace("-", "_")
        connector["source_name"] = connector["source_name"].replace(" ", "_")
        connector["source_name"] = connector["source_name"].replace(":", "_")
        source_name = connector["source_name"]
        source_bound = connector["lb"]
        connector["target_name"] = connector["target_name"].replace("-", "_")
        connector["target_name"] = connector["target_name"].replace(" ", "_")
        connector["target_name"] = connector["target_name"].replace(":", "_")
        target_name = connector["target_name"]
        target_bound = connector["rb"]

        if connector["relationship"] == "Association":
            plantuml_text += f" {target_name} -- {source_name} \n"

        if connector["relationship"] == "Generalization":
            plantuml_text += f" {target_name} <|-- {source_name} \n"

        if connector["relationship"] == "Aggregation":
            plantuml_text += f" {target_name} --> {source_name} \n"

    plantuml_text += "@enduml\n"

    server = PlantUML(url="http://www.plantuml.com/plantuml/img/")
    image_bytes = server.processes(plantuml_text)
    image_stream = BytesIO(image_bytes)

    return image_stream


