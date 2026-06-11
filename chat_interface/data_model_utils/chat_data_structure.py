# chat_data_structure.py
# This module transforms the user data model into a JSON format that is understandable by the LLM.
# It provides utility functions to extract and simplify relevant information from UML-like data structures
# (packages, classes, datatypes, enumerations, connectors) for downstream processing by language models.

def _shorten_package(elem):
    """
    Extracts and simplifies relevant fields from a package element for LLM consumption.
    Keeps the name, root URI, and tags containing 'definition' or 'uri'.
    Args:
        elem (dict): The package element dictionary.
    Returns:
        dict: A simplified package dictionary.
    """
    package_dict = {}
    package_dict["name"] = elem.get("name")

    if "uri" in elem:
        package_dict["uri"] = elem["uri"]

    tags = []
    for tag in elem.get("tags", []):
        tag_name = str(tag.get("name", "")).lower()
        if "definition" in tag_name or "uri" in tag_name:
            tags.append({
                "name": tag.get("name"),
                "value": tag.get("value")
            })

    package_dict["tags"] = tags
    return package_dict


def _shorten_class(elem):
    """
    Extracts and simplifies relevant fields from a class element for LLM consumption.
    Keeps the name, root URI, filtered tags, and simplified attributes.
    Args:
        elem (dict): The class element dictionary.
    Returns:
        dict: A simplified class dictionary.
    """
    class_dict = {}
    class_dict["name"] = elem.get("name")

    if "uri" in elem:
        class_dict["uri"] = elem["uri"]

    tags = []
    for tag in elem.get("tags", []):
        tag_name = str(tag.get("name", "")).lower()
        if "definition" in tag_name or "uri" in tag_name:
            tags.append({
                "name": tag.get("name"),
                "value": tag.get("value")
            })
    class_dict["tags"] = tags

    attributes = []
    try:
        for attribute in elem.get("attributes", []):
            attribute_dict = {}
            attribute_dict["name"] = attribute.get("name")
            attribute_dict["type"] = attribute.get("type")

            if "uri" in attribute:
                attribute_dict["uri"] = attribute["uri"]

            if "lower_bounds" in attribute:
                attribute_dict["lower_bounds"] = attribute["lower_bounds"]
            if "upper_bounds" in attribute:
                attribute_dict["upper_bounds"] = attribute["upper_bounds"]

            # Filter tags on the attribute level to avoid token bloat
            attr_tags = []
            for tag in attribute.get("tags_attribute", []):
                tag_name = str(tag.get("name", "")).lower()
                if "definition" in tag_name or "uri" in tag_name:
                    attr_tags.append({
                        "name": tag.get("name"),
                        "value": tag.get("value")
                    })
            attribute_dict["tags"] = attr_tags
            attributes.append(attribute_dict)

        class_dict["attributes"] = attributes
    except Exception:
        pass

    return class_dict


def _shorten_datatype(elem):
    """
    Extracts and simplifies relevant fields from a datatype element for LLM consumption.
    Keeps the name, root URI, filtered tags, and attributes.
    Args:
        elem (dict): The datatype element dictionary.
    Returns:
        dict: A simplified datatype dictionary.
    """
    datatype_dict = {}
    datatype_dict["name"] = elem.get("name")

    if "uri" in elem:
        datatype_dict["uri"] = elem["uri"]

    tags = []
    for tag in elem.get("tags", []):
        tag_name = str(tag.get("name", "")).lower()
        if "definition" in tag_name or "uri" in tag_name:
            tags.append({
                "name": tag.get("name"),
                "value": tag.get("value")
            })
    datatype_dict["tags"] = tags

    attributes = []
    try:
        for attribute in elem.get("attributes", []):
            attribute_dict = {}
            attribute_dict["name"] = attribute.get("name")
            attribute_dict["type"] = attribute.get("type")

            if "uri" in attribute:
                attribute_dict["uri"] = attribute["uri"]

            if "lower_bounds" in attribute:
                attribute_dict["lower_bounds"] = attribute["lower_bounds"]
            if "upper_bounds" in attribute:
                attribute_dict["upper_bounds"] = attribute["upper_bounds"]

            attributes.append(attribute_dict)

        datatype_dict["attributes"] = attributes
    except Exception:
        pass

    return datatype_dict


def _shorten_enum(elem):
    """
    Extracts and simplifies relevant fields from an enumeration element for LLM consumption.
    Keeps the name, root URI, filtered tags, and categories.
    Args:
        elem (dict): The enumeration element dictionary.
    Returns:
        dict: A simplified enumeration dictionary.
    """
    enum_dict = {}
    enum_dict["name"] = elem.get("name")

    if "uri" in elem:
        enum_dict["uri"] = elem["uri"]

    tags = []
    for tag in elem.get("tags", []):
        tag_name = str(tag.get("name", "")).lower()
        if "definition" in tag_name or "uri" in tag_name:
            tags.append({
                "name": tag.get("name"),
                "value": tag.get("value")
            })
    enum_dict["tags"] = tags

    try:
        enum_dict["categories"] = elem.get("categories", [])
    except Exception:
        pass

    return enum_dict


def _shorten_elements(elements):
    """
    Processes a list of elements and sorts them into packages, classes, datatypes, and enumerations,
    applying the appropriate shortening function to each.
    Args:
        elements (list): List of element dictionaries.
    Returns:
        dict: Dictionary with keys 'packages', 'classes', 'datatypes', 'enumerations'.
    """
    packages = []
    classes = []
    datatypes = []
    enumerations = []

    for elem in elements:
        elem_type = elem.get("type")
        if elem_type == "uml:Package":
            packages.append(_shorten_package(elem))
        elif elem_type == "uml:Class":
            classes.append(_shorten_class(elem))
        elif elem_type == "uml:DataType":
            datatypes.append(_shorten_datatype(elem))
        elif elem_type == "uml:Enumeration":
            enumerations.append(_shorten_enum(elem))
        else:
            print(f"ERROR SHORTEN: {elem}")

    return {
        "packages": packages,
        "classes": classes,
        "datatypes": datatypes,
        "enumerations": enumerations,
    }


def _shorten_connector(conn):
    """
    Extracts and simplifies relevant fields from a connector element for LLM consumption.
    Keeps source/target names, relationship type, relation name (label), URI, and bounds.
    Args:
        conn (dict): The connector element dictionary.
    Returns:
        dict: A simplified connector dictionary.
    """
    conn_dict = {}
    conn_dict["source_name"] = conn.get("source_name")
    conn_dict["target_name"] = conn.get("target_name")
    conn_dict["relationship"] = conn.get("relationship")

    if "name" in conn:
        conn_dict["name"] = conn["name"]
    if "uri" in conn:
        conn_dict["uri"] = conn["uri"]

    if conn.get("lb") is not None:
        conn_dict["lb"] = conn["lb"]
    if conn.get("lt") is not None:
        conn_dict["lt"] = conn["lt"]
    if conn.get("rb") is not None:
        conn_dict["rb"] = conn["rb"]
    if conn.get("rt") is not None:
        conn_dict["rt"] = conn["rt"]

    return conn_dict


def shorten_json(json_data):
    """
    Transforms the full user data model into a simplified JSON format for LLM input.
    Processes elements and connectors using the above utility functions.
    Args:
        json_data (dict): The original user data model as a dictionary.
    Returns:
        dict: The transformed, LLM-ready data model.
    """
    data_model = {}

    data_model["elements"] = _shorten_elements(json_data.get("elements", []))

    connectors = []
    for conn in json_data.get("connectors", []):
        connectors.append(_shorten_connector(conn))

    data_model["connectors"] = connectors
    return data_model
