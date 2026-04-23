# chat_data_structure.py
# This module transforms the user data model into a JSON format that is understandable by the LLM.
# It provides utility functions to extract and simplify relevant information from UML-like data structures
# (packages, classes, datatypes, enumerations, connectors) for downstream processing by language models.


def _shorten_package(elem):
    """
    Extracts and simplifies relevant fields from a package element for LLM consumption.
    Only keeps the name and tags containing 'definition' or 'uri'.
    Args:
        elem (dict): The package element dictionary.
    Returns:
        dict: A simplified package dictionary.
    """
    package_dict = {}
    package_dict["name"] = elem["name"]

    tags = []
    for tag in elem["tags"]:
        if "definition" in tag["name"] or "uri" in tag["name"]:
            tag_dict = {}
            tag_dict["name"] = tag["name"]
            tag_dict["value"] = tag["value"]
            tags.append(tag_dict)

    package_dict["tags"] = tags

    return package_dict


def _shorten_class(elem):
    """
    Extracts and simplifies relevant fields from a class element for LLM consumption.
    Keeps the name, filtered tags, and attributes (name, type, tags).
    Args:
        elem (dict): The class element dictionary.
    Returns:
        dict: A simplified class dictionary.
    """
    class_dict = {}
    class_dict["name"] = elem["name"]

    tags = []
    for tag in elem["tags"]:
        if "definition" in tag["name"] or "uri" in tag["name"]:
            tag_dict = {}
            tag_dict["name"] = tag["name"]
            tag_dict["value"] = tag["value"]
            tags.append(tag_dict)
    class_dict["tags"] = tags

    try:
        attributes = []
        for attribute in elem["attributes"]:
            attribute_dict = {}
            attribute_dict["name"] = attribute["name"]
            attribute_dict["type"] = attribute["type"]
            #attribute_dict["lower_bounds"] = attribute["lower_bounds"]
            #attribute_dict["upper_bounds"] = attribute["upper_bounds"]
            attribute_dict["tags"] = attribute["tags_attribute"]
            attributes.append(attribute_dict)

        class_dict["attributes"] = attributes

    except Exception:
        pass

    return class_dict


def _shorten_datatype(elem):
    """
    Extracts and simplifies relevant fields from a datatype element for LLM consumption.
    Keeps the name, filtered tags, and attributes (name, type, lower_bounds, upper_bounds).
    Args:
        elem (dict): The datatype element dictionary.
    Returns:
        dict: A simplified datatype dictionary.
    """
    datatype_dict = {}
    datatype_dict["name"] = elem["name"]

    tags = []
    for tag in elem["tags"]:
        if "definition" in tag["name"] or "uri" in tag["name"]:
            tag_dict = {}
            tag_dict["name"] = tag["name"]
            tag_dict["value"] = tag["value"]
            tags.append(tag_dict)
    datatype_dict["tags"] = tags

    try:
        attributes = []
        for attribute in elem["attributes"]:
            attribute_dict = {}
            attribute_dict["name"] = attribute["name"]
            attribute_dict["type"] = attribute["type"]
            attribute_dict["lower_bounds"] = attribute["lower_bounds"]
            attribute_dict["upper_bounds"] = attribute["upper_bounds"]

        datatype_dict["attributes"] = attributes

    except Exception:
        pass

    return datatype_dict


def _shorten_enum(elem):
    """
    Extracts and simplifies relevant fields from an enumeration element for LLM consumption.
    Keeps the name, filtered tags, and categories.
    Args:
        elem (dict): The enumeration element dictionary.
    Returns:
        dict: A simplified enumeration dictionary.
    """
    enum_dict = {}
    enum_dict["name"] = elem["name"]

    tags = []
    for tag in elem["tags"]:
        if "definition" in tag["name"] or "uri" in tag["name"]:
            tag_dict = {}
            tag_dict["name"] = tag["name"]
            tag_dict["value"] = tag["value"]
            tags.append(tag_dict)
    enum_dict["tags"] = tags
    try:
        enum_dict["categories"] = elem["categories"]

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
        if elem["type"] == "uml:Package":
            packages.append(_shorten_package(elem))
        elif elem["type"] == "uml:Class":
            classes.append(_shorten_class(elem))
        elif elem["type"] == "uml:DataType":
            datatypes.append(_shorten_datatype(elem))
        elif elem["type"] == "uml:Enumeration":
            enumerations.append(_shorten_enum(elem))
        else:
            print(f"ERROR SHORTEN: {elem}")

    elements = {
        "packages": packages,
        "classes": classes,
        "datatypes": datatypes,
        "enumerations": enumerations,
    }

    return elements


def _shorten_connector(conn):
    """
    Extracts and simplifies relevant fields from a connector element for LLM consumption.
    Keeps source/target names, relationship, and bounds if present.
    Args:
        conn (dict): The connector element dictionary.
    Returns:
        dict: A simplified connector dictionary.
    """
    conn_dict = {}
    conn_dict["source_name"] = conn["source_name"]
    conn_dict["target_name"] = conn["target_name"]
    conn_dict["relationship"] = conn["relationship"]
    if conn["lb"] is not None:
        conn_dict["lb"] = conn["lb"]
    if conn["lt"] is not None:
        conn_dict["lt"] = conn["lt"]
    if conn["rb"] is not None:
        conn_dict["rb"] = conn["rb"]
    if conn["rt"] is not None:
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

    data_model["elements"] = _shorten_elements(json_data["elements"])

    connectors = []
    for conn in json_data["connectors"]:
        connectors.append(_shorten_connector(conn))

    data_model["connectors"] = connectors

    return data_model
