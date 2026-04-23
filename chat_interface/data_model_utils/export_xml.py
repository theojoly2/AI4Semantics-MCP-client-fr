from typing import (
    Any,
)
from xml.etree.ElementTree import (
    Element,
    SubElement,
    fromstring,
    register_namespace,
    tostring,
)


def _add_packaged_element(
    element_dict: dict[str, Any]
) -> Element:
    """
    Creates an XML 'packagedElement' for a given element.

    :param element_dict: dictionary containing the element's attributes.
    :return: An XML element representing the packaged element.
    """
    element = Element('packagedElement')
    element.set('xmi:id', element_dict["ID"])
    element.set('xmi:type', element_dict["type"])
    element.set('name', element_dict["name"])
    return element


def _add_package(
    element_dict: dict[str, Any],
) -> Element:
    """
    Creates an XML representation for a UML package element.

    :param element_dict: dictionary containing the package's details and tags.
    :return: An XML element representing the UML package.
    """
    element = Element('element')
    element.set('xmi:idref', element_dict["ID"])
    element.set('xmi:type', element_dict["type"])
    element.set('name', element_dict["name"])

    model = SubElement(element, 'model')
    model.set("package", element_dict["package"])

    tags = SubElement(element, 'tags')
    for tag_dict in element_dict["tags"]:
        tag = SubElement(tags, 'tag')
        tag.set("name", tag_dict["name"])
        tag.set("value", tag_dict["value"])
        tag.set("modelElement", element_dict["ID"])

    return element


def _add_class(
    element_dict: dict[str, Any],
) -> Element:
    """
    Creates an XML representation for a UML class element, including attributes and tags.

    :param element_dict: dictionary containing the class's details, attributes, and tags.
    :return: An XML element representing the UML class.
    """
    element = Element('element')
    element.set('xmi:idref', element_dict["ID"])
    element.set('xmi:type', element_dict["type"])
    element.set('name', element_dict["name"])

    model = SubElement(element, 'model')
    model.set("package", element_dict["package"])

    tags = SubElement(element, 'tags')
    for tag_dict in element_dict["tags"]:
        tag = SubElement(tags, 'tag')
        tag.set("name", tag_dict["name"])
        tag.set("value", tag_dict["value"])
        tag.set("modelElement", element_dict["ID"])

    try:
        attributes = SubElement(element, 'attributes')
        for attribute_dict in element_dict['attributes']:
            attribute = SubElement(attributes, 'attribute')
            attribute.set("name", attribute_dict["name"])

            if attribute_dict.get('type') is not None:
                properties = SubElement(attribute, 'properties')
                properties.set('type', attribute_dict['type'])

            if attribute_dict.get('lower_bounds') is not None:
                bounds = SubElement(attribute, 'bounds')
                bounds.set("lower", attribute_dict['lower_bounds'])
                bounds.set('upper', attribute_dict['upper_bounds'])

            if attribute_dict.get('tags_attribute') is not None:
                tags = SubElement(attribute, 'tags')
                for tag_attribute_dict in attribute_dict['tags_attribute']:
                    tag = SubElement(tags, 'tag')
                    tag.set('name', tag_attribute_dict["name"])
                    tag.set('value', tag_attribute_dict['value'])

    except KeyError:
        pass

    return element


def _add_datatype(
    element_dict: dict[str, Any],
) -> Element:
    """
    Creates an XML representation for a UML datatype element, including attributes and tags.

    :param element_dict: dictionary containing the datatype's details, attributes, and tags.
    :return: An XML element representing the UML datatype.
    """
    element = Element('element')
    element.set('xmi:idref', element_dict["ID"])
    element.set('xmi:type', element_dict["type"])
    element.set('name', element_dict["name"])

    model = SubElement(element, 'model')
    model.set("package", element_dict["package"])

    tags = SubElement(element, 'tags')
    for tag_dict in element_dict["tags"]:
        tag = SubElement(tags, 'tag')
        tag.set("name", tag_dict["name"])
        tag.set("value", tag_dict["value"])
        tag.set("modelElement", element_dict["ID"])

    try:
        attributes = SubElement(element, 'attributes')
        for attribute_dict in element_dict['attributes']:
            attribute = SubElement(attributes, 'attribute')
            attribute.set("name", attribute_dict["name"])

            if attribute_dict.get('type') is not None:
                properties = SubElement(attribute, 'properties')
                properties.set('type', attribute_dict['type'])

            if attribute_dict.get('lower_bounds') is not None:
                bounds = SubElement(attribute, 'bounds')
                bounds.set("lower", attribute_dict['lower_bounds'])
                bounds.set('upper', attribute_dict['upper_bounds'])

            if attribute_dict.get('tags_attribute') is not None:
                tags = SubElement(attribute, 'tags')
                for tag_attribute_dict in attribute_dict['tags_attribute']:
                    tag = SubElement(tags, 'tag')
                    tag.set('name', tag_attribute_dict["name"])
                    tag.set('value', tag_attribute_dict['value'])

    except KeyError:
        pass

    return element


def _add_enumeration(
    element_dict: dict[str, Any],
) -> Element:
    """
    Creates an XML representation for a UML enumeration element.

    :param element_dict: dictionary containing the enumeration's details, categories, and tags.
    :return: An XML element representing the UML enumeration.
    """
    element = Element('element')
    element.set('xmi:idref', element_dict["ID"])
    element.set('xmi:type', element_dict["type"])
    element.set('name', element_dict["name"])

    model = SubElement(element, 'model')
    model.set("package", element_dict["package"])

    properties = SubElement(element, "properties")
    properties.set("sType", "Enumeration")

    tags = SubElement(element, 'tags')
    for tag_dict in element_dict["tags"]:
        tag = SubElement(tags, 'tag')
        tag.set("name", tag_dict["name"])
        tag.set("value", tag_dict["value"])
        tag.set("modelElement", element_dict["ID"])

    try:
        attributes = SubElement(element, 'attributes')
        for category in element_dict["categories"]:
            attribute = SubElement(attributes, 'attribute')
            attribute.set("name", category)

    except KeyError:
        pass

    xrefs = SubElement(element, 'xrefs')

    return element


def _add_connector(
    connector_dict: dict[str, Any],
) -> Element:
    """
    Creates an XML representation for a UML connector.

    :param connector_dict: dictionary containing connector's details, relationships, and tags.
    :return: An XML element representing the UML connector.
    """
    connector = Element('connector')

    source = SubElement(connector, 'source')
    model = SubElement(source, 'model')
    model.set('name', connector_dict['source_name'])

    tags = SubElement(source, 'tags')
    for tag_dict in connector_dict['tags_source']:
        tag = SubElement(tags, 'tag')
        tag.set('name', tag_dict['name'])
        tag.set('value', tag_dict['value'])

    target = SubElement(connector, 'target')
    model = SubElement(target, 'model')
    model.set('name', connector_dict['target_name'])

    tags = SubElement(target, 'tags')
    for tag_dict in connector_dict['tags_target']:
        tag = SubElement(tags, 'tag')
        tag.set('name', tag_dict['name'])
        tag.set('value', tag_dict['value'])

    properties = SubElement(connector, 'properties')
    properties.set('ea_type', connector_dict['relationship'])

    labels = SubElement(connector, 'labels')
    for bound in ['lb', 'lt', 'rb', 'rt']:
        if connector_dict.get(bound) is not None:
            labels.set(bound, connector_dict[bound])

    tags = SubElement(connector, 'tags')
    for tag_dict in connector_dict['tags']:
        tag = SubElement(tags, 'tag')
        tag.set('name', tag_dict['name'])
        tag.set('value', tag_dict['value'])

    return connector


def _fill_in_xml(
    root: Element,
    json_data: dict[str, Any],
) -> Element:
    """
    Populates the XML root element with elements and connectors based on the provided JSON data.

    :param root: The root XML element.
    :param json_data: JSON data containing UML elements and connectors.
    :return: The updated root XML element.
    """
    # Process elements
    for element in json_data["elements"]:
        if len(root[1].findall('packagedElement')) == 0:
            root[1].append(_add_packaged_element(element))

        else:
            match_found = False
            for package in root[1].iter('packagedElement'):
                if package.get('xmi:id') == element['package']:
                    package.append(_add_packaged_element(element))
                    match_found = True
                    break

            if not match_found:
                root[1].append(_add_packaged_element(element))

    # Add detailed XML representation for specific element types
    for element in json_data["elements"]:
        if element["type"] == "uml:Package":
            root[2][0].append(_add_package(element))

        elif element["type"] == "uml:Class":
            root[2][0].append(_add_class(element))

        elif element["type"] == "uml:DataType":
            root[2][0].append(_add_datatype(element))

        elif element["type"] == "uml:Enumeration":
            root[2][0].append(_add_enumeration(element))

        else:
            print("Element not exported:", element["type"])

    # Process connectors
    for connector in json_data["connectors"]:
        root[2][1].append(_add_connector(connector))

    return root


def _get_root() -> Element:
    """
    Creates the base XML structure for the UML model, including namespaces and the basic structure.

    :return: The root XML element.
    """
    xml_root = """<?xml version="1.0" encoding="windows-1252"?>
                <xmi:XMI xmi:version="2.1" xmlns:uml="http://schema.omg.org/spec/UML/2.1" xmlns:xmi="http://schema.omg.org/spec/XMI/2.1" xmlns:thecustomprofile="http://www.sparxsystems.com/profiles/thecustomprofile/1.0">
                    <xmi:Documentation exporter="Enterprise Architect" exporterVersion="6.5"/>
                    <uml:Model xmi:type="uml:Model" name="EA_Model" visibility="public">
                    </uml:Model>
                    <xmi:Extension extender="Enterprise Architect" extenderID="6.5">
                        <elements>
                        </elements>
                        <connectors>
                        </connectors>
                    </xmi:Extension>
                </xmi:XMI>
    """

    root = fromstring(xml_root.encode('windows-1252'))

    # Register namespaces for the UML and XMI schema
    register_namespace('uml', 'http://schema.omg.org/spec/UML/2.1')  
    register_namespace('xmi', 'http://schema.omg.org/spec/XMI/2.1')  
    register_namespace('thecustomprofile', 'http://www.sparxsystems.com/profiles/thecustomprofile/1.0')

    return root


def json_to_xml(
    json_data: dict[str, Any],
) -> bytes:
    """
    Converts JSON data describing UML elements and connectors into an XML representation.

    :param json_data: JSON object containing UML elements and connectors.
    :return: XML data as bytes, encoded in 'windows-1252'.
    """
    root = _get_root()
    root = _fill_in_xml(root, json_data)

    # Serialize the XML tree into bytes
    xml_bytes: bytes = tostring(root, encoding="windows-1252")

    return xml_bytes
