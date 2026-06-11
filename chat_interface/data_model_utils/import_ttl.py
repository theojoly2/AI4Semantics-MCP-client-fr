from __future__ import annotations

from typing import Any
from io import BytesIO

import rdflib
import json
import uuid

"""
ttl_to_uml_json.py
Convert an OWL/RDF ontology in Turtle (TTL) to an Enterprise Architect-like UML JSON.

- Classes (owl:Class)  -> elements of type "uml:Class"
- ObjectProperties     -> connectors of type "Association" (domain -> range)
- DatatypeProperties   -> attributes on the domain class
- rdfs:subClassOf      -> connectors of type "Generalization" (child -> parent)
- owl:Ontology         -> root Package (name from rdfs:label @en if present)

IDs are deterministic per URI (uuid5), prefixed EA-like: EAPK_ for packages, EAID_ for classes.
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef, Literal

# Namespaces that often appear
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
UML_META = Namespace("urn:ai4semantics:uml:")


# ---------- Helpers ----------

def ea_id(prefix: str, uri: str) -> str:
    """
    EA-like deterministic ID from a URI: EAID/EAPK + UUID5 (underscored groups, uppercased).
    """
    u = uuid.uuid5(uuid.NAMESPACE_URL, str(uri))
    s = str(u).replace('-', '_').upper()
    return f"{prefix}_{s}"


def local_name(uri: URIRef) -> str:
    """
    Derive a human-friendly name from a URI: last segment after '/' or '#'.
    """
    s = str(uri)
    if '#' in s:
        s = s.split('#')[-1]
    else:
        s = s.rstrip('/').split('/')[-1]
    return s


def get_label(g: Graph, s: URIRef, lang: str = "en") -> str | None:
    vals = list(g.objects(s, RDFS.label))
    if not vals:
        return None
    # prefer requested language
    for v in vals:
        if isinstance(v, Literal) and v.language == lang:
            return str(v)
    # fallback to first literal/string
    return str(vals[0])


def get_comment(g: Graph, s: URIRef, lang: str = "en") -> str | None:
    vals = list(g.objects(s, RDFS.comment))
    if not vals:
        return None
    for v in vals:
        if isinstance(v, Literal) and v.language == lang:
            return str(v)
    return str(vals[0])


def get_literal_value(g: Graph, s: URIRef, p: URIRef) -> str | None:
    """Safely extract a raw string from a given triple property."""
    vals = list(g.objects(s, p))
    if not vals:
        return None
    return str(vals[0])


def primitive_from_range(r: URIRef) -> str | None:
    """
    Map RDF/XSD types to UML primitive names; None means treat as classifier name.
    Extend as needed.
    """
    s = str(r)
    if s in (str(RDFS.Literal), str(RDF.langString)):
        return "String"
    # Common XSD mappings
    xsd_map = {
        str(XSD.string): "String",
        str(XSD.boolean): "Boolean",
        str(XSD.integer): "Integer",
        str(XSD.int): "Integer",
        str(XSD.long): "Integer",
        str(XSD.float): "Real",
        str(XSD.double): "Real",
        str(XSD.decimal): "Real",
        str(XSD.date): "Date",
        str(XSD.dateTime): "DateTime",
        str(XSD.time): "Time",
        str(XSD.gYear): "String",       # could be a specialized type
        str(XSD.gYearMonth): "String",
        str(XSD.anyURI): "URI",
    }
    return xsd_map.get(s)


def role_name_from_label(label: str | None) -> str | None:
    """
    EA diagrams often show role names like '+isAbout'. We'll prefix with '+' if label exists.
    """
    if not label:
        return None
    return f"+{label}"


# ---------- Extraction ----------

def extract_ontology_package(g: Graph, custom_name: str | None = None) -> dict:
    # Identify owl:Ontology subject(s)
    ontos = list(g.subjects(RDF.type, OWL.Ontology))
    if ontos:
        onto = ontos[0]
        name = custom_name or get_label(g, onto, "en") or local_name(onto)
        pkg_uri = str(onto)
    else:
        # Fallback package when no owl:Ontology triple is present
        name = custom_name or "Generated"
        pkg_uri = f"urn:pkg:{name}"
    pkg_id = ea_id("EAPK", pkg_uri)
    return {
        "name": name,
        "ID": pkg_id,
        "uri": pkg_uri,
        "type": "uml:Package",
        "visibility": "public",
        "package": "",
        "tags": []
    }


def build_model(g: Graph, package_name: str | None = None) -> dict:
    model = {"elements": [], "connectors": []}

    # Root package
    root_pkg = extract_ontology_package(g, custom_name=package_name)
    model["elements"].append(root_pkg)
    root_pkg_id = root_pkg["ID"]

    # Index classes for quick lookup
    class_elems: dict[str, dict] = {}

    # 1) Classes
    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        name = get_label(g, cls, "en") or local_name(cls)
        cls_id = ea_id("EAID", uri)
        tags = [{"name": "uri", "value": uri}]
        comment = get_comment(g, cls, "en")
        if comment:
            tags.append({"name": "definition-en", "value": comment})
        label = get_label(g, cls, "en")
        if label:
            tags.append({"name": "label-en", "value": label})
        
        # Optional: usage scope tag aligned to many SEMIC models
        tags.append({"name": "class-usage-scope", "value": "main"})

        elem = {
            "name": name,
            "ID": cls_id,
            "uri": uri,
            "type": "uml:Class",
            "visibility": "public",
            "package": root_pkg_id,
            "tags": tags,
            "attributes": []
        }
        class_elems[uri] = elem
        model["elements"].append(elem)

    # Helper to ensure a class element exists for non-primitive ranges
    def ensure_class(uri_ref: URIRef):
        u = str(uri_ref)
        if u not in class_elems:
            name = get_label(g, uri_ref, "en") or local_name(uri_ref)
            cls_id = ea_id("EAID", u)
            elem = {
                "name": name,
                "ID": cls_id,
                "uri": u,
                "type": "uml:Class",
                "visibility": "public",
                "package": root_pkg_id,
                "tags": [{"name": "uri", "value": u}],
                "attributes": []
            }
            class_elems[u] = elem
            model["elements"].append(elem)
        return class_elems[u]

    # 2) Datatype properties -> attributes
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        prop_uri = str(prop)
        prop_label = get_label(g, prop, "en")
        prop_comment = get_comment(g, prop, "en")
        
        # Domains
        for domain in g.objects(prop, RDFS.domain):
            domain_uri = str(domain)
            if domain_uri not in class_elems:
                ensure_class(domain)
            domain_elem = class_elems[domain_uri]
            
            # Range
            ranges = list(g.objects(prop, RDFS.range)) or [RDFS.Literal]
            for rng in ranges:
                primitive = primitive_from_range(rng)
                if primitive:
                    attr_type = primitive
                else:
                    rng_elem = ensure_class(rng)
                    attr_type = rng_elem["name"]

                attr_name = prop_label or local_name(prop)
                attr_tags = [{"name": "uri", "value": prop_uri}]
                if prop_label:
                    attr_tags.append({"name": "label-en", "value": prop_label})
                if prop_comment:
                    attr_tags.append({"name": "definition-en", "value": prop_comment})

                domain_elem["attributes"].append({
                    "name": attr_name,
                    "visibility": "public",
                    "type": attr_type,
                    "uri": prop_uri,
                    "lower_bounds": get_literal_value(g, prop, UML_META.lowerBound) or "",
                    "upper_bounds": get_literal_value(g, prop, UML_META.upperBound) or "",
                    "tags_attribute": attr_tags
                })

    # 3) Object properties -> associations
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_uri = str(prop)
        prop_label = get_label(g, prop, "en")
        prop_comment = get_comment(g, prop, "en")

        domains = list(g.objects(prop, RDFS.domain))
        ranges = list(g.objects(prop, RDFS.range))
        
        # If any side is missing, we cannot create a robust connector
        if not domains or not ranges:
            continue

        relationship = get_literal_value(g, prop, UML_META.relationshipType) or "Association"

        for domain in domains:
            domain_elem = ensure_class(domain)
            for rng in ranges:
                range_elem = ensure_class(rng)
                
                connector_id = ea_id("CONN", f"{str(domain)}:{str(rng)}:{prop_uri}")
                
                tgt_tags = [{"name": "uri", "value": prop_uri}, {"name": "connector_id", "value": connector_id}]
                if prop_comment:
                    tgt_tags.append({"name": "definition-en", "value": prop_comment})
                if prop_label:
                    tgt_tags.append({"name": "label-en", "value": prop_label})

                connector = {
                    "connector_id": connector_id,
                    "source_id": domain_elem["ID"],
                    "target_id": range_elem["ID"],
                    "source_name": domain_elem["name"],
                    "target_name": range_elem["name"],
                    "relationship": relationship,
                    "name": prop_label or local_name(prop),
                    "uri": prop_uri,
                    "lb": get_literal_value(g, prop, UML_META.leftMultiplicity) or "",
                    "lt": get_literal_value(g, prop, UML_META.leftRole) or "",
                    "rb": get_literal_value(g, prop, UML_META.rightMultiplicity) or "",
                    "rt": get_literal_value(g, prop, UML_META.rightRole) or role_name_from_label(prop_label) or "",
                    "tags": [{"name": "connector_id", "value": connector_id}],
                    "tags_source": [],
                    "tags_target": tgt_tags
                }
                model["connectors"].append(connector)

    # 4) rdfs:subClassOf -> generalizations
    for child, parent in g.subject_objects(RDFS.subClassOf):
        if isinstance(parent, URIRef):
            child_elem = ensure_class(child)
            parent_elem = ensure_class(parent)
            
            connector_id = ea_id("CONN", f"{str(child)}:{str(parent)}:subClassOf")
            
            gen = {
                "connector_id": connector_id,
                "source_id": child_elem["ID"],
                "target_id": parent_elem["ID"],
                "source_name": child_elem["name"],
                "target_name": parent_elem["name"],
                "relationship": "Generalization",
                "name": "subClassOf",
                "uri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                "lb": "", 
                "lt": "", 
                "rb": "", 
                "rt": "",
                "tags": [{"name": "connector_id", "value": connector_id}], 
                "tags_source": [], 
                "tags_target": [{"name": "connector_id", "value": connector_id}]
            }
            model["connectors"].append(gen)

    return model


def ttl_to_json(uploaded_file: BytesIO) -> dict[str, Any]:
    """
    Imports a TTL file, parses it as an RDF graph, and serializes to JSON-LD and TTL.

    :param uploaded_file: The uploaded TTL file (BytesIO).
    :return: A dict with "ttl" (JSON-LD as Python objects), "ttl_raw" (TTL string), and "xmi" (Unified elements structure).
    """
    try:
        # Ensure pointer at start
        uploaded_file.seek(0)
        data_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        g = rdflib.Graph()
        # Use parse(data=...) to avoid rdflib treating input as a filename
        g.parse(data=data_bytes.decode("utf-8", errors="replace"), format="ttl")

        # Serialize the graph to JSON-LD (string), then to Python dict
        json_ld_data: str = g.serialize(format="json-ld", indent=4)
        json_data = json.loads(json_ld_data) if json_ld_data else {}

        ttl_raw_bytes = g.serialize(format="ttl")
        ttl_raw = ttl_raw_bytes.decode("utf-8") if isinstance(ttl_raw_bytes, (bytes, bytearray)) else str(ttl_raw_bytes)

        xmi = build_model(g)

        return {
            "ttl": json_data,
            "ttl_raw": ttl_raw,
            "xmi": xmi,
        }
    except Exception as e:
        # Let caller show a friendly message
        raise
