import json
from typing import Any
from rdflib import Graph


def jsonld_to_ttl_bytes(jsonld_obj: Any) -> bytes:
    """
    Parse a JSON-LD object into an RDF graph and serialize it to Turtle bytes.
    """
    if jsonld_obj is None:
        return b""

    if isinstance(jsonld_obj, (bytes, bytearray)):
        jsonld_str = jsonld_obj.decode("utf-8", errors="replace")
    elif isinstance(jsonld_obj, str):
        jsonld_str = jsonld_obj
    else:
        jsonld_str = json.dumps(jsonld_obj, ensure_ascii=False)

    g = Graph()
    g.parse(data=jsonld_str, format="json-ld")

    turtle_data = g.serialize(format="turtle")
    if isinstance(turtle_data, str):
        turtle_data = turtle_data.encode("utf-8")

    return turtle_data or b""
