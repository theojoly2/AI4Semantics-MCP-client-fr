
import json
from typing import Any
from rdflib import Graph

def jsonld_to_ttl_bytes(jsonld_obj: Any) -> bytes:
    """
    Parse a JSON-LD object into an RDF graph and serialize to RDF/XML bytes.
    """
    # Ensure we pass a JSON string to rdflib.parse(data=..., format='json-ld')
    jsonld_str = json.dumps(jsonld_obj)

    g = Graph()
    # If your JSON-LD uses external @context URLs, rdflib can resolve them when network is allowed.
    # If your runtime is offline, ensure @context is inlined or provide a local mapping.
    g.parse(data=jsonld_str, format="json-ld")

    # 'xml' (aka application/rdf+xml) is standard RDF/XML output
    ttl: bytes = g.serialize(format="ttl")
    # rdflib returns str in some versions and bytes in others; normalize to bytes
    if isinstance(ttl, str):
        ttl = ttl.encode("utf-8")
    return ttl