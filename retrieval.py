"""
retrieval.py is our primary HG-RAG retrieval pipeline

Steps:
  1. Entity resolution      — find anchor node by name
  2. Hierarchy anchoring    — walk up to parents (k_up levels)
  3. Graph traversal        — collect neighbors (k_side hops)
  4. Context serialization  — structured text output
"""


def resolve_entity(G, name):
    # Case-insensitive name match to find anchor node. Returns node ID or None.
    name_lower = name.strip().lower()
    for node in G.nodes():
        if str(node).lower() == name_lower:
            return node
    return None


def _get_parent(G, node):
    # Return the single 'contains' parent of node, or None
    for u, _, d in G.in_edges(node, data=True):
        if d.get("relation") == "contains":
            return u
    return None


def get_ancestors(G, node, k_up):
    # Walk up the hierarchy k_up levels. Returns ordered list [parent, grandparent, ...] 
    ancestors = []
    current = node
    for _ in range(k_up):
        parent = _get_parent(G, current)
        if parent is None:
            break
        ancestors.append(parent)
        current = parent
    return ancestors


"""
Pranu's note: Subgraphs are capped at 15 neighbor nodes to keep context from growing too large.
The problem this aims to solve is problems with relational questions being answered poorly,
if a country has too many neighbors, it can be difficult for the LLM to answer relational questions
because the correct answer may be buried in the middle of the list of neighbors. So we have this in
place and prioritizing adversarial neighbors so they are never cut by the cap. One way to elevate
our RAG model from just being a dumb fetcher. 
"""

def get_lateral_neighbors(G, node, k_side):
    # Return nodes reachable via non-hierarchy edges within k_side hops
    # Traverses all non-contains edges (trade_with, borders, hostile, unfriendly, neutral, etc.)
    visited = {node}
    frontier = {node}
    for _ in range(k_side):
        next_frontier = set()
        for n in frontier:
            for _, v, d in G.out_edges(n, data=True):
                if d.get("relation") != "contains" and v not in visited:
                    next_frontier.add(v)
            for u, _, d in G.in_edges(n, data=True):
                if d.get("relation") != "contains" and u not in visited:
                    next_frontier.add(u)
        visited |= next_frontier
        frontier = next_frontier
    visited.discard(node)
    return list(visited)


def get_children(G, node, k_down):
    # Return direct children (and their children up to k_down levels)
    if k_down <= 0:
        return []
    children = [v for _, v, d in G.out_edges(node, data=True) if d.get("relation") == "contains"]
    result = list(children)
    for child in children:
        result.extend(get_children(G, child, k_down - 1))
    return result


def retrieve_subgraph(G, anchor_node, k_up=2, k_side=1, k_down=0):
    """
    Retrieve a relevant subgraph around anchor_node.

    Parameters: 
    k_up   : hierarchy levels upward (e.g. 2 = city -> country -> planet)
    k_side : lateral neighbor hops   (e.g. 1 = immediate trade/border neighbors)
    k_down : children levels         (e.g. 0 = no children)

    Returns dict with 'nodes' and 'relations'.
    """
    ancestors = get_ancestors(G, anchor_node, k_up)
    include = {anchor_node}
    include.update(ancestors)
    neighbors = get_lateral_neighbors(G, anchor_node, k_side)

    # Prioritize adversarial neighbors so they are never cut by the cap.
    # Collect all nodes connected to the anchor via hostile/unfriendly edges first,
    # then fill remaining slots with other neighbors (trade, border, etc.).
    _ADVERSARIAL = {"hostile", "unfriendly"}
    adversarial = {
        v for _, v, d in G.out_edges(anchor_node, data=True)
        if d.get("relation") in _ADVERSARIAL
    } | {
        u for u, _, d in G.in_edges(anchor_node, data=True)
        if d.get("relation") in _ADVERSARIAL
    }
    other_neighbors = [n for n in neighbors if n not in adversarial]
    remaining_slots = max(0, 15 - len(adversarial))
    include.update(adversarial)
    include.update(other_neighbors[:remaining_slots])

    include.update(get_children(G, anchor_node, k_down))

    subgraph_nodes = {n: dict(G.nodes[n]) for n in include if n in G}

    relations = []
    seen_pairs = set()
    for u, v, d in G.edges(data=True):
        if u in include and v in include:
            rel = d.get("relation", "related")
            key = (min(u, v), max(u, v), rel)
            if key not in seen_pairs:
                seen_pairs.add(key)
                relations.append((u, v, rel))

    return {
        "nodes": subgraph_nodes,
        "relations": relations,
        "anchor": anchor_node,
        "hierarchy_chain": [anchor_node] + ancestors,
    }


def extract_anchor(G, query, model):
    """
    Extract the anchor entity from a natural language query using the LLM
    then resolve it against the graph with a fuzzy (relative proximity) fallback
    Returns a node ID or None
    """
    import difflib
    from llm import build_entity_extraction_prompt, query_llm

    prompt, system = build_entity_extraction_prompt(query)
    raw = query_llm(prompt, model=model, system=system, max_tokens=15).strip()

    # Normalise: strip punctuation, take only the first line/word-group
    raw = raw.splitlines()[0].strip(".,!?;:\"'()[]").strip()

    node_names = [str(n) for n in G.nodes()]
    names_lower = [n.lower() for n in node_names]

    # Try progressively looser matches
    candidates = [raw]
    if " " in raw:
        candidates.append(raw.split()[0])   # first word only

    for candidate in candidates:
        anchor = resolve_entity(G, candidate)
        if anchor is not None:
            return anchor
        matches = difflib.get_close_matches(
            candidate.lower(), names_lower, n=1, cutoff=0.5
        )
        if matches:
            return resolve_entity(G, matches[0])

    return None


def serialize_context(subgraph_data):
    """
    Serialize a retrieved subgraph into structured text (not prose).

    Format:
      [CITY]
      Name: Pranaville
      Population: 420k
      ...
      [RELATIONS]
      Pranaville <-> Pranavaria (trade_with)
    """
    nodes = subgraph_data["nodes"]
    relations = subgraph_data["relations"]
    chain = subgraph_data.get("hierarchy_chain", [])

    by_type = {}
    for name, attrs in nodes.items():
        t = attrs.get("type", "unknown")
        by_type.setdefault(t, []).append((name, attrs))

    lines = []

    # Explicit location chain so the LLM knows containment without inference
    # Removed arrow notation due to misreadings by the LLM
    # Opted for plain English sentences that directly answer "what X is Y in?" instead
    if chain:
        chain_labels = [
            f"{n} ({nodes[n].get('type', '?')})" for n in chain if n in nodes
        ]
        if chain_labels:
            lines.append("[LOCATION CHAIN]")
            lines.append(" → ".join(chain_labels))
            # Prose containment statements: "CityX is a city in CountryY. CountryY is a country on PlanetZ."
            for i in range(len(chain) - 1):
                child  = chain[i]
                parent = chain[i + 1]
                if child not in nodes or parent not in nodes:
                    continue
                child_type  = nodes[child].get("type", "entity")
                parent_type = nodes[parent].get("type", "entity")
                lines.append(f"{child} is a {child_type} located in {parent} (a {parent_type}).")
            lines.append("")

    for type_label in ["planet", "country", "city"]:
        for name, attrs in by_type.get(type_label, []):
            lines.append(f"[{type_label.upper()}]")
            lines.append(f"Name: {name}")
            for k, v in attrs.items():
                if k == "type":
                    continue
                if isinstance(v, list):
                    v = ", ".join(v) if v else "none"
                lines.append(f"{k.capitalize()}: {v}")
            lines.append("")

    non_hierarchy = [(u, v, r) for u, v, r in relations if r != "contains"]
    if non_hierarchy:
        lines.append("[RELATIONS]")
        for u, v, rel in non_hierarchy:
            lines.append(f"{u} <-> {v} ({rel})")
        lines.append("")

    return "\n".join(lines)


# Flat-world serialization for the baseline 
def serialize_flat_world(G, max_chars=None):
    """
    Serialize the entire world as a single flat text block.
    """
    lines = ["=== WORLD KNOWLEDGE ===", ""]

    for node, attrs in G.nodes(data=True):
        t = attrs.get("type", "unknown")
        lines.append(f"[{t.upper()}] {node}")
        for k, v in attrs.items():
            if k == "type":
                continue
            if isinstance(v, list):
                v = ", ".join(v) if v else "none"
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("[ALL RELATIONS]")
    seen_rel = set()
    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "")
        key = (min(str(u), str(v)), max(str(u), str(v)), rel)
        if key not in seen_rel:
            seen_rel.add(key)
            lines.append(f"  {u} <-> {v} ({rel})")

    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


# Simple vector RAG baseline

def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def build_rag_index(G, embed_model="nomic-embed-text"):
    """
    Build a dense vector index for simple RAG: one chunk per graph node,
    containing the node's attributes and its direct relations.
    Requires an Ollama embedding model (default: nomic-embed-text).
    Run 'ollama pull nomic-embed-text' once before use
    Returns a list of (chunk_text, embedding) tuples.
    """
    from llm import get_embedding

    index = []
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "entity")
        lines = [f"[{node_type.upper()}] {node}"]
        for k, v in attrs.items():
            if k == "type":
                continue
            if isinstance(v, list):
                v = ", ".join(v) if v else "none"
            lines.append(f"  {k}: {v}")
        # Containment parent 
        for u, _, d in G.in_edges(node, data=True):
            if d.get("relation") == "contains":
                lines.append(f"  located in: {u}")
        # Lateral relations
        rels = [
            f"{d.get('relation')} {v}"
            for _, v, d in G.out_edges(node, data=True)
            if d.get("relation") != "contains"
        ]
        if rels:
            lines.append(f"  relations: {'; '.join(rels)}")
        chunk = "\n".join(lines)
        index.append((chunk, get_embedding(chunk, model=embed_model)))
    return index


def retrieve_rag_chunks(query: str, index: list, embed_model: str = "nomic-embed-text", top_k: int = 10) -> str:
    """
    Embed the query and return the top-k most similar chunks as a single context string.
    top_k=10 is chosen to roughly match the HG-RAG subgraph size for a fair comparison.
    """
    from llm import get_embedding

    query_emb = get_embedding(query, model=embed_model)
    scored = sorted(
        [(_cosine_similarity(query_emb, emb), chunk) for chunk, emb in index if emb],
        reverse=True,
    )
    return "\n\n".join(chunk for _, chunk in scored[:top_k])
