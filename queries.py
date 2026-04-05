"""
queries.py is the file which generates 50 fixed queries from the world graph with ground-truth answers.

Query types (per research plan):
  1. local_fact         — city/country attributes
  2. hierarchical       — "what country/planet is X in?"
  3. neighborhood       — border / trade neighbors
  4. multi_hop          — surface level impact analysis across trade relations
"""

import random

# traversal helpers
def _get_parent(G, node, parent_type):
    for u, _, d in G.in_edges(node, data=True):
        if d.get("relation") == "contains" and G.nodes[u].get("type") == parent_type:
            return u
    return None
def _get_neighbors_by_rel(G, node, rel):
    return [v for _, v, d in G.out_edges(node, data=True) if d.get("relation") == rel]

def generate_queries(G, seed=42, max_queries=50):
    """
    Build a deterministic set of queries derivable from the graph.

    Each query dict has:
      query     — natural language question
      expected  — ground truth answer string
      anchor    — ground-truth node name; used to validate the query is answerable
                  and as a fallback if NLP entity extraction fails
      type      — query category
    """
    rng = random.Random(seed)
    queries = []

    cities   = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "city"]
    countries = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "country"]

    # 1. Local facts: city attributes 
    for city, attrs in rng.sample(cities, min(10, len(cities))):
        # Population
        queries.append({
            "query": f"What is the population of {city}?",
            "expected": str(attrs.get("population", "")),
            "anchor": city,
            "type": "local_fact",
        })
        # Exports
        if attrs.get("exports"):
            queries.append({
                "query": f"What does {city} export?",
                "expected": ", ".join(attrs["exports"]),
                "anchor": city,
                "type": "local_fact",
            })
        # Stability
        queries.append({
            "query": f"What is the stability level of {city}?",
            "expected": str(attrs.get("stability", "")),
            "anchor": city,
            "type": "local_fact",
        })

    # 2. Hierarchical: parent lookups
    for city, _ in rng.sample(cities, min(10, len(cities))):
        country = _get_parent(G, city, "country")
        planet = _get_parent(G, country, "planet") if country else None

        if country:
            queries.append({
                "query": f"What country is {city} located in?",
                "expected": country,
                "anchor": city,
                "type": "hierarchical",
            })
        if planet:
            queries.append({
                "query": f"What planet is {city} on?",
                "expected": planet,
                "anchor": city,
                "type": "hierarchical",
            })
        if country and planet:
            queries.append({
                "query": f"Which planet does the country {country} belong to?",
                "expected": planet,
                "anchor": country,
                "type": "hierarchical",
            })

    # 3. Neighborhood reasoning: borders, trade, etc.
    for city, _ in rng.sample(cities, min(10, len(cities))):
        borders = _get_neighbors_by_rel(G, city, "borders")
        trades = _get_neighbors_by_rel(G, city, "trade_with")

        if borders:
            queries.append({
                "query": f"Which cities share a border with {city}?",
                "expected": ", ".join(borders),
                "anchor": city,
                "type": "neighborhood",
            })
        if trades:
            queries.append({
                "query": f"Which cities does {city} have trade relations with?",
                "expected": ", ".join(trades),
                "anchor": city,
                "type": "neighborhood",
            })

    """
    Pranu's note: Making sure the multi-hop questions fit the smaller world size was key 
    by generalizing these questions. I initially wanted to have elaborate questions 
    referincing specific events like a mine collapse or drought affecting agriculture,
    but smaller world sizes would have a large chance of these questions applying to
    cities without ore or agricultural exports due to small world sizes having less variety, 
    contradicting the premise and confusing LLMs. As you can see, these questions are only 
    administered after checking if we have proper entities to attribute these questions to.
    """

    # 4. Multi-hop: causal impact analysis across trade relations
    # 4a. Export disruption: labor strike halting a city's actual exports.
    # Only generated when at least one trade partner actually imports what this city exports.
    for city, attrs in rng.sample(cities, min(6, len(cities))):
        city_exports = attrs.get("exports", [])
        if not city_exports:
            continue
        trade_partners = _get_neighbors_by_rel(G, city, "trade_with")
        affected = [
            p for p in trade_partners
            if any(exp in G.nodes[p].get("imports", []) for exp in city_exports)
        ]
        if affected:
            exports_str = " and ".join(city_exports)
            affected_details = [
                f"{p} (imports {', '.join(e for e in city_exports if e in G.nodes[p].get('imports', []))} from {city})"
                for p in affected
            ]
            answer_key = (
                f"The cities affected are {', '.join(affected)}. "
                f"They rely on {city} for imported goods: {'; '.join(affected_details)}. "
                f"The labor strike halting exports of {exports_str} directly disrupts their supply."
            )
            queries.append({
                "query": (
                    f"{city} is experiencing a labor strike that halts the export of {exports_str}. "
                    f"Which cities are affected by this and why?"
                ),
                "expected": " ".join(affected),
                "answer_key": answer_key,
                "anchor": city,
                "type": "multi_hop",
            })

    # 4b. Import halt: city stops importing due to surplus; trade partners who do trade take a hit.
    # Only generated when at least one trade partner exports what this city imports.
    for city, attrs in rng.sample(cities, min(6, len(cities))):
        city_imports = attrs.get("imports", [])
        if not city_imports:
            continue
        trade_partners = _get_neighbors_by_rel(G, city, "trade_with")
        affected = [
            p for p in trade_partners
            if any(imp in G.nodes[p].get("exports", []) for imp in city_imports)
        ]
        if affected:
            affected_details = [
                f"{p} (exports {', '.join(i for i in city_imports if i in G.nodes[p].get('exports', []))} to {city})"
                for p in affected
            ]
            answer_key = (
                f"The cities that would take an economic hit are {', '.join(affected)}. "
                f"They supply goods to {city}: {'; '.join(affected_details)}. "
                f"With {city} halting all imports, these trade partners lose a key market."
            )
            queries.append({
                "query": (
                    f"{city} has a surplus of goods and temporarily halts the imports of any goods "
                    f"to save money. Which cities would take an economic hit from this?"
                ),
                "expected": " ".join(affected),
                "answer_key": answer_key,
                "anchor": city,
                "type": "multi_hop",
            })

    # 4c. War declaration: country hostility.
    # Only generated for countries with at least one hostile/unfriendly neighbor.
    # If no such country exists, the multi_hop 25% slot is filled entirely by 4a/4b queries.
    for country, _ in rng.sample(countries, min(6, len(countries))):
        hostile    = _get_neighbors_by_rel(G, country, "hostile")
        unfriendly = _get_neighbors_by_rel(G, country, "unfriendly")
        targets    = hostile if hostile else unfriendly
        if targets:
            relation_type = "hostile" if hostile else "unfriendly"
            answer_key = (
                f"The most likely target would be {', '.join(targets)}. "
                f"{country} has a {relation_type} relationship with "
                f"{'them' if len(targets) > 1 else targets[0]}, "
                f"making military escalation the expected outcome."
            )
            queries.append({
                "query": (
                    f"{country} has declared war on a neighboring country. "
                    f"Which country may that be and why?"
                ),
                "expected": " ".join(targets),
                "answer_key": answer_key,
                "anchor": country,
                "type": "multi_hop",
            })

    # Remove duplicates by query text
    seen = set()
    unique = []
    for q in queries:
        if q["query"] not in seen:
            seen.add(q["query"])
            unique.append(q)

    # Enforce 25% share per type: cap each type at max_queries // 4
    per_type_cap = max(1, max_queries // 4)
    by_type_grouped: dict = {}
    for q in unique:
        by_type_grouped.setdefault(q["type"], []).append(q)

    balanced = []
    for type_qs in by_type_grouped.values():
        balanced.extend(type_qs[:per_type_cap])

    rng.shuffle(balanced)
    return balanced[:max_queries]
