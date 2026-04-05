"""
world_builder.py — Generate synthetic hierarchical world graphs for HG-RAG experiments.
Hierarchy: Planets -> Countries -> Cities

My worlds take inspiration from those popular grand strategy map games like Stellaris, mirroring events 
you would take note of and care about as you're playing these games as a nation's (or planet's) leader
"""

import networkx as nx
import random

# Name Pools, Claude helped me with creating a bunch of dummy names and events 
PREFIXES = [
    "Ash", "Bell", "Crest", "Dark", "East", "Fern", "Glen", "High", "Iron", "Jade",
    "Keld", "Lake", "Moor", "Nord", "Oak", "Pine", "Rock", "Storm", "Tor", "Vale",
    "West", "Xen", "York", "Zor", "Amber", "Blue", "Cold", "Dusk", "Edge", "Fire",
    "Grim", "Holt", "Isle", "Jorn", "Kray", "Loch", "Marsh", "Neth", "Onyx", "Port",
    "Quen", "Rune", "Sand", "Thorn", "Umbre", "Veld", "Wren", "Xar", "Yore", "Zeal",
]
SUFFIXES = [
    "burg", "ford", "haven", "port", "gate", "moor", "vale", "peak", "bridge", "field",
    "wood", "wick", "ton", "holm", "mere", "fell", "cliff", "dale", "cross", "bay",
    "keep", "hold", "mark", "watch", "spire", "stead", "grove", "ridge", "cove", "strand",
]
PLANET_NAMES = ["Vortan", "Celenis", "Dravos", "Umbriel", "Kastra"]
COUNTRY_NAMES = [
    "Barnes", "Korith", "Nalden", "Strevon", "Archus", "Venthai", "Ormis", "Pelcor", "Sundar", "Trevak",
    "Braxis", "Halden", "Feroth", "Qundis", "Morthal", "Lyssen", "Carvon", "Eldris", "Wamban", "Nothis",
    "Zipher", "Acroth", "Bennis", "Delvon", "Espar", "Fhoran", "Gremen", "Halvor", "Ixthal", "Jorvek",
    "Kalden", "Lomber", "Mivax", "Norder", "Ospen", "Praxis", "Qlaven", "Rethon", "Sunder", "Theron",
]
EXPORTS_POOL = [
    "grain", "minerals", "textiles", "technology", "fish", "timber", "weapons", "spices",
    "medicine", "fuel", "gems", "livestock", "machinery", "pottery", "silk", "coal",
    "electronics", "chemicals", "pharmaceuticals", "oil",
]
STABILITY_OPTIONS = ["High", "Medium", "Low", "Unstable"]
CLIMATE_OPTIONS = ["Arid", "Temperate", "Tropical", "Arctic", "Desert", "Oceanic"]


def _make_city_name(used_names, rng):
    for _ in range(200):
        name = rng.choice(PREFIXES) + rng.choice(SUFFIXES)
        if name not in used_names:
            used_names.add(name)
            return name
    name = f"City{len(used_names)}"
    used_names.add(name)
    return name


def generate_world(size="small", seed=42):
    """
    Generate a hierarchical world graph using NetworkX.

    Sizes:
      small  — 2 planets, 3 countries each, 3 cities each  (~18 cities)
      medium — 3 planets, 5 countries each, 10 cities each (~150 cities)
      large  — 4 planets, 8 countries each, 25 cities each (~800 cities)

    Returns a NetworkX DiGraph.
    """
    rng = random.Random(seed)
    G = nx.DiGraph()
    used_names = set()

    params = {
        "small":  (2, 3, 3), # x planets, y countries each, z cities each
        "medium": (3, 5, 10),
        "large":  (4, 8, 25),
    }
    n_planets, n_countries_per_planet, n_cities_per_country = params[size]

    planet_names = PLANET_NAMES[:n_planets]
    country_names = COUNTRY_NAMES[: n_planets * n_countries_per_planet]
    all_cities_global = []

    for pi, planet in enumerate(planet_names):
        G.add_node(planet,
                   type="planet",
                   population=f"{rng.randint(1, 9)}B",
                   climate=rng.choice(CLIMATE_OPTIONS),
                   exports=rng.sample(EXPORTS_POOL, 2))

        countries = country_names[pi * n_countries_per_planet: (pi + 1) * n_countries_per_planet]
        planet_cities = []

        for country in countries:
            G.add_node(country,
                       type="country",
                       population=f"{rng.randint(10, 500)}M",
                       army_size=f"{rng.randint(5, 500)}k",
                       stability=rng.choice(STABILITY_OPTIONS),
                       exports=rng.sample(EXPORTS_POOL, 2))
            G.add_edge(planet, country, relation="contains")

            cities = []
            for _ in range(n_cities_per_country):
                city = _make_city_name(used_names, rng)
                G.add_node(city,
                           type="city",
                           population=f"{rng.randint(10, 999)}k",
                           exports=rng.sample(EXPORTS_POOL, 2),
                           stability=rng.choice(STABILITY_OPTIONS))
                G.add_edge(country, city, relation="contains")
                cities.append(city)

            # Border relations between adjacent cities in same country
            for i in range(len(cities)):
                for j in range(i + 1, min(i + 3, len(cities))):
                    G.add_edge(cities[i], cities[j], relation="borders")
                    G.add_edge(cities[j], cities[i], relation="borders")

            planet_cities.extend(cities)
            all_cities_global.extend(cities)

        # Trade relations between cities on same planet
        n_trade = min(len(planet_cities) * 2, 50)
        for _ in range(n_trade):
            if len(planet_cities) < 2:
                break
            c1, c2 = rng.sample(planet_cities, 2)
            if not G.has_edge(c1, c2):
                G.add_edge(c1, c2, relation="trade_with")
                G.add_edge(c2, c1, relation="trade_with")

        # Country-level relationships within this planet
        # Some country pairs trade (automatically friendly); the rest get a random quality
        country_trading_pairs: set = set()
        n_country_trades = max(1, n_countries_per_planet // 2)
        for _ in range(n_country_trades * 3):
            if len(countries) < 2:
                break
            c1, c2 = rng.sample(countries, 2)
            if (c1, c2) not in country_trading_pairs:
                country_trading_pairs.add((c1, c2))
                country_trading_pairs.add((c2, c1))
                G.add_edge(c1, c2, relation="trade_with")
                G.add_edge(c2, c1, relation="trade_with")

        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                c1, c2 = countries[i], countries[j]
                if (c1, c2) not in country_trading_pairs:
                    rel = rng.choice(["neutral", "unfriendly", "hostile"])
                    G.add_edge(c1, c2, relation=rel)
                    G.add_edge(c2, c1, relation=rel)

    # Post-processing: assign city imports from trade partner exports.
    # For each trade pair (A, B): B gets at least one of A's exports as an import, and vice versa.
    # This guarantees export-disruption and import-halt queries are always semantically coherent.
    for city in all_cities_global:
        G.nodes[city]["imports"] = []

    seen_trade_pairs = set()
    for city in all_cities_global:
        for _, neighbor, d in G.out_edges(city, data=True):
            if d.get("relation") != "trade_with":
                continue
            if G.nodes[neighbor].get("type") != "city":
                continue
            pair = tuple(sorted([city, neighbor]))
            if pair in seen_trade_pairs:
                continue
            seen_trade_pairs.add(pair)

            city_exports     = G.nodes[city].get("exports", [])
            neighbor_exports = G.nodes[neighbor].get("exports", [])

            if city_exports:
                to_add = rng.choice(city_exports)
                n_imp  = G.nodes[neighbor]["imports"]
                if to_add not in n_imp and len(n_imp) < 2:
                    n_imp.append(to_add)

            if neighbor_exports:
                to_add = rng.choice(neighbor_exports)
                c_imp  = G.nodes[city]["imports"]
                if to_add not in c_imp and len(c_imp) < 2:
                    c_imp.append(to_add)

    return G


def world_stats(G):
    # Return a dict with node counts by type 
    counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts
