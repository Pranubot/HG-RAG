"""
evaluate.py — scoring outputs based on rubric

Metrics:
  All query types:
    factual_accuracy    0-2  keyword match against ground-truth expected answer
                             2 = ≥70% keywords matched, 1 = ≥30%, 0 = below 30%
    hallucination_rate  0-1  fraction of capitalised entity-like tokens in the answer
                             that are not valid graph nodes (lower is better)
    locality_awareness  0-1  fraction of graph nodes mentioned in the answer that
                             belong to the same planet as the anchor (1.0 = perfect)

  Multi-hop only (in addition to the above):
    llm_judge_score     1-5  holistic rating from an LLM judge comparing the response
                             against a deterministic answer key built from graph data
                             5 = correct entities + reasoning, 1 = completely wrong
    llm_judge_reason        one-sentence explanation from the judge
"""

import re
import pandas as pd

from retrieval import (
    resolve_entity,
    retrieve_subgraph,
    serialize_context,
    extract_anchor,
    build_rag_index,
    retrieve_rag_chunks,
)
from llm import query_llm, build_hgrag_prompt, build_rag_prompt, build_judge_prompt, EMBED_MODEL

# Max chars given to the baseline (removed)
MAX_FLAT_CHARS = None

# Common words to ignore when checking for hallucinated entity names (generated with claude)
_STOP = {
    "the", "a", "an", "in", "on", "at", "to", "of", "and", "or", "is", "are", "was",
    "were", "has", "have", "this", "that", "with", "for", "from", "it", "its", "be",
    "by", "not", "no", "yes", "city", "country", "planet", "region", "area", "world",
    "located", "found", "known", "called", "named", "based", "according", "following",
    "answer", "question", "information", "context", "provided", "given", "currently",
    "north", "south", "east", "west", "high", "low", "medium", "large", "small",
    "export", "exports", "import", "imports", "trade", "border", "borders",
    "stability", "population", "events", "climate", "conflict", "conflicts",
    "there", "here", "some", "any", "all", "each", "which", "what", "where", "how",
    "does", "do", "did", "can", "could", "would", "should", "will", "may", "might",
}


# scoring helpers
def score_factual_accuracy(answer: str, expected: str) -> int:
    """
    Keyword overlap between the answer and the ground-truth expected string.
    Special-cases boolean/negative expected answers.
    2 = ≥70% of expected keywords found in answer (fully correct)
    1 = ≥30% found (partially correct)
    0 = <30% found (wrong / irrelevant)
    """
    answer_l = answer.lower()
    expected_l = expected.lower().strip()

    # Boolean / negative queries
    if expected_l in ("no", "false", "none", "not found"):
        neg_words = ("no ", "not ", "does not", "don't", "doesn't", "never", "none",
                     "false", "no evidence", "not in context", "not mentioned")
        return 2 if any(w in answer_l for w in neg_words) else 0

    # Keyword overlap
    keywords = [w.strip(".,;()[]") for w in expected_l.split() if len(w) > 2]
    if not keywords:
        return 1  

    hits = sum(1 for kw in keywords if kw in answer_l)
    ratio = hits / len(keywords)
    if ratio >= 0.7:
        return 2
    if ratio >= 0.3:
        return 1
    return 0


def score_hallucination(answer: str, G) -> float:
    """
    Extract capitalised proper noun-like tokens from the answer
    Return fraction that are NOT valid graph nodes
    Lower is better (0 = no hallucinations)
    """
    all_nodes_lower = {str(n).lower() for n in G.nodes()}

    # Extract single capitalised words and two-word phrases
    tokens = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})?\b', answer)
    entity_like = [t for t in tokens if t.lower() not in _STOP]

    if not entity_like:
        return 0.0

    hallucinated = sum(1 for t in entity_like if t.lower() not in all_nodes_lower)
    return round(hallucinated / len(entity_like), 3)


def _get_planet(G, node):
    """Walk up 'contains' edges until a planet node is found."""
    visited = set()
    queue = [node]
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        if G.nodes[n].get("type") == "planet":
            return n
        for u, _, d in G.in_edges(n, data=True):
            if d.get("relation") == "contains":
                queue.append(u)
    return None


def score_locality(answer: str, anchor_node: str, G) -> float:
    """
    Check that all node names mentioned in the answer belong to the same planet as the anchor.  
    Returns fraction of mentioned nodes that are locally correct (1.0 = perfect locality)
    """
    answer_l = answer.lower()
    mentioned = [n for n in G.nodes() if str(n).lower() in answer_l]

    if not mentioned:
        return 1.0  # neutral, nothing to check

    anchor_planet = _get_planet(G, anchor_node)
    if anchor_planet is None:
        return 1.0

    correct = sum(
        1 for n in mentioned
        if _get_planet(G, n) == anchor_planet or n == anchor_planet
    )
    return round(correct / len(mentioned), 3)


def score_llm_judge(query: str, answer_key: str, response: str, model: str) -> tuple:
    """
    Ask an LLM judge to holistically rate a multi-hop response.
    The judge receives the question, a deterministic answer key built from graph data,
    and the system's response, then scores on conceptual correctness (entity identification
    + relational reasoning) w/ rubric given by me
    Returns score: int 1-5 or None on parse failure, reason: str
    """
    prompt, system = build_judge_prompt(query, answer_key, response)
    raw = query_llm(prompt, model=model, system=system, max_tokens=80)
    score_m  = re.search(r'SCORE:\s*([1-5])', raw)
    reason_m = re.search(r'REASON:\s*(.+)',   raw)
    score  = int(score_m.group(1))       if score_m  else None
    reason = reason_m.group(1).strip()   if reason_m else raw[:200]
    return score, reason


# Experiment runner                                                   

def run_experiment(G, queries, model="qwen2.5:7b", embed_model=EMBED_MODEL, verbose=True):
    """
    Run both systems (baseline + HG-RAG) on all queries.
    baseline: per-node chunks embedded with embed_model, top-10 retrieved by cosine similarity.
    hgrag:      k-hop subgraph around the anchor node, serialized as structured text.
    Returns a pandas DataFrame with one row per (query, system).
    """
    if verbose:
        print("Building vector RAG index...", end=" ", flush=True)
    rag_index = build_rag_index(G, embed_model=embed_model)
    if verbose:
        print(f"done — {len(rag_index)} chunks indexed")
    records = []

    for i, q in enumerate(queries):
        if verbose:
            print(f"  [{i+1}/{len(queries)}] {q['type']:15s}  {q['query'][:55]}...")

        anchor = resolve_entity(G, q["anchor"])
        if anchor is None:
            if verbose:
                print(f"    WARNING: anchor '{q['anchor']}' not found — skipping")
            continue

        # Multi-hop questions need more tokens to name all entities + explain reasoning.
        max_tok = 220 if q["type"] == "multi_hop" else 120

        # HG-RAG (NLP-extracted anchor node)
        nlp_anchor = extract_anchor(G, q["query"], model=model)

        # Type validation: fuzzy matching in large worlds can return a city when a country
        # is needed (or vice versa). If the extracted type mismatches the ground-truth
        # anchor type, fall back to ground-truth. The entity name is in the query, so
        # this isn't privileged information just a guarding against misfires.
        # (Explicitly mention this limitation in the paper)
        if nlp_anchor is not None:
            nlp_type = G.nodes.get(nlp_anchor, {}).get("type")
            gt_type  = G.nodes.get(anchor,     {}).get("type")
            if nlp_type != gt_type:
                if verbose:
                    print(f"    WARNING: extraction type mismatch ({nlp_type} vs expected {gt_type}) — using ground-truth anchor")
                nlp_anchor = anchor

        if nlp_anchor is not None:
            subgraph = retrieve_subgraph(G, nlp_anchor, k_up=2, k_side=1, k_down=0)
            hg_context = serialize_context(subgraph)
            prompt_b, sys_b = build_hgrag_prompt(hg_context, q["query"])
            answer_b = query_llm(prompt_b, model=model, system=sys_b, max_tokens=max_tok)
        else:
            hg_context = ""
            answer_b = "[Entity extraction failed]"
            if verbose:
                print(f"    WARNING: NLP extraction failed — '{q['query'][:50]}'")

        # Simple RAG baseline: retrieve top-10 chunks by cosine similarity
        rag_context = retrieve_rag_chunks(q["query"], rag_index, embed_model=embed_model)
        prompt_a, sys_a = build_rag_prompt(rag_context, q["query"])
        answer_a = query_llm(prompt_a, model=model, system=sys_a, max_tokens=max_tok)

        systems = [
            ("baseline",   answer_a, anchor,              rag_context),
            ("hgrag",      answer_b, nlp_anchor or anchor, hg_context),
        ]

        for system, answer, sys_anchor, sys_context in systems:
            factual = score_factual_accuracy(answer, q["expected"])

            llm_judge_score  = None
            llm_judge_reason = None
            if q["type"] == "multi_hop" and q.get("answer_key"):
                llm_judge_score, llm_judge_reason = score_llm_judge(
                    q["query"], q["answer_key"], answer, model
                )

            records.append({
                "query_id":           i,
                "query":              q["query"],
                "query_type":         q["type"],
                "anchor":             q["anchor"],
                "extracted_anchor":   str(nlp_anchor) if nlp_anchor else None,
                "expected":           q["expected"],
                "answer_key":         q.get("answer_key"),
                "system":             system,
                "answer":             answer,
                "factual_accuracy":   factual,
                "hallucination_rate": score_hallucination(answer, G),
                "locality_awareness": score_locality(answer, sys_anchor, G),
                "llm_judge_score":    llm_judge_score,
                "llm_judge_reason":   llm_judge_reason,
                "context_chars":      len(sys_context),
            })

    return pd.DataFrame(records)


# Display results                                     
METRICS = ["factual_accuracy", "hallucination_rate", "locality_awareness"]


def _build_summary_text(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("OVERALL RESULTS (mean per system)")
    lines.append("=" * 60)
    lines.append(df.groupby("system")[METRICS].mean().round(3).to_string())

    lines.append("\n" + "=" * 60)
    lines.append("RESULTS BY QUERY TYPE")
    lines.append("=" * 60)
    lines.append(df.groupby(["query_type", "system"])[METRICS].mean().round(3).to_string())

    # Multi-hop LLM judge breakdown
    mh = df[df["query_type"] == "multi_hop"]
    if not mh.empty:
        mh_judged = mh[mh["llm_judge_score"].notna()]
        if not mh_judged.empty:
            lines.append("\n" + "=" * 60)
            lines.append("MULTI-HOP LLM JUDGE SCORE (mean per system, scale 1-5)")
            lines.append("=" * 60)
            lines.append(mh_judged.groupby("system")["llm_judge_score"].mean().round(3).to_string())

        # Per-question detail: answer key + both system responses
        lines.append("\n" + "=" * 60)
        lines.append("MULTI-HOP QUESTION DETAILS")
        lines.append("=" * 60)
        has_size = "world_size" in df.columns
        group_keys = ["world_size", "query_id"] if has_size else ["query_id"]
        for keys, group in mh.groupby(group_keys, sort=True):
            row = group.iloc[0]
            if has_size:
                world_size, qid = keys
                lines.append(f"\n[Q{int(qid)+1} | {world_size}] {row['query']}")
            else:
                qid = keys
                lines.append(f"\n[Q{int(qid)+1}] {row['query']}")
            lines.append(f"  Answer Key : {row['answer_key']}")
            for _, r in group.iterrows():
                score_str = f"  (judge: {int(r['llm_judge_score'])})" if pd.notna(r["llm_judge_score"]) else ""
                lines.append(f"  {r['system'].upper():8s}{score_str}: {r['answer']}")

    return "\n".join(lines)


def print_summary(df: pd.DataFrame):
    print("\n" + _build_summary_text(df))


def save_summary(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_build_summary_text(df))
    print(f"Summary saved → {path}")
