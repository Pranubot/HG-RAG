# HG-RAG: Hierarchy-Guided Retrieval-Augmented Generation

A research project done by Pranav Yadav for CSE 188 : Natural Language Processing in the University of California, Merced. This is the code used in my paper benchmarking **HG-RAG** (Hierarchy-Guided RAG) against a flat-context baseline on a synthetic hierarchical world graph. The experiment measures whether hierarchy-aware subgraph retrieval improves factual accuracy, reduces hallucinations, and enables multi-hop reasoning compared to dumping the entire world into an LLM's context window.

Details of the study including an more in-depth explanation on research methods are present in my paper, found here:

---

## Overview

The world is modeled as a three-level hierarchy: **Planets → Countries → Cities**, with lateral relations (trade, borders, hostility) between nodes at the same level. Two retrieval systems answer the same set of questions:

| Architecture | Strategy |
|---|---|
| **baseline** | Dense vector RAG: one chunk per node, top-10 retrieved by cosine similarity (nomic-embed-text) |
| **hgrag**    | Hierarchy-aware subgraph: anchor node + k_up=2 ancestors + k_side=1 lateral neighbors |

HG-RAG uses an LLM to extract the anchor entity from each query, walks up the containment hierarchy, and collects relevant lateral neighbors, prioritizing adversarial neighbors (hostile/unfriendly) so they are never dropped by the neighbor cap.

---

## File Structure

```
.
├── main.py          # CLI entry point - parses args and orchestrates the experiment
├── world_builder.py # Generates procedural hierarchical world graphs (NetworkX DiGraph)
├── queries.py       # Generates 50 deterministic queries with ground-truth answers
├── retrieval.py     # HG-RAG retrieval pipeline (entity resolution, subgraph traversal, serialization)
├── evaluate.py      # Scoring + experiment runner (factual accuracy, hallucination, locality, multi-hop with LLM judge)
├── llm.py           # Thin Ollama wrapper + all prompt templates
├── requirements.txt 
└── results/         # Output CSVs and summary text files
```

---

## Prerequisites

**1. Ollama must be running:**
```bash
ollama serve
```

**2. Pull the required models (one-time):**
```bash
ollama pull mistral:latest
ollama pull nomic-embed-text
```

**3. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

---

## Running Experiments

**Quick smoke test** - 5 queries, prints answers to terminal:
```bash
python main.py --demo
```

**Small world** (~18 cities):
```bash
python main.py --size small
```

**Medium world** (~150 cities):
```bash
python main.py --size medium
```

**Large world** (~800 cities):
```bash
python main.py --size large
```

**All three sizes in sequence:**
```bash
python main.py --size all
```

### Optional Flags

| Flag | Default | Description |
|---|---|---|
| `--model`       | `mistral:latest`    | Ollama model for generation and entity extraction |
| `--embed_model` | `nomic-embed-text`  | Ollama model for baseline RAG embeddings |
| `--seed`        | `42`                | Random seed for reproducibility (any int) |
| `--output`      | `results/`          | Folder for output files |

---

## World Sizes

| Size | Structure | Total Cities |
|---|---|---|
| Small  | 2 planets → 3 countries each → 3 cities each  | ~18  |
| Medium | 3 planets → 5 countries each → 10 cities each | ~150 |
| Large  | 4 planets → 8 countries each → 25 cities each | ~800 |

---

## Query Types (25% of total question pool each)

| Query Type | Description | Example |
|---|---|---|
| `local_fact`   | Direct attribute lookup          | "What is the population of Pranaville?" |
| `hierarchical` | Parent chain questions           | "What country is Pranavaria in?" |
| `neighborhood` | Border/trade neighbor questions  | "Which cities share a border with Pranuhaven?" |
| `multi_hop`    | Causal reasoning across the graph | "A labor strike halts exports from Prandale - which trade partners are affected?" |

Multi-hop queries cover three templates:
**Export disruption** - labor strike halting a city's exports
**Import halt** - city stops importing; trade partners lose a market
**War declaration** - country declares war on a hostile/unfriendly neighbor

---

## Evaluation Metrics

**All query types:**
| Metric | Range | Description |
|---|---|---|
| `factual_accuracy`   | 0–2 | Keyword overlap with ground-truth; 2 = ≥70% match, 1 = ≥30%, 0 = below 30% |
| `hallucination_rate` | 0–1 | Fraction of capitalized entity-like tokens in the answer that are not valid graph nodes (lower is better) |
| `locality_awareness` | 0–1 | Fraction of graph nodes mentioned in the answer that belong to the same planet as the anchor (1.0 = perfect) |

**Multi-hop only (additional):**

| Metric | Range | Description |
|---|---|---|
| `llm_judge_score`  | 1–5 | Holistic LLM judge rating: correct entities + reasoning |
| `llm_judge_reason` | N/A | One-sentence explanation from the judge |

The LLM judge rubric weights entity correctness heavily, a well-reasoned answer naming wrong entities cannot score above 2.

---

## Output Files

After each run, the `results/` folder contains:

```
results_{size}.csv       # Raw per-query DataFrame (one row per query per system)
summary_{size}.txt       # Human-readable summary tables:
                         #   - Overall metrics by system
                         #   - Metrics broken down by query type
                         #   - Multi-hop LLM judge scores
                         #   - Multi-hop question details with answer keys
```

For `--size all`:
```
results_all.csv
summary_all.txt
```
