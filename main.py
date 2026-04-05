"""
main.py is the main file to begin the HG-RAG experiment 

Usage examples:
  python main.py --size small
  python main.py --size medium --model mistral:latest
  python main.py --size all    --output results/
  python main.py --size small  --demo          # 5 queries only and prints answers, run to make sure llm works
"""

import argparse
import os
import sys
import pandas as pd

from world_builder import generate_world, world_stats
from queries import generate_queries
from evaluate import run_experiment, save_summary, METRICS
from llm import check_ollama, DEFAULT_MODEL

# main function that parses  arguments and runs the experiment
def main():
    parser = argparse.ArgumentParser(
        description="HG-RAG Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "all"],
        default="small",
        help="World size regime (default: small)\n"
             "  small  ~18 cities  (control)\n"
             "  medium ~150 cities (main result)\n"
             "  large  ~800 cities (stress test)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory for CSV output (default: results/)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run only 5 queries and print full answers",
    )
    args = parser.parse_args()

    # LLM check
    print("Checking Ollama connection...", end=" ", flush=True)
    if not check_ollama():
        print("FAILED")
        print("\nOllama is not running. Start it with:  ollama serve")
        print(f"Then pull a model with:                ollama pull {DEFAULT_MODEL}")
        sys.exit(1)
    print("OK")

    os.makedirs(args.output, exist_ok=True) # output directory

    sizes = ["small", "medium", "large"] if args.size == "all" else [args.size]
    all_dfs = []

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  World size : {size}")
        print(f"  Model      : {args.model}")
        print(f"{'='*60}")

        # Build world graph
        print("Building world graph...", end=" ", flush=True)
        G = generate_world(size=size, seed=args.seed)
        stats = world_stats(G)
        print(f"done — {stats.get('planet',0)} planets, "
              f"{stats.get('country',0)} countries, "
              f"{stats.get('city',0)} cities, "
              f"{G.number_of_edges()} edges")

        # Generate queries
        max_q = 5 if args.demo else 50
        queries = generate_queries(G, seed=args.seed, max_queries=max_q)
        print(f"Queries generated: {len(queries)}")

        # Run experiment
        print("Running experiment (both systems per query)...\n")
        df = run_experiment(G, queries, model=args.model, verbose=True)

        if df.empty:
            print("No results — check that anchor nodes were resolved.")
            continue

        df["world_size"] = size
        all_dfs.append(df)

        # print brief metrics to terminal
        print(df.groupby("system")[["factual_accuracy", "hallucination_rate", "locality_awareness"]].mean().round(3).to_string())
        mh = df[(df["query_type"] == "multi_hop") & df["llm_judge_score"].notna()]
        if not mh.empty:
            print("\nMulti-hop LLM judge (mean 1-5):")
            print(mh.groupby("system")["llm_judge_score"].mean().round(3).to_string())

        # Save per-size CSV + summary
        out_path = os.path.join(args.output, f"results_{size}.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")
        summary_path = os.path.join(args.output, f"summary_{size}.txt")
        save_summary(df, summary_path)

    # combined output for multi-size runs
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(args.output, "results_all.csv")
        combined.to_csv(combined_path, index=False)

        print("\n" + "=" * 60)
        print("RESULTS BY WORLD SIZE + SYSTEM")
        print("=" * 60)
        print(combined.groupby(["world_size", "system"])[METRICS].mean().round(3).to_string())
        print(f"\nCombined results saved → {combined_path}")
        save_summary(combined, os.path.join(args.output, "summary_all.txt"))


if __name__ == "__main__":
    main()
