#!/usr/bin/env python3
"""
CLI for Video Search Engine
Usage:
  python cli.py index /path/to/video.mp4
  python cli.py search "person carrying a bag"
  python cli.py search "two people talking after 18:00" --top-k 5
  python cli.py benchmark
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src import IndexingPipeline, QueryEngine


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/cli.log"),
        ],
    )


def cmd_index(args):
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.verbose)

    pipeline = IndexingPipeline(
        index_dir=args.index_dir,
        thumbnail_dir=args.thumbnail_dir,
        log_dir="logs",
        batch_size=args.batch_size,
        temporal_window=args.temporal_window,
        uniform_interval_sec=args.sample_interval,
    )

    bench = pipeline.run(args.video_input, force_reindex=args.force)

    print("\n" + "=" * 50)
    print("INDEXING COMPLETE")
    print("=" * 50)
    for k, v in bench.items():
        print(f"  {k:<35} {v}")
    print("=" * 50)


def cmd_search(args):
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.verbose)

    qe = QueryEngine(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
    )

    results = qe.search(
        query=args.query,
        top_k=args.top_k,
        video_filter=args.video_filter,
    )

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}  |  Latency: {results[0].get('query_latency_ms', '?')}ms")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        print(
            f"\n[{i:02d}] {r['timestamp_str']}  score={r['score']:.4f}"
        )
        print(f"     Video : {Path(r.get('video_path','?')).name}")
        print(f"     Thumb : {r.get('thumbnail_path','—')}")

    print(f"\nResults saved to {args.output_dir}/results.json & results.csv")


def cmd_benchmark(args):
    bench_path = Path("logs/benchmark.json")
    if not bench_path.exists():
        print("No benchmark found. Run 'index' first.")
        return
    with open(bench_path) as f:
        bench = json.load(f)
    print("\nBENCHMARK REPORT")
    print("=" * 50)
    for k, v in bench.items():
        print(f"  {k:<35} {v}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Video Search Engine — Natural Language Video Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py index /data/cctv.mp4
  python cli.py index /data/videos/ --sample-interval 3 --force
  python cli.py search "person with red bag near door"
  python cli.py search "two people talking after 18:00" --top-k 5
  python cli.py benchmark
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--thumbnail-dir", default="thumbnails")
    parser.add_argument("--output-dir", default="outputs")

    sub = parser.add_subparsers(dest="command")

    # ── index ──────────────────────────────────────────────────────────
    idx = sub.add_parser("index", help="Index a video or directory of videos")
    idx.add_argument("video_input", help="Path to video file or directory")
    idx.add_argument("--force", "-f", action="store_true", help="Force re-index")
    idx.add_argument("--batch-size", type=int, default=32)
    idx.add_argument("--sample-interval", type=float, default=5.0,
                     help="Uniform sample every N seconds (fallback)")
    idx.add_argument("--temporal-window", type=int, default=2,
                     help="Temporal smoothing window (±N frames)")

    # ── search ─────────────────────────────────────────────────────────
    srch = sub.add_parser("search", help="Search the indexed video archive")
    srch.add_argument("query", help="Natural language query")
    srch.add_argument("--top-k", type=int, default=10)
    srch.add_argument("--video-filter", help="Filter results by video name substring")

    # ── benchmark ──────────────────────────────────────────────────────
    sub.add_parser("benchmark", help="Display indexing benchmark report")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
