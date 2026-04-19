"""
Query Engine — natural language search with temporal filter parsing
Saves results to results.json / results.csv automatically
"""

import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import CLIPEmbedder
from .vector_index import VideoVectorIndex

logger = logging.getLogger(__name__)


def parse_time_filter(query: str) -> Tuple[str, Optional[Tuple[float, float]]]:
    """
    Parses temporal expressions from queries and returns
    (cleaned_query, (start_sec, end_sec)) or (query, None).

    Handles:
      "after 6pm"         → (21600, inf)
      "before 8:30"       → (0, 30600)
      "between 18:00 and 20:00"
      "after 1:30:00"
    """

    def hms_to_sec(h, m=0, s=0):
        return int(h) * 3600 + int(m) * 60 + int(s)

    # Pattern: "between HH:MM and HH:MM" or "from HH:MM to HH:MM"
    between = re.search(
        r"(between|from)\s+(\d{1,2})[:\.](\d{2})(?:[:\.](\d{2}))?"
        r"\s*(and|to)\s+(\d{1,2})[:\.](\d{2})(?:[:\.](\d{2}))?",
        query,
        re.IGNORECASE,
    )
    if between:
        s_h, s_m, s_s = between.group(2), between.group(3), between.group(4) or "0"
        e_h, e_m, e_s = between.group(6), between.group(7), between.group(8) or "0"
        start = hms_to_sec(s_h, s_m, s_s)
        end = hms_to_sec(e_h, e_m, e_s)
        cleaned = query[: between.start()].strip() + " " + query[between.end() :].strip()
        return cleaned.strip(), (start, end)

    # Pattern: "after HH:MM" or "after 6pm/am"
    after = re.search(
        r"after\s+(\d{1,2})(?:[:\.](\d{2}))?(?:[:\.](\d{2}))?\s*(am|pm)?",
        query,
        re.IGNORECASE,
    )
    if after:
        h, m, s = after.group(1), after.group(2) or "0", after.group(3) or "0"
        ampm = (after.group(4) or "").lower()
        h_int = int(h)
        if ampm == "pm" and h_int < 12:
            h_int += 12
        elif ampm == "am" and h_int == 12:
            h_int = 0
        start = hms_to_sec(h_int, m, s)
        cleaned = query[: after.start()].strip() + " " + query[after.end() :].strip()
        return cleaned.strip(), (start, float("inf"))

    # Pattern: "before HH:MM"
    before = re.search(
        r"before\s+(\d{1,2})(?:[:\.](\d{2}))?(?:[:\.](\d{2}))?\s*(am|pm)?",
        query,
        re.IGNORECASE,
    )
    if before:
        h, m, s = before.group(1), before.group(2) or "0", before.group(3) or "0"
        ampm = (before.group(4) or "").lower()
        h_int = int(h)
        if ampm == "pm" and h_int < 12:
            h_int += 12
        end = hms_to_sec(h_int, m, s)
        cleaned = query[: before.start()].strip() + " " + query[before.end() :].strip()
        return cleaned.strip(), (0, end)

    return query, None


class QueryEngine:
    def __init__(
        self,
        index_dir: str = "index",
        output_dir: str = "outputs",
        embedder: Optional[CLIPEmbedder] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vector_index = VideoVectorIndex(index_dir=index_dir)
        self.vector_index.load()

        self.embedder = embedder or CLIPEmbedder()
        self.query_log: List[Dict] = []

    def search(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Main search entry point.
        Automatically parses temporal filters from the query.
        """
        t0 = time.time()

        # Parse temporal filter
        cleaned_query, time_filter = parse_time_filter(query)
        effective_query = cleaned_query if cleaned_query else query

        logger.info(
            f"Query: '{query}' | Cleaned: '{effective_query}' "
            f"| Time filter: {time_filter}"
        )

        # Encode text
        query_embedding = self.embedder.embed_text(effective_query)

        # ANN search
        results = self.vector_index.search(
            query_embedding,
            top_k=top_k,
            time_filter=time_filter,
            video_filter=video_filter,
        )

        total_ms = (time.time() - t0) * 1000

        # Enrich results
        for r in results:
            r["query"] = query
            r["query_latency_ms"] = round(total_ms, 2)
            r["queried_at"] = datetime.now().isoformat()

        self.query_log.extend(results)
        self._save_results(results, query)

        logger.info(f"Returned {len(results)} results in {total_ms:.1f}ms")
        return results

    def _save_results(self, results: List[Dict], query: str):
        """Append to results.json and results.csv."""
        # JSON
        json_path = self.output_dir / "results.json"
        existing = []
        if json_path.exists():
            with open(json_path) as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        existing.extend(results)
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)

        # CSV
        csv_path = self.output_dir / "results.csv"
        write_header = not csv_path.exists()
        fields = [
            "query", "timestamp_str", "timestamp_sec", "score",
            "video_path", "thumbnail_path", "query_latency_ms", "queried_at",
        ]
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(results)

        logger.info(f"Results saved to {json_path} and {csv_path}")
