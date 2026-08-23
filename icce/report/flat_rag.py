"""
Flat retrieval baseline for the grounding ablation.

To claim that the *graph* helps we must rule out the cheaper explanation:
"any historical context helps". This store keeps one text record per past
observation and retrieves the top-k by cosine similarity over TF-IDF bags of
words -- the standard RAG baseline, with no graph structure, no community
summaries and no entity aggregation.

Pure Python and dependency-free so it cannot silently differ from the graph
condition through some third-party retriever's behaviour.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Record:
    key: str
    text: str
    lat: float
    lon: float
    tf: Counter = field(default_factory=Counter)


class FlatRagStore:
    def __init__(self, radius_deg: Optional[float] = None) -> None:
        self.records: List[Record] = []
        self.df: Counter = Counter()
        self.radius_deg = radius_deg

    def add(self, key: str, text: str, lat: float, lon: float) -> None:
        tf = Counter(_tok(text))
        self.records.append(Record(key, text, lat, lon, tf))
        for term in tf:
            self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = len(self.records)
        return math.log((1 + n) / (1 + self.df.get(term, 0))) + 1.0

    def _vec(self, tf: Counter) -> Dict[str, float]:
        v = {t: (1 + math.log(c)) * self._idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(
        self,
        query: str,
        lat: float,
        lon: float,
        k: int = 5,
        exclude_key: Optional[str] = None,
    ) -> List[Tuple[float, Record]]:
        if not self.records:
            return []
        qv = self._vec(Counter(_tok(query)))
        hits: List[Tuple[float, Record]] = []
        for r in self.records:
            if exclude_key is not None and r.key == exclude_key:
                continue
            if self.radius_deg is not None:
                if math.hypot(r.lat - lat, r.lon - lon) > self.radius_deg:
                    continue
            rv = self._vec(r.tf)
            if len(qv) > len(rv):
                qv_, rv_ = rv, qv
            else:
                qv_, rv_ = qv, rv
            score = sum(w * rv_.get(t, 0.0) for t, w in qv_.items())
            if score > 0:
                hits.append((score, r))
        hits.sort(key=lambda t: -t[0])
        return hits[:k]

    def context_block(self, query: str, lat: float, lon: float, k: int = 5,
                      exclude_key: Optional[str] = None) -> str:
        hits = self.search(query, lat, lon, k, exclude_key)
        if not hits:
            return ""
        lines = ["=== RETRIEVED PAST OBSERVATIONS (flat top-k) ==="]
        for score, r in hits:
            lines.append(f"  [{score:.2f}] {r.text}")
        lines.append("=== END RETRIEVED OBSERVATIONS ===")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.records)
