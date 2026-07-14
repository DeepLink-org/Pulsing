# SPDX-License-Identifier: Apache-2.0
"""BM25-lite scoring for tool_search."""

from __future__ import annotations

import json
import math
import re


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t]


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    q_terms = tokenize(query)
    if not q_terms or not documents:
        return [0.0] * len(documents)

    doc_terms = [tokenize(d) for d in documents]
    n = float(len(documents))
    avgdl = sum(len(t) for t in doc_terms) / max(n, 1.0)

    df: dict[str, int] = {}
    for terms in doc_terms:
        seen: set[str] = set()
        for t in terms:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for terms in doc_terms:
        dl = float(len(terms))
        score = 0.0
        for qt in q_terms:
            tf = float(terms.count(qt))
            if tf == 0:
                continue
            df_q = float(df.get(qt, 0))
            idf = ((n - df_q + 0.5) / (df_q + 0.5)) + 1.0
            idf = math.log(idf)
            score += (
                idf
                * (tf * (k1 + 1.0))
                / (tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0)))
            )
        scores.append(score)
    return scores
