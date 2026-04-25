"""Helpers for temporal evidence handling."""

from __future__ import annotations

import re
from typing import Iterable


YEAR_RE = re.compile(r"(19|20)\d{2}")


def parse_year(timestamp: str | None) -> int | None:
    """Extract a 4-digit year from a timestamp-like string."""

    if not timestamp:
        return None
    match = YEAR_RE.search(str(timestamp))
    if not match:
        return None
    return int(match.group(0))


def choose_earliest_and_latest(items: Iterable[tuple[str, str | None, float]]) -> tuple[str | None, str | None]:
    """Choose doc IDs for earliest and latest scored documents."""

    by_doc: dict[str, tuple[int, float]] = {}
    for doc_id, timestamp, score in items:
        year = parse_year(timestamp)
        if year is None:
            continue
        existing = by_doc.get(doc_id)
        if existing is None or score > existing[1]:
            by_doc[doc_id] = (year, score)
    if len(by_doc) < 2:
        return None, None

    sorted_docs = sorted(by_doc.items(), key=lambda item: (item[1][0], -item[1][1], item[0]))
    earliest = sorted_docs[0][0]
    latest = sorted_docs[-1][0]
    if earliest == latest:
        return None, None
    return earliest, latest
