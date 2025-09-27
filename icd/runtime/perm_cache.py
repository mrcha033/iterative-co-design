"""Permutation cache helpers used by the orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


def _signature_from_clusters(clusters: Sequence[Sequence[int]]) -> str:
    canonical = ["-".join(str(idx) for idx in sorted(cluster)) for cluster in clusters]
    canonical.sort()
    return ";".join(canonical)


@dataclass
class PermutationCacheEntry:
    pi: Sequence[int]
    hash: str
    signature: str
    updated_at: str
    meta: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "pi": list(map(int, self.pi)),
            "hash": self.hash,
            "signature": self.signature,
            "updated_at": self.updated_at,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PermutationCacheEntry":
        return cls(
            pi=list(map(int, data.get("pi", []))),
            hash=str(data.get("hash", "")),
            signature=str(data.get("signature", "")),
            updated_at=str(data.get("updated_at", "")),
            meta=dict(data.get("meta", {})),
        )


def load_entry(path: str | Path) -> PermutationCacheEntry | None:
    file = Path(path)
    if not file.exists():
        return None
    with file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("version") != 1:
        return None
    return PermutationCacheEntry.from_dict(data)


def save_entry(path: str | Path, pi: Sequence[int], hash_value: str, clusters: Sequence[Sequence[int]], meta: dict[str, object] | None = None) -> PermutationCacheEntry:
    signature = _signature_from_clusters(clusters)
    entry = PermutationCacheEntry(
        pi=list(pi),
        hash=hash_value,
        signature=signature,
        updated_at=datetime.now(timezone.utc).isoformat(),
        meta=dict(meta or {}),
    )
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        json.dump(entry.to_dict(), handle, indent=2, ensure_ascii=False)
    return entry


def should_invalidate(existing: PermutationCacheEntry, clusters: Sequence[Sequence[int]]) -> bool:
    return existing.signature != _signature_from_clusters(clusters)

