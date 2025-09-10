# Runtime & Memory Plan

One-line: Define KV-cache manager, streams/events, OOM/fragmentation handling, optional CUDA graph capture policy.

## KV-cache
- Blocked layout with `block` param; resizing and eviction policy tag.
- Hash key: model+task+S/Q/K meta+seed+device+driver → `pi_hash`.

## Streams/Events
- Determinism-first: single stream by default; events only for timing.
- File lock for artifact writes.

## OOM/Fragmentation
- Retry with smaller batch/sequence; log and rollback.

## Graph Capture (optional)
- Capture only in linear mode; iterative path excludes capture to avoid invalidation.

