# RCA Sample: Iterative Perm Regression

- **Incident ID**: RCA-2025-02
- **Date**: 2025-02-15
- **Owners**: Runtime Ops
- **Summary**: Iterative permutation run regressed ΔJ gate due to stale permutation cache.
- **Impact**: Acceptance blocked for 3 hours.
- **Timeline**:
  - 09:00 UTC: Iterative run fails ΔJ gate.
  - 09:05 UTC: Cache identified as stale.
  - 09:30 UTC: Cache purged and run re-triggered.
- **Root Cause**: Cache invalidation did not consider cluster signature changes.
- **Mitigations**: Added cluster-aware signature to cache entries.
- **Follow-ups**: Monitor cache metrics for 7 days.
