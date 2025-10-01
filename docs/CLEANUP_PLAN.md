# Codebase Cleanup Plan

## 🎯 Analysis Summary

After ultra-deep analysis of the codebase, identified **significant redundancy and complexity**. This document provides a systematic cleanup plan to streamline the project without losing functionality.

## 📊 Current State

**Total Files**: ~850 files (including venv)
**Scripts**: 25 in scripts/, 11 in experiments/scripts/
**Documentation**: 43 markdown files in docs/
**Problem**: Too many overlapping/stub/obsolete files

---

## 🔴 CRITICAL: Redundant & Duplicate Files

### 1. Duplicate Documentation (REMOVE ONE)

#### Production Deployment (DUPLICATE - Different content)
- `docs/Production_Deployment_Guide.md` (312 lines) - **KEEP** (comprehensive guide)
- `docs/production_deployment.md` (40 lines) - **DELETE** (case study stub)

**Action**: Delete `docs/production_deployment.md`

#### Table Aggregation Scripts (DUPLICATE LOGIC)
- `experiments/scripts/aggregate_table1.py` (151 lines) - **KEEP** (full version)
- `experiments/scripts/aggregate_table1_minimal.py` (92 lines) - **DELETE** (minimal version)

**Action**: Delete `experiments/scripts/aggregate_table1_minimal.py`

### 2. Overlapping Functionality (CONSOLIDATE)

#### Hardware Checking Scripts
- `scripts/check_hardware_setup.py` ✅ **KEEP** - NEW comprehensive check
- `scripts/check_cuda_env.py` (58 lines) - **DEPRECATE** - Old, specific version check only

**Reason**: `check_hardware_setup.py` does everything `check_cuda_env.py` does, plus more:
- Checks CUDA availability
- Finds NCU binary
- Checks all Python dependencies
- Provides actionable recommendations

**Action**: Add deprecation warning to `check_cuda_env.py`, plan to remove in next release

#### Validation & Metrics Scripts
- `scripts/validate_mechanistic_claim.py` ✅ **KEEP** - NEW comprehensive validation
- `scripts/extract_mechanism_metrics.py` (157 lines) - **KEEP BUT CLARIFY** - Extracts from existing runs
- `scripts/mediation_analysis.py` (97 lines) - **KEEP BUT CLARIFY** - Statistical analysis

**Reason**: These serve different purposes:
- `validate_mechanistic_claim.py` - Runs NEW experiments, generates data
- `extract_mechanism_metrics.py` - Parses EXISTING metrics.json files
- `mediation_analysis.py` - Performs Baron-Kenny statistical analysis

**Action**: Add clear docstrings distinguishing their purposes

---

## 🟡 STUB/PLACEHOLDER DOCUMENTATION (CLEAN UP)

### Tiny Stub Files (<10 lines, no real content)

**DELETE (Archive to docs/stubs/):**
1. `docs/Change_Announcement_Playbook.md` (5 lines)
2. `docs/Doc_Update_Backlog.md` (5 lines)
3. `docs/Env_Fingerprint_Privacy.md` (5 lines)
4. `docs/External_Reference_Policy.md` (5 lines)
5. `docs/Milestone_Definitions.md` (5 lines)
6. `docs/Regression_Baseline_Guide.md` (5 lines)
7. `docs/StableHLO_Build.md` (5 lines)
8. `docs/Iterative_CoDesign_Checklist.md` (6 lines)
9. `docs/CI_Matrix.md` (7 lines)
10. `docs/Resource_Plan.md` (7 lines)
11. `docs/Completeness_Scoring.md` (9 lines)

**Rationale**: These are planning stubs that were never fleshed out. Archiving instead of deleting preserves history without cluttering main docs/.

**Action**:
```bash
mkdir -p docs/archive/stubs
mv docs/Change_Announcement_Playbook.md docs/archive/stubs/
mv docs/Doc_Update_Backlog.md docs/archive/stubs/
# ... etc for all 11 files
```

---

## 🟢 POTENTIALLY OBSOLETE (EVALUATE & REMOVE)

### Scripts Replaced by New Infrastructure

#### 1. `scripts/nsight_stub.py` (25 lines)
**Status**: Likely obsolete
**Reason**: We now have real NCU integration in `icd/measure/l2_ncu.py`
**Action**: Verify not used anywhere, then DELETE

#### 2. `scripts/check_gap_status.py` (362 lines)
**Status**: Historical, now obsolete
**Reason**: Analyzed gaps between paper and implementation. Gaps are now CLOSED.
**Action**: Move to `docs/archive/historical/` for reference

---

## 🔵 OVERLAPPING DOCUMENTATION (CONSOLIDATE)

### Getting Started Guides

**Current situation:**
- `docs/QUICKSTART.md` (62 lines) - Quick start
- `docs/USAGE.md` (280 lines) - Detailed usage
- `docs/VALIDATION_README.md` ✅ (346 lines) - NEW validation guide
- `README.md` - Main entry point

**Problem**: Users don't know where to start

**Recommendation**:
1. **KEEP** `README.md` - Main landing page with links to everything
2. **KEEP** `docs/VALIDATION_README.md` - NEW, for real experiments
3. **MERGE** `QUICKSTART.md` → `USAGE.md` as first section
4. **DELETE** standalone `QUICKSTART.md`

### Architecture & Operations Docs

**Current situation:**
- `docs/ICD.md` (464 lines) - Interface design document
- `docs/PRD.md` (222 lines) - Product requirements
- `docs/SAS.md` (368 lines) - System architecture & security
- `docs/SOP.md` (257 lines) - Standard operating procedures

**Problem**: Significant overlap in describing the system

**Recommendation**:
- **KEEP ALL** but add clear navigation at the top of each:
  - PRD: What we're building (requirements)
  - ICD: How interfaces work (technical design)
  - SAS: Architecture & security (ops view)
  - SOP: Operational procedures (runbook)

**Action**: Add cross-references between these docs

### Hardware & Profiling Docs

**Current situation:**
- `docs/Hardware_Profiling_Integration.md` ✅ (304 lines) - NEW, comprehensive
- `docs/Hardware_Requirements.md` (335 lines) - Pre-existing hardware specs
- `docs/Cross_Vendor_Profiling.md` (330 lines) - ROCm, VTune integration

**Assessment**:
- `Hardware_Profiling_Integration.md` - NVIDIA/CUDA specific
- `Hardware_Requirements.md` - General requirements, some overlap
- `Cross_Vendor_Profiling.md` - AMD/Intel specific

**Recommendation**: **KEEP ALL** - They cover different aspects:
- Integration guide: HOW to set up profiling
- Requirements: WHAT hardware you need
- Cross-vendor: Support for non-NVIDIA

**Action**: Add navigation section linking these three

---

## 🟣 EXPERIMENTAL SCRIPTS (REORGANIZE)

### Current State: `experiments/scripts/`

```
experiments/scripts/
├── aggregate_quantization.py
├── aggregate_table1.py
├── aggregate_table1_minimal.py          ❌ DELETE (duplicate)
├── run_experimental_matrix.py           ✅ KEEP (called by generate_paper_data.py)
├── fix_directories.sh
├── run_mechanism_deepdive.sh
├── run_quantization_batch.sh
├── run_table1_batch.sh
└── runpod_quickstart.sh
```

**Problem**: Mix of aggregation scripts, batch runners, and platform-specific helpers

**Recommendation**: Reorganize into subdirectories:

```
experiments/scripts/
├── aggregation/               # Data aggregation
│   ├── aggregate_table1.py
│   └── aggregate_quantization.py
├── batch_runners/             # Batch execution
│   ├── run_experimental_matrix.py
│   ├── run_mechanism_deepdive.sh
│   ├── run_quantization_batch.sh
│   └── run_table1_batch.sh
└── platform/                  # Platform-specific
    ├── runpod_quickstart.sh
    └── fix_directories.sh
```

---

## 📋 CLEANUP PRIORITY LEVELS

### Priority 1: IMMEDIATE (Safe Deletions)
Execute these now - no dependencies:

1. Delete `docs/production_deployment.md` (duplicate)
2. Delete `experiments/scripts/aggregate_table1_minimal.py` (duplicate)
3. Archive 11 stub docs to `docs/archive/stubs/`
4. Delete `scripts/nsight_stub.py` (obsolete)

**Estimated cleanup**: ~13 files, ~500 lines

### Priority 2: NEAR-TERM (Verification Needed)
Within 1-2 weeks - verify no dependencies:

1. Move `scripts/check_gap_status.py` to `docs/archive/historical/`
2. Deprecate `scripts/check_cuda_env.py` (add warning, keep for compatibility)
3. Merge `QUICKSTART.md` into `USAGE.md`
4. Reorganize `experiments/scripts/` into subdirectories

**Estimated cleanup**: ~4 files moved/merged

### Priority 3: LONG-TERM (Documentation)
Ongoing - improve discoverability:

1. Add cross-references between architecture docs (PRD, ICD, SAS, SOP)
2. Add navigation section to hardware docs (3 guides)
3. Update README with clearer doc hierarchy
4. Add purpose docstrings to all scripts

---

## 🎯 FINAL RECOMMENDATIONS

### What to KEEP (Essential)

**Core Implementation**:
- All `icd/` modules ✅
- All `tests/` ✅
- All `configs/` ✅

**NEW Validation Infrastructure** ✅:
- `icd/measure/l2_ncu.py`
- `icd/measure/cuda_latency.py`
- `scripts/check_hardware_setup.py`
- `scripts/validate_mechanistic_claim.py`
- `scripts/generate_paper_data.py`
- `docs/VALIDATION_README.md`
- `docs/Hardware_Profiling_Integration.md`

**Core Documentation**:
- `README.md`
- `docs/USAGE.md` (merge with QUICKSTART)
- `docs/Gap_Analysis.md` (historical value)
- Architecture docs: PRD, ICD, SAS, SOP
- Theoretical docs: Theoretical_Analysis.md, Experimental_Procedures.md

**Experiment Infrastructure**:
- `experiments/scripts/run_experimental_matrix.py`
- `experiments/scripts/aggregate_*.py` (except minimal)
- Batch runner scripts

### What to DELETE

**Immediate**:
- `docs/production_deployment.md`
- `experiments/scripts/aggregate_table1_minimal.py`
- `scripts/nsight_stub.py`
- 11 stub docs (→ archive)

**After Verification**:
- `scripts/check_cuda_env.py` (deprecate first)
- `scripts/check_gap_status.py` (→ archive)

### What to REORGANIZE

**Merge**:
- `docs/QUICKSTART.md` → `docs/USAGE.md` (first section)

**Reorganize**:
- `experiments/scripts/` → subdirectories (aggregation/, batch_runners/, platform/)

**Add Cross-References**:
- Architecture docs (PRD ↔ ICD ↔ SAS ↔ SOP)
- Hardware docs (Requirements ↔ Integration ↔ Cross-Vendor)

---

## 📊 Impact Analysis

### Before Cleanup
- Scripts: 25 (scripts/) + 11 (experiments/scripts/) = **36 total**
- Docs: **43 markdown files**
- Clarity: **Low** (users confused where to start)

### After Cleanup
- Scripts: ~30 total (**-6 deleted/moved**)
- Docs: ~30 active + ~15 archived (**-13 from main view**)
- Clarity: **High** (clear navigation, no duplicates)

### Benefits
✅ Easier to navigate
✅ Clear purpose for each script/doc
✅ No duplicate content
✅ Historical files preserved in archive
✅ Faster onboarding for new contributors

---

## 🚀 Execution Plan

### Phase 1: Safe Deletions (Now)
```bash
# 1. Delete duplicates
git rm docs/production_deployment.md
git rm experiments/scripts/aggregate_table1_minimal.py
git rm scripts/nsight_stub.py

# 2. Archive stubs
mkdir -p docs/archive/stubs
git mv docs/Change_Announcement_Playbook.md docs/archive/stubs/
git mv docs/Doc_Update_Backlog.md docs/archive/stubs/
git mv docs/Env_Fingerprint_Privacy.md docs/archive/stubs/
git mv docs/External_Reference_Policy.md docs/archive/stubs/
git mv docs/Milestone_Definitions.md docs/archive/stubs/
git mv docs/Regression_Baseline_Guide.md docs/archive/stubs/
git mv docs/StableHLO_Build.md docs/archive/stubs/
git mv docs/Iterative_CoDesign_Checklist.md docs/archive/stubs/
git mv docs/CI_Matrix.md docs/archive/stubs/
git mv docs/Resource_Plan.md docs/archive/stubs/
git mv docs/Completeness_Scoring.md docs/archive/stubs/

git commit -m "chore: Remove duplicate and stub files

- Delete duplicate production deployment doc
- Delete duplicate table aggregation script
- Archive 11 stub/placeholder docs to docs/archive/stubs/
- Delete obsolete nsight_stub.py (replaced by real implementation)"
```

### Phase 2: Deprecations (This Week)
```bash
# Add deprecation warnings
# Then later delete after transition period
```

### Phase 3: Documentation Improvements (Ongoing)
```bash
# Merge, reorganize, cross-reference
```

---

## ✅ Checklist

- [ ] Phase 1: Delete duplicates and archive stubs
- [ ] Phase 2: Deprecate overlapping scripts
- [ ] Phase 3: Merge QUICKSTART into USAGE
- [ ] Phase 4: Reorganize experiments/scripts/
- [ ] Phase 5: Add cross-references to docs
- [ ] Phase 6: Update README with doc hierarchy
- [ ] Phase 7: Verify all tests still pass

---

**Status**: Ready for execution
**Estimated Time**: Phase 1 (30 min), Full cleanup (2-3 hours spread over 1-2 weeks)
**Risk**: Low (archiving instead of deleting, preserves git history)
