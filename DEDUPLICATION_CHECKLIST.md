# Code Deduplication Checklist - T-013

**Generated**: 2025-07-12  
**Status**: High Priority - 394 duplication patterns identified  
**Files Analyzed**: 64 Python files (20,611 total lines)

## Executive Summary

🔴 **CRITICAL**: High duplication severity detected requiring immediate attention
- **Import duplications**: 38 patterns
- **Function duplications**: 57 patterns  
- **Class duplications**: 14 patterns
- **Code block patterns**: 50 patterns
- **Constants/literals**: 235 patterns

## High Priority Deduplication Opportunities

### 1. 🔥 Critical Function Duplications

#### A. Result Loading Infrastructure (IMMEDIATE ACTION REQUIRED)
**Files**: `scripts/generate_figures.py` vs `scripts/generate_tables.py`

**Duplicated Functions**:
- `__init__(self, results_dir, output_dir)` - Lines 62 vs 24
- `_load_results(self)` - Lines 77 vs 39
- `_find_experiment(self, model, strategy)` - Lines 481 vs 263
- `_extract_latency(self, exp_data)` - Lines 500 vs 282

**Impact**: Nearly identical infrastructure classes with ~400 lines of duplication
**Strategy**: Create shared `BaseResultAnalyzer` class in `src/utils/result_analyzer.py`

**Action Items**:
- [ ] Create `src/utils/result_analyzer.py` with `BaseResultAnalyzer` class
- [ ] Extract common methods: `_load_results`, `_find_experiment`, `_extract_latency`
- [ ] Refactor `FigureGenerator` and `TableGenerator` to inherit from base class
- [ ] Update imports in dependent files
- [ ] Run tests to ensure functionality preserved

#### B. GPU Information Utilities (HIGH PRIORITY)
**Files**: `scripts/run_experiment.py` vs `src/profiler/calibration.py`

**Duplicated Functions**:
- `_get_gpu_info(self)` - Lines 152 vs 189

**Impact**: Identical GPU detection logic in multiple modules
**Strategy**: Create shared `src/utils/gpu_utils.py` module

**Action Items**:
- [ ] Create `src/utils/gpu_utils.py` with `get_gpu_info()` function
- [ ] Update both files to import and use shared function
- [ ] Standardize GPU info format across all modules
- [ ] Add unit tests for GPU info utility

#### C. Permutation Management Duplication (HIGH PRIORITY)
**Files**: `src/co_design/apply.py` vs `src/models/permutable_model.py`

**Duplicated Functions**:
- `get_applied_permutations(self)` - Lines 454 vs 274
- `has_permutation(self, layer_name)` - Lines 458 vs 278

**Impact**: Core permutation tracking logic duplicated
**Strategy**: Consolidate into shared mixin class or utility

**Action Items**:
- [ ] Create `PermutationTrackingMixin` in `src/co_design/mixins.py`
- [ ] Extract common permutation tracking methods
- [ ] Update both classes to use mixin
- [ ] Ensure consistent permutation storage format
- [ ] Update unit tests for both modules

### 2. 🟡 Medium Priority Duplications

#### A. Cache Information Methods
**Files**: `src/co_design/correlation.py` vs `src/models/manager.py`

**Duplicated**: `get_cache_info(self)` methods
**Strategy**: Move to shared utility or base class

#### B. HDS Temperature Management
**Files**: Multiple locations in `src/co_design/hds.py`

**Duplicated**: `update_temperature(self, new_temperature)` - 3 occurrences
**Strategy**: Extract to shared method or use inheritance

#### C. Configuration Initialization
**Multiple Files**: 8 classes with identical `__init__(self, config)` pattern

**Strategy**: Create `ConfigurableBase` class with standard config handling

### 3. 🟢 Lower Priority Optimizations

#### A. Import Consolidation
**High-frequency imports** (suggest common import modules):
- `from pathlib import Path` - 36 occurrences
- `from typing import Dict` - 21 occurrences  
- `from typing import Optional` - 21 occurrences
- `from typing import List` - 18 occurrences

**Strategy**: Consider common import module or type alias file

#### B. Test Infrastructure
**Common test patterns**:
- `__init__(self)` in test classes - 27 occurrences
- `forward(self, x)` in test models - 28 occurrences

**Strategy**: Create shared test utilities and base classes

## Implementation Sequence

### Phase 1: Critical Infrastructure (Week 1)
1. **BaseResultAnalyzer** - Consolidate `generate_figures.py` and `generate_tables.py`
2. **GPU Utilities** - Extract `_get_gpu_info` to shared module
3. **Permutation Tracking** - Create mixin for permutation management

### Phase 2: Module Consolidation (Week 2)  
1. **Configuration Base Class** - Standardize config initialization
2. **HDS Refactoring** - Eliminate temperature management duplication
3. **Cache Information** - Consolidate cache info methods

### Phase 3: Test and Import Optimization (Week 3)
1. **Test Base Classes** - Create shared test infrastructure
2. **Import Cleanup** - Optimize common imports
3. **Validation and Testing** - Ensure all changes work correctly

## Risk Assessment

### High Risk Changes
- **BaseResultAnalyzer**: Core analysis infrastructure - requires careful testing
- **Permutation Tracking**: Critical to IASP functionality - validate thoroughly

### Medium Risk Changes  
- **GPU Utilities**: Used in profiling - ensure compatibility across environments
- **Configuration Base**: Wide impact - gradual rollout recommended

### Low Risk Changes
- **Import consolidation**: Primarily cosmetic - low functional impact
- **Test infrastructure**: Isolated to test environment

## Validation Plan

### Pre-Refactoring
- [ ] Run full test suite to establish baseline
- [ ] Document current functionality of each duplicated component
- [ ] Create integration tests for critical paths

### During Refactoring
- [ ] Maintain backward compatibility during transition
- [ ] Test each module individually after changes
- [ ] Verify no regression in functionality

### Post-Refactoring  
- [ ] Run complete test suite including performance tests
- [ ] Validate all CLI commands work correctly
- [ ] Check that Table 1 replication still passes
- [ ] Measure reduction in codebase size and complexity

## Success Metrics

### Quantitative Goals
- [ ] Reduce duplication patterns from 394 to <100
- [ ] Decrease codebase size by >15% (target: <17,500 lines)
- [ ] Maintain 100% test pass rate
- [ ] Zero regression in core functionality

### Qualitative Goals
- [ ] Improved code maintainability
- [ ] Clearer separation of concerns
- [ ] Enhanced extensibility for future features
- [ ] Better developer experience

## Dependencies and Coordination

### Required Reviews
- [ ] **Architecture Review**: BaseResultAnalyzer design 
- [ ] **Testing Review**: Validation strategy for critical changes
- [ ] **Performance Review**: Ensure no performance regression

### Coordination Points
- [ ] Notify team before starting Phase 1 changes
- [ ] Daily check-ins during critical infrastructure changes
- [ ] Code review required for all high-risk changes

## Rollback Plan

### Emergency Rollback
- [ ] Maintain feature branches for each major change
- [ ] Keep original implementations until validation complete
- [ ] Document rollback procedures for each phase

### Partial Rollback
- [ ] Design changes to be independently reversible
- [ ] Maintain compatibility shims during transition
- [ ] Test rollback procedures before starting changes

---

**Next Steps**: 
1. Get team approval for this deduplication plan
2. Create feature branch for Phase 1 work
3. Begin with BaseResultAnalyzer implementation
4. Schedule regular progress reviews

**Contact**: Submit feedback and questions for T-013 deduplication plan