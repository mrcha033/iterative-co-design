# Code Duplication Audit Report - T-013
Generated: 2025-07-12 16:56:11

## Executive Summary

- **Total Python files analyzed**: 64
- **Import duplication patterns**: 38
- **Function duplication patterns**: 57
- **Class duplication patterns**: 14
- **Code block patterns**: 50
- **Repeated constants/literals**: 235

**Duplication Severity**: 🔴 HIGH (394 total patterns)

## Imports Duplication

### import os
**Occurrences**: 8

- `audit_code_duplication.py`
- `scripts/run_experiment.py`
- `src/co_design/correlation.py`
- `src/models/manager.py`
- `src/profiler/ncu.py`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import sys
**Occurrences**: 13

- `audit_code_duplication.py`
- `check_syntax.py`
- `main.py`
- `test_basic_functionality.py`
- `validate_implementation.py`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from pathlib import Path
**Occurrences**: 36

- `audit_code_duplication.py`
- `check_syntax.py`
- `main.py`
- `validate_implementation.py`
- `scripts/check_coverage.py`
- ... and 31 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from typing import Dict
**Occurrences**: 21

- `audit_code_duplication.py`
- `main.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/performance_report.py`
- ... and 16 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from typing import List
**Occurrences**: 18

- `audit_code_duplication.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/performance_report.py`
- `scripts/run_experiment.py`
- ... and 13 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from typing import Tuple
**Occurrences**: 14

- `audit_code_duplication.py`
- `scripts/generate_figures.py`
- `scripts/run_experiment.py`
- `src/co_design/correlation.py`
- `src/co_design/hds.py`
- ... and 9 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from typing import Any
**Occurrences**: 16

- `audit_code_duplication.py`
- `main.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/run_experiment.py`
- ... and 11 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import json
**Occurrences**: 17

- `audit_code_duplication.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/implementation_demo.py`
- `scripts/performance_report.py`
- ... and 12 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import argparse
**Occurrences**: 8

- `main.py`
- `scripts/check_coverage.py`
- `scripts/generate_correlation_matrix.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import logging
**Occurrences**: 9

- `main.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/run_experiment.py`
- `src/profiler/calibration.py`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from src.utils.config import load_config
**Occurrences**: 5

- `main.py`
- `scripts/generate_correlation_matrix.py`
- `scripts/quick_start_demo.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/unit/test_config.py`

**Suggested Action**: [TO BE DETERMINED]

### from src.utils.exceptions import IterativeCoDesignError
**Occurrences**: 5

- `main.py`
- `scripts/run_experiment.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/unit/test_correlation.py`
- `tests/unit/test_spectral.py`

**Suggested Action**: [TO BE DETERMINED]

### import torch
**Occurrences**: 39

- `test_basic_functionality.py`
- `scripts/run_experiment.py`
- `scripts/test_framework.py`
- `src/co_design/apply.py`
- `src/co_design/correlation.py`
- ... and 34 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import numpy as np
**Occurrences**: 30

- `test_basic_functionality.py`
- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/performance_report.py`
- `scripts/run_experiment.py`
- ... and 25 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from src.models.permutable_model import PermutableModel
**Occurrences**: 10

- `test_basic_functionality.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/integration/test_hds_ptq_integration.py`
- `tests/integration/test_iasp_integration.py`
- `tests/integration/test_permutation_integration.py`
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import torch.nn as nn
**Occurrences**: 30

- `test_basic_functionality.py`
- `scripts/run_experiment.py`
- `src/co_design/apply.py`
- `src/co_design/correlation.py`
- `src/co_design/hds.py`
- ... and 25 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import yaml
**Occurrences**: 6

- `validate_implementation.py`
- `scripts/implementation_demo.py`
- `scripts/quick_start_demo.py`
- `src/utils/config.py`
- `tests/integration/test_experiment_pipeline.py`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import subprocess
**Occurrences**: 5

- `scripts/check_coverage.py`
- `src/profiler/ncu.py`
- `src/utils/profiler.py`
- `tests/integration/test_cli_correlation.py`
- `tests/integration/test_cli_full_pipeline.py`

**Suggested Action**: [TO BE DETERMINED]

### from src.models.manager import ModelManager
**Occurrences**: 6

- `scripts/generate_correlation_matrix.py`
- `scripts/run_experiment.py`
- `scripts/test_framework.py`
- `tests/integration/test_model_integration.py`
- `tests/integration/test_permutation_integration.py`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from src.utils.dataset_manager import DatasetManager
**Occurrences**: 5

- `scripts/generate_correlation_matrix.py`
- `scripts/run_experiment.py`
- `scripts/test_framework.py`
- `tests/integration/test_model_integration.py`
- `tests/unit/test_dataset_manager.py`

**Suggested Action**: [TO BE DETERMINED]

### from src.co_design.correlation import CorrelationMatrixComputer
**Occurrences**: 4

- `scripts/generate_correlation_matrix.py`
- `tests/integration/test_iasp_integration.py`
- `tests/integration/test_permutation_integration.py`
- `tests/unit/test_correlation.py`

**Suggested Action**: [TO BE DETERMINED]

### from typing import Optional
**Occurrences**: 21

- `scripts/generate_figures.py`
- `scripts/generate_tables.py`
- `scripts/performance_report.py`
- `scripts/run_experiment.py`
- `src/co_design/apply.py`
- ... and 16 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import tempfile
**Occurrences**: 18

- `scripts/implementation_demo.py`
- `scripts/quick_start_demo.py`
- `scripts/test_framework.py`
- `src/profiler/ncu.py`
- `src/utils/profiler.py`
- ... and 13 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import time
**Occurrences**: 6

- `scripts/run_experiment.py`
- `src/profiler/calibration.py`
- `src/profiler/latency.py`
- `src/profiler/ncu.py`
- `src/utils/profiler.py`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### import warnings
**Occurrences**: 14

- `scripts/run_experiment.py`
- `src/co_design/apply.py`
- `src/co_design/correlation.py`
- `src/co_design/hds.py`
- `src/co_design/iasp.py`
- ... and 9 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from tqdm import tqdm
**Occurrences**: 4

- `scripts/run_experiment.py`
- `src/co_design/correlation.py`
- `src/co_design/hds.py`
- `src/co_design/ptq.py`

**Suggested Action**: [TO BE DETERMINED]

### from src.co_design.iasp import IASPPermutationOptimizer
**Occurrences**: 6

- `scripts/run_experiment.py`
- `tests/integration/test_iasp_integration.py`
- `tests/integration/test_permutation_integration.py`
- `tests/integration/test_real_models.py`
- `tests/performance/test_benchmarks.py`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from typing import Union
**Occurrences**: 9

- `src/co_design/apply.py`
- `src/co_design/hds.py`
- `src/co_design/ptq.py`
- `src/models/manager.py`
- `src/models/permutable_model.py`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from models.permutable_model import PermutableModel
**Occurrences**: 5

- `src/co_design/apply.py`
- `src/co_design/correlation.py`
- `src/co_design/hds.py`
- `src/co_design/iasp.py`
- `src/co_design/ptq.py`

**Suggested Action**: [TO BE DETERMINED]

### from utils.exceptions import IterativeCoDesignError
**Occurrences**: 6

- `src/co_design/apply.py`
- `src/co_design/correlation.py`
- `src/co_design/hds.py`
- `src/co_design/iasp.py`
- `src/co_design/ptq.py`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from dataclasses import dataclass
**Occurrences**: 5

- `src/co_design/hds.py`
- `src/co_design/ptq.py`
- `src/profiler/calibration.py`
- `src/profiler/latency.py`
- `src/profiler/ncu.py`

**Suggested Action**: [TO BE DETERMINED]

### import torch.nn.functional as F
**Occurrences**: 4

- `src/co_design/hds.py`
- `src/co_design/ptq.py`
- `src/models/gcn_model.py`
- `tests/integration/test_real_models.py`

**Suggested Action**: [TO BE DETERMINED]

### from torch.utils.data import DataLoader
**Occurrences**: 5

- `src/utils/dataset_manager.py`
- `tests/integration/test_cli_full_pipeline.py`
- `tests/integration/test_real_models.py`
- `tests/integration/test_real_models.py`
- `tests/unit/test_dataset_manager.py`

**Suggested Action**: [TO BE DETERMINED]

### import shutil
**Occurrences**: 4

- `src/utils/dataset_manager.py`
- `src/utils/dataset_manager.py`
- `tests/integration/test_cli_full_pipeline.py`
- `tests/integration/test_experiment_pipeline.py`

**Suggested Action**: [TO BE DETERMINED]

### import pytest
**Occurrences**: 22

- `tests/integration/test_cli_correlation.py`
- `tests/integration/test_cli_full_pipeline.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/integration/test_hds_ptq_integration.py`
- `tests/integration/test_iasp_integration.py`
- ... and 17 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from unittest.mock import patch
**Occurrences**: 17

- `tests/integration/test_cli_full_pipeline.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/integration/test_hds_ptq_integration.py`
- `tests/integration/test_profiler_integration.py`
- `tests/integration/test_real_models.py`
- ... and 12 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from unittest.mock import Mock
**Occurrences**: 16

- `tests/integration/test_cli_full_pipeline.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/integration/test_hds_ptq_integration.py`
- `tests/integration/test_iasp_integration.py`
- `tests/integration/test_profiler_integration.py`
- ... and 11 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### from unittest.mock import MagicMock
**Occurrences**: 5

- `tests/integration/test_experiment_pipeline.py`
- `tests/unit/test_dataset_manager.py`
- `tests/unit/test_profiler_calibration.py`
- `tests/unit/test_profiler_ncu.py`
- `tests/unit/test_spectral.py`

**Suggested Action**: [TO BE DETERMINED]

## Functions Duplication

### signature: main()
**Occurrences**: 13

- `audit_code_duplication.py:481`
- `check_syntax.py:25`
- `main.py:242`
- `test_basic_functionality.py:153`
- `validate_implementation.py:60`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: __init__(self)
**Occurrences**: 27

- `test_basic_functionality.py:56`
- `test_basic_functionality.py:107`
- `src/profiler/calibration.py:219`
- `tests/integration/test_experiment_pipeline.py:115`
- `tests/integration/test_hds_ptq_integration.py:22`
- ... and 22 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: forward(self, x)
**Occurrences**: 28

- `test_basic_functionality.py:60`
- `test_basic_functionality.py:112`
- `src/profiler/calibration.py:229`
- `tests/integration/test_cli_full_pipeline.py:34`
- `tests/integration/test_cli_full_pipeline.py:66`
- ... and 23 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: __init__(self, results_dir, output_dir)
**Occurrences**: 2

- `scripts/generate_figures.py:62`
- `scripts/generate_tables.py:24`

**Suggested Action**: [TO BE DETERMINED]

### signature: _load_results(self)
**Occurrences**: 2

- `scripts/generate_figures.py:77`
- `scripts/generate_tables.py:39`

**Suggested Action**: [TO BE DETERMINED]

### signature: _find_experiment(self, model, strategy)
**Occurrences**: 2

- `scripts/generate_figures.py:481`
- `scripts/generate_tables.py:263`

**Suggested Action**: [TO BE DETERMINED]

### signature: _extract_latency(self, exp_data)
**Occurrences**: 2

- `scripts/generate_figures.py:500`
- `scripts/generate_tables.py:282`

**Suggested Action**: [TO BE DETERMINED]

### signature: __getattr__(self, name)
**Occurrences**: 2

- `scripts/quick_start_demo.py:160`
- `tests/integration/test_experiment_pipeline.py:467`

**Suggested Action**: [TO BE DETERMINED]

### signature: __init__(self, config)
**Occurrences**: 8

- `scripts/run_experiment.py:35`
- `src/co_design/hds.py:342`
- `src/co_design/ptq.py:278`
- `src/profiler/calibration.py:91`
- `src/profiler/latency.py:79`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: _get_gpu_info(self)
**Occurrences**: 2

- `scripts/run_experiment.py:152`
- `src/profiler/calibration.py:189`

**Suggested Action**: [TO BE DETERMINED]

### signature: get_applied_permutations(self)
**Occurrences**: 2

- `src/co_design/apply.py:454`
- `src/models/permutable_model.py:274`

**Suggested Action**: [TO BE DETERMINED]

### signature: has_permutation(self, layer_name)
**Occurrences**: 2

- `src/co_design/apply.py:458`
- `src/models/permutable_model.py:278`

**Suggested Action**: [TO BE DETERMINED]

### signature: get_cache_info(self)
**Occurrences**: 2

- `src/co_design/correlation.py:355`
- `src/models/manager.py:268`

**Suggested Action**: [TO BE DETERMINED]

### signature: forward(self, training)
**Occurrences**: 2

- `src/co_design/hds.py:92`
- `src/co_design/hds.py:222`

**Suggested Action**: [TO BE DETERMINED]

### signature: update_temperature(self, new_temperature)
**Occurrences**: 3

- `src/co_design/hds.py:169`
- `src/co_design/hds.py:253`
- `src/co_design/hds.py:328`

**Suggested Action**: [TO BE DETERMINED]

### signature: forward(self)
**Occurrences**: 2

- `src/co_design/hds.py:306`
- `src/models/permutable_model.py:34`

**Suggested Action**: [TO BE DETERMINED]

### signature: prepare_model(self, model)
**Occurrences**: 2

- `src/co_design/hds.py:359`
- `src/co_design/ptq.py:292`

**Suggested Action**: [TO BE DETERMINED]

### signature: _get_target_layers(self, model)
**Occurrences**: 2

- `src/co_design/hds.py:394`
- `src/co_design/ptq.py:346`

**Suggested Action**: [TO BE DETERMINED]

### signature: _replace_layer(self, model, layer_name, new_layer)
**Occurrences**: 2

- `src/co_design/hds.py:410`
- `src/co_design/ptq.py:467`

**Suggested Action**: [TO BE DETERMINED]

### signature: validate_permutation(self, permutation)
**Occurrences**: 2

- `src/co_design/iasp.py:394`
- `src/co_design/spectral.py:382`

**Suggested Action**: [TO BE DETERMINED]

### signature: validate(self)
**Occurrences**: 2

- `src/profiler/calibration.py:37`
- `src/profiler/ncu.py:47`

**Suggested Action**: [TO BE DETERMINED]

### signature: to_dict(self)
**Occurrences**: 4

- `src/profiler/calibration.py:67`
- `src/profiler/latency.py:52`
- `src/profiler/ncu.py:75`
- `src/utils/config.py:268`

**Suggested Action**: [TO BE DETERMINED]

### signature: _load_baseline(self)
**Occurrences**: 2

- `src/profiler/calibration.py:110`
- `tests/performance/test_benchmarks.py:881`

**Suggested Action**: [TO BE DETERMINED]

### signature: model_forward()
**Occurrences**: 3

- `src/profiler/calibration.py:268`
- `src/profiler/latency.py:400`
- `src/utils/profiler.py:435`

**Suggested Action**: [TO BE DETERMINED]

### signature: _create_profile_script(self, model, inputs)
**Occurrences**: 2

- `src/profiler/ncu.py:186`
- `src/utils/profiler.py:161`

**Suggested Action**: [TO BE DETERMINED]

### signature: load_config(config_path)
**Occurrences**: 2

- `src/utils/config.py:283`
- `src/utils/config.py:316`

**Suggested Action**: [TO BE DETERMINED]

### signature: __len__(self)
**Occurrences**: 2

- `src/utils/graph_dataset.py:47`
- `src/utils/text_dataset.py:74`

**Suggested Action**: [TO BE DETERMINED]

### signature: __getitem__(self, idx)
**Occurrences**: 2

- `src/utils/graph_dataset.py:51`
- `src/utils/text_dataset.py:80`

**Suggested Action**: [TO BE DETERMINED]

### signature: setup_method(self)
**Occurrences**: 37

- `tests/integration/test_cli_correlation.py:14`
- `tests/integration/test_cli_full_pipeline.py:83`
- `tests/integration/test_cli_full_pipeline.py:426`
- `tests/integration/test_cli_full_pipeline.py:486`
- `tests/integration/test_experiment_pipeline.py:28`
- ... and 32 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: teardown_method(self)
**Occurrences**: 4

- `tests/integration/test_cli_full_pipeline.py:144`
- `tests/integration/test_cli_full_pipeline.py:430`
- `tests/integration/test_cli_full_pipeline.py:495`
- `tests/integration/test_experiment_pipeline.py:108`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_permutation_application(self)
**Occurrences**: 2

- `tests/integration/test_iasp_integration.py:167`
- `tests/unit/test_model_manager.py:175`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_error_handling(self)
**Occurrences**: 2

- `tests/integration/test_iasp_integration.py:321`
- `tests/integration/test_model_integration.py:162`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_model_summary(self)
**Occurrences**: 2

- `tests/integration/test_model_integration.py:144`
- `tests/unit/test_model_manager.py:203`

**Suggested Action**: [TO BE DETERMINED]

### signature: __init__(self, input_size)
**Occurrences**: 2

- `tests/integration/test_profiler_integration.py:47`
- `tests/integration/test_real_models.py:443`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_init(self)
**Occurrences**: 2

- `tests/unit/test_apply.py:44`
- `tests/unit/test_spectral.py:36`

**Suggested Action**: [TO BE DETERMINED]

### signature: mock_get_layer(name)
**Occurrences**: 2

- `tests/unit/test_apply.py:298`
- `tests/unit/test_apply.py:319`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_manager_initialization(self)
**Occurrences**: 2

- `tests/unit/test_config.py:25`
- `tests/unit/test_dataset_manager.py:21`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_config_validation(self)
**Occurrences**: 5

- `tests/unit/test_config.py:40`
- `tests/unit/test_iasp.py:45`
- `tests/unit/test_profiler_calibration.py:50`
- `tests/unit/test_profiler_latency.py:119`
- `tests/unit/test_profiler_ncu.py:47`

**Suggested Action**: [TO BE DETERMINED]

### signature: __init__(self, input_dim, hidden_dim, output_dim)
**Occurrences**: 2

- `tests/unit/test_correlation.py:19`
- `tests/unit/test_model_manager.py:17`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_initialization(self)
**Occurrences**: 10

- `tests/unit/test_correlation.py:51`
- `tests/unit/test_hds.py:49`
- `tests/unit/test_hds.py:105`
- `tests/unit/test_hds.py:155`
- `tests/unit/test_hds.py:213`
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: test_default_config(self)
**Occurrences**: 6

- `tests/unit/test_hds.py:20`
- `tests/unit/test_iasp.py:21`
- `tests/unit/test_profiler_calibration.py:23`
- `tests/unit/test_profiler_latency.py:22`
- `tests/unit/test_profiler_ncu.py:24`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: test_custom_config(self)
**Occurrences**: 6

- `tests/unit/test_hds.py:31`
- `tests/unit/test_iasp.py:31`
- `tests/unit/test_profiler_calibration.py:34`
- `tests/unit/test_profiler_latency.py:32`
- `tests/unit/test_profiler_ncu.py:34`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: test_temperature_update(self)
**Occurrences**: 3

- `tests/unit/test_hds.py:90`
- `tests/unit/test_hds.py:182`
- `tests/unit/test_hds.py:248`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_forward(self)
**Occurrences**: 3

- `tests/unit/test_hds.py:117`
- `tests/unit/test_hds.py:163`
- `tests/unit/test_ptq.py:139`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_get_target_layers(self)
**Occurrences**: 2

- `tests/unit/test_hds.py:219`
- `tests/unit/test_ptq.py:212`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_prepare_model(self)
**Occurrences**: 2

- `tests/unit/test_hds.py:227`
- `tests/unit/test_ptq.py:220`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_results_creation(self)
**Occurrences**: 4

- `tests/unit/test_iasp.py:71`
- `tests/unit/test_profiler_calibration.py:73`
- `tests/unit/test_profiler_latency.py:50`
- `tests/unit/test_profiler_ncu.py:70`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_to_dict(self)
**Occurrences**: 4

- `tests/unit/test_iasp.py:87`
- `tests/unit/test_profiler_calibration.py:90`
- `tests/unit/test_profiler_latency.py:73`
- `tests/unit/test_profiler_ncu.py:84`

**Suggested Action**: [TO BE DETERMINED]

### signature: test_profiler_initialization(self)
**Occurrences**: 2

- `tests/unit/test_profiler_latency.py:113`
- `tests/unit/test_profiler_ncu.py:110`

**Suggested Action**: [TO BE DETERMINED]

### identical_body: _load_results
**Occurrences**: 2

- `scripts/generate_figures.py:77`
  ```python
  def _load_results(self) -> Dict[str, Any]:
        """Load all experiment results from JSON files."""
        results = {}
        
        for result_file in self.results_dir.glob('**/results.json'):...
  ```
- `scripts/generate_tables.py:39`
  ```python
  def _load_results(self) -> Dict[str, Any]:
        """Load all experiment results from JSON files."""
        results = {}
        
        for result_file in self.results_dir.glob('**/results.json'):...
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: _find_experiment
**Occurrences**: 2

- `scripts/generate_figures.py:481`
  ```python
  def _find_experiment(self, model: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Find experiment result matching model and strategy."""
        for exp_id, exp_data in self.results.items(...
  ```
- `scripts/generate_tables.py:263`
  ```python
  def _find_experiment(self, model: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Find experiment result matching model and strategy."""
        for exp_id, exp_data in self.results.items(...
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: __getattr__
**Occurrences**: 2

- `scripts/quick_start_demo.py:160`
  ```python
  def __getattr__(self, name):
                return None
  ```
- `tests/integration/test_experiment_pipeline.py:467`
  ```python
  def __getattr__(self, name):
                return None
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: _replace_layer
**Occurrences**: 2

- `src/co_design/hds.py:410`
  ```python
  def _replace_layer(self, model: PermutableModel, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model."""
        # Navigate to the parent module
        parts = layer_name....
  ```
- `src/co_design/ptq.py:467`
  ```python
  def _replace_layer(self, model: PermutableModel, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model."""
        # Navigate to the parent module
        parts = layer_name....
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: teardown_method
**Occurrences**: 3

- `tests/integration/test_cli_full_pipeline.py:144`
  ```python
  def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
  ```
- `tests/integration/test_cli_full_pipeline.py:430`
  ```python
  def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
  ```
- `tests/integration/test_cli_full_pipeline.py:495`
  ```python
  def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: forward
**Occurrences**: 2

- `tests/unit/test_iasp.py:146`
  ```python
  def forward(self, x):
                return self.mixer.in_proj(x)
  ```
- `tests/unit/test_iasp.py:318`
  ```python
  def forward(self, x):
                return self.mixer.in_proj(x)
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: __init__
**Occurrences**: 2

- `tests/unit/test_iasp.py:367`
  ```python
  def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 6)
  ```
- `tests/unit/test_iasp.py:445`
  ```python
  def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 6)
  ```

**Suggested Action**: [TO BE DETERMINED]

### identical_body: setup_method
**Occurrences**: 2

- `tests/unit/test_dataset_manager.py:17`
  ```python
  def setup_method(self):
        """Setup test fixtures."""
        self.dataset_manager = DatasetManager()
  ```
- `tests/unit/test_dataset_manager.py:257`
  ```python
  def setup_method(self):
        """Setup test fixtures."""
        self.dataset_manager = DatasetManager()
  ```

**Suggested Action**: [TO BE DETERMINED]

## Classes Duplication

### signature: SimpleModel(nn.Module)
**Occurrences**: 10

- `test_basic_functionality.py:55`
- `tests/integration/test_hds_ptq_integration.py:21`
- `tests/integration/test_iasp_integration.py:17`
- `tests/integration/test_permutation_integration.py:45`
- `tests/unit/test_hds.py:201`
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: TestModel(nn.Module)
**Occurrences**: 8

- `test_basic_functionality.py:106`
- `tests/unit/test_iasp.py:159`
- `tests/unit/test_iasp.py:175`
- `tests/unit/test_iasp.py:311`
- `tests/unit/test_iasp.py:366`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: MockArgs()
**Occurrences**: 2

- `scripts/quick_start_demo.py:153`
- `tests/integration/test_experiment_pipeline.py:459`

**Suggested Action**: [TO BE DETERMINED]

### signature: MockModel(nn.Module)
**Occurrences**: 3

- `tests/integration/test_experiment_pipeline.py:114`
- `tests/unit/test_correlation.py:17`
- `tests/unit/test_model_manager.py:15`

**Suggested Action**: [TO BE DETERMINED]

### signature: ToyModel(nn.Module)
**Occurrences**: 2

- `tests/integration/test_profiler_integration.py:27`
- `tests/unit/test_iasp.py:139`

**Suggested Action**: [TO BE DETERMINED]

### signature: TestUtilityFunctions()
**Occurrences**: 6

- `tests/unit/test_hds.py:262`
- `tests/unit/test_iasp.py:348`
- `tests/unit/test_profiler_calibration.py:373`
- `tests/unit/test_profiler_latency.py:237`
- `tests/unit/test_profiler_ncu.py:314`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### signature: TestIntegration()
**Occurrences**: 2

- `tests/unit/test_hds.py:330`
- `tests/unit/test_ptq.py:356`

**Suggested Action**: [TO BE DETERMINED]

### signature: TestErrorHandling()
**Occurrences**: 3

- `tests/unit/test_iasp.py:421`
- `tests/unit/test_profiler_latency.py:355`
- `tests/unit/test_profiler_ncu.py:366`

**Suggested Action**: [TO BE DETERMINED]

### method_pattern: SimpleModel: __init__, forward
**Occurrences**: 10

- `test_basic_functionality.py:55`
- `tests/integration/test_hds_ptq_integration.py:21`
- `tests/integration/test_iasp_integration.py:17`
- `tests/integration/test_permutation_integration.py:45`
- `tests/unit/test_hds.py:201`
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### method_pattern: TestModel: __init__, forward
**Occurrences**: 2

- `test_basic_functionality.py:106`
- `tests/unit/test_iasp.py:311`

**Suggested Action**: [TO BE DETERMINED]

### method_pattern: MockArgs: __getattr__
**Occurrences**: 2

- `scripts/quick_start_demo.py:153`
- `tests/integration/test_experiment_pipeline.py:459`

**Suggested Action**: [TO BE DETERMINED]

### method_pattern: MockModel: __init__, forward
**Occurrences**: 3

- `tests/integration/test_experiment_pipeline.py:114`
- `tests/unit/test_correlation.py:17`
- `tests/unit/test_model_manager.py:15`

**Suggested Action**: [TO BE DETERMINED]

### method_pattern: ToyModel: __init__, forward
**Occurrences**: 2

- `tests/integration/test_profiler_integration.py:27`
- `tests/unit/test_iasp.py:139`

**Suggested Action**: [TO BE DETERMINED]

### method_pattern: TestModel: __init__
**Occurrences**: 6

- `tests/unit/test_iasp.py:159`
- `tests/unit/test_iasp.py:175`
- `tests/unit/test_iasp.py:366`
- `tests/unit/test_iasp.py:398`
- `tests/unit/test_iasp.py:429`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

## Patterns Duplication

### pattern_logger\.|print\(.*\)...
**Occurrences**: 10

- `audit_code_duplication.py:37`
  ```python
  print("📁 Scanning Python files...")
  ```
- `audit_code_duplication.py:50`
  ```python
  print(f"  Found {len(self.python_files)} Python files")
  ```
- `audit_code_duplication.py:55`
  ```python
  print("\n📦 Analyzing import statements...")
  ```
- `audit_code_duplication.py:83`
  ```python
  print(f"  ⚠️  Error parsing {file_path}: {e}")
  ```
- `audit_code_duplication.py:90`
  ```python
  print(f"  Found {len(self.duplicates['imports'])} common import patterns")
  ```
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### pattern_with open\(.*\) as...
**Occurrences**: 10

- `audit_code_duplication.py:61`
  ```python
  with open(file_path, 'r', encoding='utf-8') as f:
  ```
- `audit_code_duplication.py:101`
  ```python
  with open(file_path, 'r', encoding='utf-8') as f:
  ```
- `audit_code_duplication.py:153`
  ```python
  with open(file_path, 'r', encoding='utf-8') as f:
  ```
- `audit_code_duplication.py:228`
  ```python
  with open(file_path, 'r', encoding='utf-8') as f:
  ```
- `audit_code_duplication.py:266`
  ```python
  with open(file_path, 'r', encoding='utf-8') as f:
  ```
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### pattern_assert.*,.*|if.*is N...
**Occurrences**: 10

- `audit_code_duplication.py:221`
  ```python
  r"assert.*,.*|if.*is None:",
  ```
- `test_basic_functionality.py:36`
  ```python
  assert len(permutation) == 4, f"Expected length 4, got {len(permutation)}"
  ```
- `test_basic_functionality.py:37`
  ```python
  assert set(permutation) == {0, 1, 2, 3}, f"Invalid permutation: {permutation}"
  ```
- `test_basic_functionality.py:38`
  ```python
  assert 'num_clusters_used' in info, "Missing num_clusters_used in info"
  ```
- `test_basic_functionality.py:39`
  ```python
  assert 'silhouette_score' in info, "Missing silhouette_score in info"
  ```
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### pattern_\.cuda\(\)|\.to\(dev...
**Occurrences**: 10

- `scripts/run_experiment.py:84`
  ```python
  torch.cuda.manual_seed_all(seed)
  ```
- `scripts/run_experiment.py:154`
  ```python
  if not torch.cuda.is_available():
  ```
- `scripts/run_experiment.py:160`
  ```python
  'device_count': torch.cuda.device_count(),
  ```
- `scripts/run_experiment.py:162`
  ```python
  'device_name': torch.cuda.get_device_name(gpu_id),
  ```
- `scripts/run_experiment.py:163`
  ```python
  'device_capability': torch.cuda.get_device_capability(gpu_id),
  ```
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### pattern_time\.time\(\)|torch...
**Occurrences**: 10

- `scripts/run_experiment.py:193`
  ```python
  start_time = time.time()
  ```
- `scripts/run_experiment.py:232`
  ```python
  self.results['duration_seconds'] = time.time() - start_time
  ```
- `scripts/run_experiment.py:237`
  ```python
  self.results['duration_seconds'] = time.time() - start_time
  ```
- `src/profiler/calibration.py:271`
  ```python
  start_time = time.time()
  ```
- `src/profiler/calibration.py:273`
  ```python
  calibration_time = time.time() - start_time
  ```
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### pattern_torch\.load|model\.l...
**Occurrences**: 10

- `scripts/run_experiment.py:378`
  ```python
  correlation_matrix = torch.load(correlation_path)
  ```
- `src/co_design/correlation.py:76`
  ```python
  return torch.load(cache_path, map_location='cpu')
  ```
- `src/co_design/correlation.py:347`
  ```python
  correlation_matrix = torch.load(matrix_path, map_location='cpu')
  ```
- `src/co_design/hds.py:673`
  ```python
  checkpoint = torch.load(filepath)
  ```
- `src/models/manager.py:170`
  ```python
  checkpoint = torch.load(path, map_location='cpu')
  ```
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: __init__
**Occurrences**: 3

- `tests/unit/test_iasp.py:140`
- `tests/unit/test_iasp.py:160`
- `tests/unit/test_iasp.py:312`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: forward
**Occurrences**: 2

- `tests/unit/test_iasp.py:146`
- `tests/unit/test_iasp.py:318`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _load_results
**Occurrences**: 2

- `scripts/generate_figures.py:77`
- `scripts/generate_tables.py:39`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _find_experiment
**Occurrences**: 2

- `scripts/generate_figures.py:481`
- `scripts/generate_tables.py:263`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _extract_latency
**Occurrences**: 2

- `scripts/generate_figures.py:500`
- `scripts/generate_tables.py:282`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: show_supported_strategies
**Occurrences**: 7

- `scripts/implementation_demo.py:36`
- `scripts/implementation_demo.py:56`
- `scripts/implementation_demo.py:76`
- `scripts/implementation_demo.py:101`
- `scripts/implementation_demo.py:161`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: show_usage_examples
**Occurrences**: 2

- `scripts/implementation_demo.py:128`
- `scripts/quick_start_demo.py:225`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: create_demo_config
**Occurrences**: 14

- `scripts/quick_start_demo.py:21`
- `scripts/quick_start_demo.py:160`
- `src/co_design/apply.py:296`
- `src/co_design/apply.py:458`
- `src/co_design/hds.py:654`
- ... and 9 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _run_baseline_benchmark
**Occurrences**: 2

- `scripts/run_experiment.py:301`
- `scripts/run_experiment.py:594`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _identify_mamba_affected_layers
**Occurrences**: 2

- `src/co_design/apply.py:232`
- `src/co_design/apply.py:260`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _apply_linear_permutation
**Occurrences**: 2

- `src/models/permutable_model.py:224`
- `src/models/permutable_model.py:241`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: get_applied_permutations
**Occurrences**: 2

- `src/co_design/apply.py:454`
- `src/models/permutable_model.py:274`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: reset_permutations
**Occurrences**: 2

- `src/co_design/apply.py:462`
- `src/models/manager.py:264`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: update_temperature
**Occurrences**: 2

- `src/co_design/hds.py:328`
- `src/co_design/hds.py:617`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _get_target_layers
**Occurrences**: 2

- `src/co_design/hds.py:394`
- `src/co_design/ptq.py:346`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: _replace_layer
**Occurrences**: 2

- `src/co_design/hds.py:410`
- `src/co_design/ptq.py:467`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: is_model_supported
**Occurrences**: 2

- `src/models/manager.py:56`
- `src/utils/dataset_manager.py:52`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: list_supported_models
**Occurrences**: 2

- `src/models/manager.py:60`
- `src/utils/dataset_manager.py:56`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: validate_model_name
**Occurrences**: 7

- `src/utils/config.py:18`
- `src/utils/config.py:42`
- `src/utils/config.py:76`
- `src/utils/config.py:125`
- `src/utils/config.py:141`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: validate_precision
**Occurrences**: 11

- `src/utils/config.py:25`
- `src/utils/config.py:49`
- `src/utils/config.py:64`
- `src/utils/config.py:70`
- `src/utils/config.py:92`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_cli_help
**Occurrences**: 2

- `tests/integration/test_cli_correlation.py:19`
- `tests/integration/test_cli_correlation.py:31`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: teardown_method
**Occurrences**: 4

- `tests/integration/test_cli_full_pipeline.py:144`
- `tests/integration/test_cli_full_pipeline.py:430`
- `tests/integration/test_cli_full_pipeline.py:495`
- `tests/integration/test_experiment_pipeline.py:108`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_permute_only_strategy
**Occurrences**: 5

- `tests/integration/test_cli_full_pipeline.py:195`
- `tests/integration/test_cli_full_pipeline.py:214`
- `tests/integration/test_cli_full_pipeline.py:232`
- `tests/integration/test_cli_full_pipeline.py:295`
- `tests/integration/test_cli_full_pipeline.py:313`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: setup_method
**Occurrences**: 2

- `tests/unit/test_dataset_manager.py:17`
- `tests/unit/test_dataset_manager.py:257`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_linear_layer_input_permutation
**Occurrences**: 2

- `tests/integration/test_real_models.py:601`
- `tests/integration/test_real_models.py:621`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: linear_op
**Occurrences**: 2

- `tests/performance/test_benchmarks.py:360`
- `tests/performance/test_benchmarks.py:520`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: spectral_clustering_run
**Occurrences**: 4

- `tests/performance/test_benchmarks.py:467`
- `tests/performance/test_benchmarks.py:561`
- `tests/performance/test_benchmarks.py:569`
- `tests/performance/test_benchmarks.py:311`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_init
**Occurrences**: 3

- `tests/unit/test_apply.py:44`
- `tests/unit/test_hds.py:213`
- `tests/unit/test_model_manager.py:123`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_validate_permutation_valid
**Occurrences**: 4

- `tests/unit/test_apply.py:49`
- `tests/unit/test_apply.py:63`
- `tests/unit/test_apply.py:75`
- `tests/unit/test_apply.py:87`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_apply_permutation_success
**Occurrences**: 2

- `tests/unit/test_apply.py:335`
- `tests/unit/test_apply.py:353`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_load_default_config
**Occurrences**: 2

- `tests/unit/test_config.py:30`
- `tests/unit/test_config.py:247`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_config_validation
**Occurrences**: 2

- `tests/unit/test_config.py:40`
- `tests/unit/test_config.py:224`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_config_validation_missing_sections
**Occurrences**: 3

- `tests/unit/test_config.py:66`
- `tests/unit/test_config.py:390`
- `tests/unit/test_config.py:395`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_merge_configs
**Occurrences**: 3

- `tests/unit/test_config.py:105`
- `tests/unit/test_config.py:235`
- `tests/unit/test_config.py:416`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_load_config_yaml
**Occurrences**: 2

- `tests/unit/test_config.py:170`
- `tests/unit/test_config.py:188`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_model_validation
**Occurrences**: 4

- `tests/unit/test_config.py:279`
- `tests/unit/test_config.py:292`
- `tests/unit/test_config.py:309`
- `tests/unit/test_config.py:322`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_environment_model_override
**Occurrences**: 2

- `tests/unit/test_config.py:366`
- `tests/unit/test_config.py:377`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_unsupported_dataset
**Occurrences**: 2

- `tests/unit/test_dataset_manager.py:99`
- `tests/unit/test_dataset_manager.py:121`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_default_config
**Occurrences**: 2

- `tests/unit/test_ptq.py:20`
- `tests/unit/test_ptq.py:31`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_get_target_layers
**Occurrences**: 2

- `tests/unit/test_hds.py:219`
- `tests/unit/test_ptq.py:212`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_prepare_model
**Occurrences**: 2

- `tests/unit/test_hds.py:227`
- `tests/unit/test_ptq.py:220`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_device_detection_cpu
**Occurrences**: 2

- `tests/unit/test_profiler_calibration.py:128`
- `tests/unit/test_profiler_calibration.py:134`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: dummy_function
**Occurrences**: 2

- `tests/unit/test_profiler_latency.py:171`
- `tests/unit/test_profiler_latency.py:211`

**Suggested Action**: [TO BE DETERMINED]

### similar_structure: test_compute_permutation_simple
**Occurrences**: 2

- `tests/unit/test_spectral.py:271`
- `tests/unit/test_spectral.py:348`

**Suggested Action**: [TO BE DETERMINED]

## Constants Duplication

### string:   ⚠️  Error parsing ...
**Occurrences**: 3

- `audit_code_duplication.py:83`
- `audit_code_duplication.py:129`
- `audit_code_duplication.py:186`

**Suggested Action**: [TO BE DETERMINED]

### string:   ⚠️  Error analyzing ...
**Occurrences**: 3

- `audit_code_duplication.py:248`
- `audit_code_duplication.py:292`
- `audit_code_duplication.py:334`

**Suggested Action**: [TO BE DETERMINED]

### string: : File not found...
**Occurrences**: 3

- `check_syntax.py:47`
- `validate_implementation.py:99`
- `validate_implementation.py:87`

**Suggested Action**: [TO BE DETERMINED]

### string: --num-iterations...
**Occurrences**: 3

- `main.py:110`
- `scripts/quick_start_demo.py:102`
- `tests/integration/test_experiment_pipeline.py:524`

**Suggested Action**: [TO BE DETERMINED]

### string: --output-dir...
**Occurrences**: 8

- `main.py:157`
- `scripts/generate_correlation_matrix.py:35`
- `scripts/generate_figures.py:528`
- `scripts/generate_tables.py:507`
- `scripts/performance_report.py:437`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: configs/default.yaml...
**Occurrences**: 4

- `main.py:71`
- `validate_implementation.py:91`
- `scripts/implementation_demo.py:27`
- `scripts/quick_start_demo.py:104`

**Suggested Action**: [TO BE DETERMINED]

### string: wikitext-103...
**Occurrences**: 48

- `main.py:231`
- `main.py:232`
- `main.py:104`
- `scripts/quick_start_demo.py:31`
- `scripts/test_framework.py:85`
- ... and 43 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: permute_only...
**Occurrences**: 16

- `main.py:81`
- `validate_implementation.py:46`
- `validate_implementation.py:140`
- `scripts/implementation_demo.py:43`
- `scripts/quick_start_demo.py:185`
- ... and 11 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: sparsity_only...
**Occurrences**: 20

- `main.py:82`
- `validate_implementation.py:46`
- `validate_implementation.py:140`
- `scripts/generate_figures.py:50`
- `scripts/generate_figures.py:191`
- ... and 15 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: linear_sparsity...
**Occurrences**: 28

- `main.py:83`
- `main.py:225`
- `validate_implementation.py:46`
- `validate_implementation.py:140`
- `scripts/generate_figures.py:51`
- ... and 23 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: iterative_sparsity...
**Occurrences**: 42

- `main.py:84`
- `main.py:225`
- `validate_implementation.py:47`
- `validate_implementation.py:141`
- `scripts/generate_figures.py:52`
- ... and 37 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: linear_quant_permute_first...
**Occurrences**: 15

- `main.py:85`
- `validate_implementation.py:47`
- `validate_implementation.py:141`
- `scripts/generate_figures.py:53`
- `scripts/generate_figures.py:108`
- ... and 10 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: linear_quant_quant_first...
**Occurrences**: 15

- `main.py:86`
- `validate_implementation.py:48`
- `validate_implementation.py:142`
- `scripts/generate_figures.py:54`
- `scripts/generate_figures.py:107`
- ... and 10 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: iterative_quant...
**Occurrences**: 18

- `main.py:87`
- `validate_implementation.py:48`
- `validate_implementation.py:142`
- `scripts/generate_figures.py:55`
- `scripts/generate_figures.py:109`
- ... and 13 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: improvement_pct...
**Occurrences**: 15

- `main.py:297`
- `main.py:298`
- `scripts/generate_tables.py:457`
- `scripts/generate_tables.py:203`
- `scripts/run_experiment.py:618`
- ... and 10 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: rich_formatting...
**Occurrences**: 3

- `main.py:257`
- `scripts/quick_start_demo.py:84`
- `tests/integration/test_experiment_pipeline.py:95`

**Suggested Action**: [TO BE DETERMINED]

### string: mean_latency_ms...
**Occurrences**: 16

- `main.py:296`
- `scripts/generate_figures.py:508`
- `scripts/generate_tables.py:291`
- `scripts/run_experiment.py:306`
- `scripts/run_experiment.py:608`
- ... and 11 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: std_latency_ms...
**Occurrences**: 13

- `main.py:296`
- `scripts/generate_figures.py:509`
- `scripts/generate_tables.py:292`
- `scripts/run_experiment.py:358`
- `scripts/run_experiment.py:309`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: num_clusters_used...
**Occurrences**: 4

- `test_basic_functionality.py:38`
- `src/co_design/spectral.py:90`
- `tests/integration/test_permutation_integration.py:162`
- `tests/unit/test_spectral.py:282`

**Suggested Action**: [TO BE DETERMINED]

### string: silhouette_score...
**Occurrences**: 4

- `test_basic_functionality.py:39`
- `src/co_design/spectral.py:91`
- `tests/integration/test_permutation_integration.py:163`
- `tests/unit/test_spectral.py:283`

**Suggested Action**: [TO BE DETERMINED]

### string: permutation_size...
**Occurrences**: 5

- `test_basic_functionality.py:83`
- `src/co_design/apply.py:107`
- `src/co_design/apply.py:353`
- `src/co_design/apply.py:81`
- `tests/unit/test_apply.py:349`

**Suggested Action**: [TO BE DETERMINED]

### string: reproducibility...
**Occurrences**: 6

- `validate_implementation.py:36`
- `scripts/quick_start_demo.py:86`
- `scripts/run_experiment.py:87`
- `scripts/run_experiment.py:94`
- `tests/integration/test_cli_full_pipeline.py:138`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: ' not found in model '...
**Occurrences**: 3

- `scripts/generate_correlation_matrix.py:110`
- `src/utils/config.py:277`
- `src/utils/exceptions.py:38`

**Suggested Action**: [TO BE DETERMINED]

### string: cached_matrices...
**Occurrences**: 13

- `scripts/generate_correlation_matrix.py:128`
- `scripts/generate_correlation_matrix.py:148`
- `scripts/generate_correlation_matrix.py:174`
- `scripts/generate_correlation_matrix.py:73`
- `scripts/generate_correlation_matrix.py:132`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: total_size_mb...
**Occurrences**: 5

- `scripts/generate_correlation_matrix.py:173`
- `scripts/generate_correlation_matrix.py:74`
- `src/co_design/correlation.py:366`
- `src/co_design/correlation.py:358`
- `tests/unit/test_correlation.py:179`

**Suggested Action**: [TO BE DETERMINED]

### string: Linear Pipeline...
**Occurrences**: 4

- `scripts/generate_figures.py:220`
- `scripts/generate_figures.py:409`
- `scripts/generate_tables.py:82`
- `scripts/generate_tables.py:409`

**Suggested Action**: [TO BE DETERMINED]

### string: Iterative Co-Design...
**Occurrences**: 4

- `scripts/generate_figures.py:410`
- `scripts/generate_tables.py:83`
- `scripts/generate_tables.py:180`
- `scripts/generate_tables.py:410`

**Suggested Action**: [TO BE DETERMINED]

### string: final_benchmark...
**Occurrences**: 11

- `scripts/generate_figures.py:502`
- `scripts/generate_tables.py:284`
- `scripts/generate_tables.py:321`
- `scripts/run_experiment.py:599`
- `scripts/run_experiment.py:605`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: quantization...
**Occurrences**: 6

- `scripts/generate_figures.py:547`
- `scripts/generate_figures.py:531`
- `scripts/generate_figures.py:551`
- `scripts/generate_tables.py:527`
- `scripts/generate_tables.py:510`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: baseline_benchmark...
**Occurrences**: 11

- `scripts/generate_figures.py:504`
- `scripts/generate_tables.py:287`
- `scripts/run_experiment.py:308`
- `scripts/run_experiment.py:608`
- `tests/integration/test_cli_full_pipeline.py:184`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: experiment_id...
**Occurrences**: 11

- `scripts/generate_figures.py:87`
- `scripts/generate_tables.py:49`
- `scripts/run_experiment.py:64`
- `scripts/run_experiment.py:170`
- `scripts/run_experiment.py:638`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: latency_std...
**Occurrences**: 3

- `scripts/generate_figures.py:239`
- `scripts/generate_figures.py:207`
- `scripts/generate_figures.py:241`

**Suggested Action**: [TO BE DETERMINED]

### string: min_latency_ms...
**Occurrences**: 5

- `scripts/generate_figures.py:510`
- `scripts/generate_tables.py:293`
- `scripts/run_experiment.py:359`
- `src/profiler/latency.py:57`
- `src/utils/profiler.py:402`

**Suggested Action**: [TO BE DETERMINED]

### string: max_latency_ms...
**Occurrences**: 5

- `scripts/generate_figures.py:511`
- `scripts/generate_tables.py:294`
- `scripts/run_experiment.py:360`
- `src/profiler/latency.py:58`
- `src/utils/profiler.py:403`

**Suggested Action**: [TO BE DETERMINED]

### string: \begin{table}[htbp]
...
**Occurrences**: 4

- `scripts/generate_tables.py:333`
- `scripts/generate_tables.py:399`
- `scripts/generate_tables.py:445`
- `scripts/generate_tables.py:473`

**Suggested Action**: [TO BE DETERMINED]

### string: \centering
...
**Occurrences**: 4

- `scripts/generate_tables.py:334`
- `scripts/generate_tables.py:400`
- `scripts/generate_tables.py:446`
- `scripts/generate_tables.py:474`

**Suggested Action**: [TO BE DETERMINED]

### string: \bottomrule
...
**Occurrences**: 4

- `scripts/generate_tables.py:390`
- `scripts/generate_tables.py:436`
- `scripts/generate_tables.py:464`
- `scripts/generate_tables.py:495`

**Suggested Action**: [TO BE DETERMINED]

### string: \end{tabular}
...
**Occurrences**: 4

- `scripts/generate_tables.py:391`
- `scripts/generate_tables.py:437`
- `scripts/generate_tables.py:465`
- `scripts/generate_tables.py:496`

**Suggested Action**: [TO BE DETERMINED]

### string: \end{table}
...
**Occurrences**: 4

- `scripts/generate_tables.py:392`
- `scripts/generate_tables.py:438`
- `scripts/generate_tables.py:466`
- `scripts/generate_tables.py:497`

**Suggested Action**: [TO BE DETERMINED]

### string: strategy_results...
**Occurrences**: 47

- `scripts/generate_tables.py:301`
- `scripts/run_experiment.py:470`
- `scripts/run_experiment.py:478`
- `scripts/run_experiment.py:489`
- `scripts/run_experiment.py:504`
- ... and 42 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: iasp_results...
**Occurrences**: 13

- `scripts/generate_tables.py:304`
- `scripts/run_experiment.py:480`
- `scripts/run_experiment.py:507`
- `scripts/run_experiment.py:554`
- `scripts/run_experiment.py:571`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: l2_cache_hit_rate...
**Occurrences**: 8

- `scripts/generate_tables.py:427`
- `scripts/generate_tables.py:144`
- `scripts/generate_tables.py:325`
- `scripts/generate_tables.py:136`
- `scripts/generate_tables.py:418`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: python main.py --strategy iterative_sparsity --mod...
**Occurrences**: 3

- `scripts/implementation_demo.py:140`
- `scripts/quick_start_demo.py:237`
- `scripts/quick_start_demo.py:249`

**Suggested Action**: [TO BE DETERMINED]

### string: Matrix Size...
**Occurrences**: 7

- `scripts/performance_report.py:61`
- `scripts/performance_report.py:92`
- `scripts/performance_report.py:118`
- `scripts/performance_report.py:127`
- `scripts/performance_report.py:151`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: spectral_vs_random...
**Occurrences**: 4

- `scripts/performance_report.py:145`
- `scripts/performance_report.py:146`
- `scripts/performance_report.py:351`
- `scripts/performance_report.py:352`

**Suggested Action**: [TO BE DETERMINED]

### string: algorithmic_benchmarks...
**Occurrences**: 4

- `scripts/performance_report.py:302`
- `scripts/performance_report.py:464`
- `scripts/performance_report.py:310`
- `scripts/performance_report.py:465`

**Suggested Action**: [TO BE DETERMINED]

### string: memory_analysis...
**Occurrences**: 6

- `scripts/performance_report.py:328`
- `scripts/performance_report.py:408`
- `scripts/performance_report.py:473`
- `scripts/performance_report.py:334`
- `scripts/performance_report.py:409`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: comparative_analysis...
**Occurrences**: 4

- `scripts/performance_report.py:344`
- `scripts/performance_report.py:478`
- `scripts/performance_report.py:345`
- `scripts/performance_report.py:479`

**Suggested Action**: [TO BE DETERMINED]

### string: mean_time_ms...
**Occurrences**: 27

- `scripts/performance_report.py:56`
- `scripts/performance_report.py:313`
- `tests/performance/test_benchmarks.py:113`
- `tests/performance/test_benchmarks.py:259`
- `tests/performance/test_benchmarks.py:608`
- ... and 22 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: peak_memory_mb...
**Occurrences**: 11

- `scripts/performance_report.py:110`
- `tests/performance/test_benchmarks.py:70`
- `tests/performance/test_benchmarks.py:108`
- `tests/performance/test_benchmarks.py:327`
- `tests/performance/test_benchmarks.py:231`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: memory_efficiency...
**Occurrences**: 6

- `scripts/performance_report.py:112`
- `scripts/performance_report.py:336`
- `scripts/performance_report.py:410`
- `tests/performance/test_benchmarks.py:232`
- `tests/performance/test_benchmarks.py:643`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: overhead_factor...
**Occurrences**: 6

- `scripts/performance_report.py:148`
- `scripts/performance_report.py:352`
- `tests/performance/test_benchmarks.py:485`
- `tests/performance/test_benchmarks.py:719`
- `tests/performance/test_benchmarks.py:722`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: speedup_factor...
**Occurrences**: 18

- `scripts/performance_report.py:160`
- `scripts/performance_report.py:188`
- `scripts/performance_report.py:188`
- `scripts/performance_report.py:189`
- `scripts/run_experiment.py:619`
- ... and 13 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: gpu_speedup...
**Occurrences**: 4

- `scripts/performance_report.py:174`
- `scripts/performance_report.py:360`
- `tests/performance/test_benchmarks.py:577`
- `tests/performance/test_benchmarks.py:756`

**Suggested Action**: [TO BE DETERMINED]

### string: pretrained_path...
**Occurrences**: 3

- `scripts/quick_start_demo.py:27`
- `scripts/run_experiment.py:284`
- `tests/integration/test_experiment_pipeline.py:38`

**Suggested Action**: [TO BE DETERMINED]

### string: state-spaces/mamba-3b...
**Occurrences**: 3

- `scripts/quick_start_demo.py:26`
- `src/models/manager.py:24`
- `tests/integration/test_experiment_pipeline.py:37`

**Suggested Action**: [TO BE DETERMINED]

### string: sequence_length...
**Occurrences**: 13

- `scripts/quick_start_demo.py:33`
- `scripts/run_experiment.py:294`
- `src/co_design/correlation.py:143`
- `src/co_design/correlation.py:143`
- `src/co_design/correlation.py:127`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: num_samples...
**Occurrences**: 16

- `scripts/quick_start_demo.py:35`
- `scripts/run_experiment.py:295`
- `src/co_design/ptq.py:567`
- `src/co_design/ptq.py:613`
- `src/co_design/ptq.py:606`
- ... and 11 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: num_clusters...
**Occurrences**: 11

- `scripts/quick_start_demo.py:40`
- `scripts/run_experiment.py:407`
- `scripts/run_experiment.py:407`
- `scripts/run_experiment.py:383`
- `src/co_design/iasp.py:111`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: correlation_threshold...
**Occurrences**: 6

- `scripts/quick_start_demo.py:41`
- `scripts/run_experiment.py:385`
- `src/co_design/iasp.py:112`
- `src/utils/config.py:69`
- `tests/integration/test_cli_full_pipeline.py:116`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: layers.0.mixer...
**Occurrences**: 4

- `scripts/quick_start_demo.py:39`
- `src/utils/config.py:304`
- `tests/integration/test_experiment_pipeline.py:50`
- `tests/integration/test_model_integration.py:55`

**Suggested Action**: [TO BE DETERMINED]

### string: ./data/correlation_matrices/...
**Occurrences**: 3

- `scripts/quick_start_demo.py:43`
- `src/utils/config.py:61`
- `tests/integration/test_experiment_pipeline.py:54`

**Suggested Action**: [TO BE DETERMINED]

### string: learning_rate...
**Occurrences**: 5

- `scripts/quick_start_demo.py:47`
- `scripts/run_experiment.py:421`
- `src/utils/config.py:98`
- `tests/integration/test_cli_full_pipeline.py:121`
- `tests/integration/test_experiment_pipeline.py:58`

**Suggested Action**: [TO BE DETERMINED]

### string: gumbel_temperature...
**Occurrences**: 3

- `scripts/quick_start_demo.py:49`
- `scripts/run_experiment.py:423`
- `tests/integration/test_experiment_pipeline.py:60`

**Suggested Action**: [TO BE DETERMINED]

### string: sparsity_ratio...
**Occurrences**: 6

- `scripts/quick_start_demo.py:50`
- `scripts/run_experiment.py:420`
- `src/utils/config.py:104`
- `tests/integration/test_cli_full_pipeline.py:120`
- `tests/integration/test_experiment_pipeline.py:61`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: calibration_samples...
**Occurrences**: 5

- `scripts/quick_start_demo.py:55`
- `scripts/run_experiment.py:461`
- `scripts/run_experiment.py:449`
- `tests/integration/test_cli_full_pipeline.py:127`
- `tests/integration/test_experiment_pipeline.py:66`

**Suggested Action**: [TO BE DETERMINED]

### string: num_iterations...
**Occurrences**: 13

- `scripts/quick_start_demo.py:59`
- `scripts/quick_start_demo.py:119`
- `scripts/quick_start_demo.py:169`
- `scripts/run_experiment.py:514`
- `scripts/run_experiment.py:538`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: save_intermediate...
**Occurrences**: 7

- `scripts/quick_start_demo.py:62`
- `scripts/run_experiment.py:399`
- `src/utils/config.py:373`
- `src/utils/config.py:374`
- `tests/integration/test_cli_full_pipeline.py:105`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: warmup_runs...
**Occurrences**: 9

- `scripts/quick_start_demo.py:65`
- `scripts/run_experiment.py:103`
- `scripts/run_experiment.py:128`
- `src/profiler/calibration.py:301`
- `src/utils/config.py:165`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: use_cuda_events...
**Occurrences**: 5

- `scripts/quick_start_demo.py:67`
- `scripts/run_experiment.py:105`
- `src/utils/profiler.py:322`
- `tests/integration/test_cli_full_pipeline.py:132`
- `tests/integration/test_experiment_pipeline.py:78`

**Suggested Action**: [TO BE DETERMINED]

### string: pytorch_profiler...
**Occurrences**: 6

- `scripts/quick_start_demo.py:72`
- `src/utils/config.py:194`
- `src/utils/profiler.py:251`
- `src/utils/profiler.py:59`
- `src/utils/profiler.py:90`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: deterministic...
**Occurrences**: 4

- `scripts/quick_start_demo.py:87`
- `scripts/run_experiment.py:87`
- `tests/integration/test_cli_full_pipeline.py:139`
- `tests/integration/test_experiment_pipeline.py:98`

**Suggested Action**: [TO BE DETERMINED]

### string: cuda_deterministic...
**Occurrences**: 4

- `scripts/quick_start_demo.py:88`
- `scripts/run_experiment.py:94`
- `tests/integration/test_cli_full_pipeline.py:140`
- `tests/integration/test_experiment_pipeline.py:99`

**Suggested Action**: [TO BE DETERMINED]

### string: lts__t_sector_hit_rate.pct...
**Occurrences**: 15

- `scripts/quick_start_demo.py:73`
- `src/profiler/ncu.py:451`
- `src/profiler/ncu.py:32`
- `src/utils/config.py:184`
- `src/utils/profiler.py:222`
- ... and 10 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: reproducibility_info...
**Occurrences**: 4

- `scripts/run_experiment.py:65`
- `tests/integration/test_experiment_pipeline.py:412`
- `tests/integration/test_experiment_pipeline.py:390`
- `tests/integration/test_experiment_pipeline.py:390`

**Suggested Action**: [TO BE DETERMINED]

### string: torch_version...
**Occurrences**: 8

- `scripts/run_experiment.py:145`
- `src/profiler/calibration.py:182`
- `src/profiler/calibration.py:390`
- `src/profiler/calibration.py:391`
- `tests/unit/test_profiler_calibration.py:162`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: cuda_version...
**Occurrences**: 7

- `scripts/run_experiment.py:146`
- `src/profiler/calibration.py:183`
- `src/profiler/calibration.py:396`
- `src/profiler/calibration.py:397`
- `tests/unit/test_profiler_calibration.py:330`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: python_version...
**Occurrences**: 3

- `scripts/run_experiment.py:147`
- `src/profiler/calibration.py:181`
- `tests/unit/test_profiler_calibration.py:161`

**Suggested Action**: [TO BE DETERMINED]

### string: device_count...
**Occurrences**: 3

- `scripts/run_experiment.py:160`
- `src/profiler/calibration.py:196`
- `tests/unit/test_profiler_calibration.py:186`

**Suggested Action**: [TO BE DETERMINED]

### string: current_device...
**Occurrences**: 3

- `scripts/run_experiment.py:161`
- `src/profiler/calibration.py:197`
- `tests/unit/test_profiler_calibration.py:187`

**Suggested Action**: [TO BE DETERMINED]

### string: device_name...
**Occurrences**: 8

- `scripts/run_experiment.py:162`
- `src/profiler/calibration.py:386`
- `src/profiler/calibration.py:386`
- `src/profiler/calibration.py:203`
- `tests/unit/test_profiler_calibration.py:332`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: device_capability...
**Occurrences**: 3

- `scripts/run_experiment.py:163`
- `src/profiler/calibration.py:204`
- `tests/unit/test_profiler_calibration.py:189`

**Suggested Action**: [TO BE DETERMINED]

### string: config.json...
**Occurrences**: 4

- `scripts/run_experiment.py:179`
- `tests/integration/test_experiment_pipeline.py:162`
- `tests/integration/test_experiment_pipeline.py:431`
- `tests/integration/test_experiment_pipeline.py:436`

**Suggested Action**: [TO BE DETERMINED]

### string: precomputed_correlation...
**Occurrences**: 4

- `scripts/run_experiment.py:374`
- `scripts/run_experiment.py:375`
- `src/utils/config.py:414`
- `src/utils/config.py:415`

**Suggested Action**: [TO BE DETERMINED]

### string: permutation...
**Occurrences**: 5

- `scripts/run_experiment.py:405`
- `tests/integration/test_cli_full_pipeline.py:210`
- `tests/integration/test_cli_full_pipeline.py:350`
- `tests/integration/test_cli_full_pipeline.py:351`
- `tests/unit/test_iasp.py:102`

**Suggested Action**: [TO BE DETERMINED]

### string: final_sparsity...
**Occurrences**: 8

- `scripts/run_experiment.py:434`
- `scripts/run_experiment.py:434`
- `src/co_design/hds.py:510`
- `tests/integration/test_experiment_pipeline.py:210`
- `tests/integration/test_experiment_pipeline.py:241`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: hds_results...
**Occurrences**: 6

- `scripts/run_experiment.py:491`
- `scripts/run_experiment.py:506`
- `tests/integration/test_cli_full_pipeline.py:228`
- `tests/integration/test_cli_full_pipeline.py:246`
- `tests/integration/test_experiment_pipeline.py:224`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: linear_iasp...
**Occurrences**: 3

- `scripts/run_experiment.py:502`
- `scripts/run_experiment.py:547`
- `scripts/run_experiment.py:566`

**Suggested Action**: [TO BE DETERMINED]

### string: initial_iasp...
**Occurrences**: 4

- `scripts/run_experiment.py:589`
- `scripts/run_experiment.py:531`
- `tests/integration/test_experiment_pipeline.py:307`
- `tests/integration/test_experiment_pipeline.py:348`

**Suggested Action**: [TO BE DETERMINED]

### string: improvements...
**Occurrences**: 7

- `scripts/run_experiment.py:614`
- `scripts/run_experiment.py:646`
- `scripts/run_experiment.py:647`
- `tests/integration/test_cli_full_pipeline.py:402`
- `tests/integration/test_cli_full_pipeline.py:404`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: baseline_latency_ms...
**Occurrences**: 7

- `scripts/run_experiment.py:615`
- `scripts/run_experiment.py:649`
- `src/profiler/calibration.py:71`
- `tests/integration/test_cli_full_pipeline.py:405`
- `tests/integration/test_cli_full_pipeline.py:412`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: final_latency_ms...
**Occurrences**: 5

- `scripts/run_experiment.py:616`
- `scripts/run_experiment.py:650`
- `tests/integration/test_cli_full_pipeline.py:406`
- `tests/integration/test_cli_full_pipeline.py:413`
- `tests/integration/test_experiment_pipeline.py:423`

**Suggested Action**: [TO BE DETERMINED]

### string: improvement_ms...
**Occurrences**: 8

- `scripts/run_experiment.py:617`
- `scripts/run_experiment.py:651`
- `src/profiler/latency.py:501`
- `tests/integration/test_cli_full_pipeline.py:407`
- `tests/integration/test_cli_full_pipeline.py:414`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: results.json...
**Occurrences**: 7

- `scripts/run_experiment.py:627`
- `tests/integration/test_cli_full_pipeline.py:368`
- `tests/integration/test_cli_full_pipeline.py:190`
- `tests/integration/test_cli_full_pipeline.py:542`
- `tests/integration/test_experiment_pipeline.py:163`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: summary.txt...
**Occurrences**: 5

- `scripts/run_experiment.py:634`
- `tests/integration/test_cli_full_pipeline.py:380`
- `tests/integration/test_cli_full_pipeline.py:191`
- `tests/integration/test_experiment_pipeline.py:164`
- `tests/integration/test_experiment_pipeline.py:433`

**Suggested Action**: [TO BE DETERMINED]

### string: duration_seconds...
**Occurrences**: 4

- `scripts/run_experiment.py:232`
- `scripts/run_experiment.py:237`
- `scripts/run_experiment.py:644`
- `tests/integration/test_experiment_pipeline.py:414`

**Suggested Action**: [TO BE DETERMINED]

### string: total_parameters...
**Occurrences**: 6

- `scripts/test_framework.py:124`
- `src/models/permutable_model.py:343`
- `tests/integration/test_model_integration.py:152`
- `tests/integration/test_model_integration.py:154`
- `tests/unit/test_model_manager.py:208`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: torch_geometric...
**Occurrences**: 6

- `scripts/test_framework.py:129`
- `tests/integration/test_cli_correlation.py:84`
- `tests/integration/test_model_integration.py:104`
- `tests/integration/test_model_integration.py:139`
- `tests/integration/test_model_integration.py:157`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: invalid-model...
**Occurrences**: 7

- `scripts/test_framework.py:42`
- `tests/integration/test_cli_correlation.py:48`
- `tests/integration/test_cli_correlation.py:56`
- `tests/integration/test_model_integration.py:166`
- `tests/unit/test_model_manager.py:52`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: expected_size...
**Occurrences**: 3

- `src/co_design/apply.py:187`
- `tests/unit/test_apply.py:59`
- `tests/unit/test_apply.py:121`

**Suggested Action**: [TO BE DETERMINED]

### string: weight_shape...
**Occurrences**: 9

- `src/co_design/apply.py:188`
- `src/co_design/correlation.py:159`
- `src/co_design/correlation.py:162`
- `src/co_design/correlation.py:162`
- `src/co_design/correlation.py:162`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: num_permuted_layers...
**Occurrences**: 3

- `src/co_design/apply.py:469`
- `tests/integration/test_permutation_integration.py:286`
- `tests/unit/test_apply.py:449`

**Suggested Action**: [TO BE DETERMINED]

### string: total_permutations...
**Occurrences**: 3

- `src/co_design/apply.py:471`
- `tests/integration/test_permutation_integration.py:287`
- `tests/unit/test_apply.py:451`

**Suggested Action**: [TO BE DETERMINED]

### string: Invalid dimension: ...
**Occurrences**: 3

- `src/co_design/apply.py:157`
- `src/models/permutable_model.py:327`
- `src/models/permutable_model.py:212`

**Suggested Action**: [TO BE DETERMINED]

### string: valid_layers...
**Occurrences**: 3

- `src/co_design/hds.py:749`
- `src/co_design/hds.py:800`
- `tests/unit/test_hds.py:322`

**Suggested Action**: [TO BE DETERMINED]

### string: invalid_layers...
**Occurrences**: 3

- `src/co_design/hds.py:750`
- `src/co_design/hds.py:802`
- `tests/unit/test_hds.py:323`

**Suggested Action**: [TO BE DETERMINED]

### string: overall_sparsity...
**Occurrences**: 4

- `src/co_design/hds.py:751`
- `src/co_design/hds.py:805`
- `tests/unit/test_hds.py:324`
- `tests/unit/test_hds.py:326`

**Suggested Action**: [TO BE DETERMINED]

### string: pattern_compliance...
**Occurrences**: 4

- `src/co_design/hds.py:752`
- `src/co_design/hds.py:806`
- `tests/unit/test_hds.py:325`
- `tests/unit/test_hds.py:327`

**Suggested Action**: [TO BE DETERMINED]

### string: training_history...
**Occurrences**: 8

- `src/co_design/hds.py:509`
- `src/co_design/hds.py:663`
- `src/co_design/hds.py:677`
- `tests/integration/test_hds_ptq_integration.py:72`
- `tests/integration/test_hds_ptq_integration.py:137`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: avg_sparsity...
**Occurrences**: 11

- `src/co_design/hds.py:636`
- `src/co_design/hds.py:495`
- `tests/integration/test_experiment_pipeline.py:210`
- `tests/integration/test_experiment_pipeline.py:241`
- `tests/integration/test_experiment_pipeline.py:281`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: optimizer_state_dict...
**Occurrences**: 3

- `src/co_design/hds.py:664`
- `src/co_design/hds.py:679`
- `src/co_design/hds.py:680`

**Suggested Action**: [TO BE DETERMINED]

### string: scheduler_state_dict...
**Occurrences**: 3

- `src/co_design/hds.py:665`
- `src/co_design/hds.py:682`
- `src/co_design/hds.py:683`

**Suggested Action**: [TO BE DETERMINED]

### string: block_coherence...
**Occurrences**: 4

- `src/co_design/iasp.py:356`
- `tests/integration/test_iasp_integration.py:312`
- `tests/integration/test_iasp_integration.py:319`
- `tests/integration/test_permutation_integration.py:429`

**Suggested Action**: [TO BE DETERMINED]

### string: permutation_length...
**Occurrences**: 5

- `src/co_design/iasp.py:357`
- `tests/integration/test_iasp_integration.py:313`
- `tests/integration/test_iasp_integration.py:316`
- `tests/integration/test_permutation_integration.py:430`
- `tests/integration/test_permutation_integration.py:431`

**Suggested Action**: [TO BE DETERMINED]

### string: correlation...
**Occurrences**: 3

- `src/co_design/ptq.py:566`
- `src/co_design/ptq.py:622`
- `tests/unit/test_ptq.py:351`

**Suggested Action**: [TO BE DETERMINED]

### string: calibration_stats...
**Occurrences**: 4

- `src/co_design/ptq.py:482`
- `tests/integration/test_hds_ptq_integration.py:158`
- `tests/unit/test_ptq.py:286`
- `tests/unit/test_ptq.py:325`

**Suggested Action**: [TO BE DETERMINED]

### string: num_quantized_layers...
**Occurrences**: 9

- `src/co_design/ptq.py:483`
- `tests/integration/test_experiment_pipeline.py:327`
- `tests/integration/test_hds_ptq_integration.py:159`
- `tests/integration/test_hds_ptq_integration.py:313`
- `tests/integration/test_hds_ptq_integration.py:104`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: quantized_layers...
**Occurrences**: 3

- `src/co_design/ptq.py:484`
- `tests/unit/test_ptq.py:288`
- `tests/unit/test_ptq.py:327`

**Suggested Action**: [TO BE DETERMINED]

### string: requires_transformers...
**Occurrences**: 6

- `src/models/manager.py:26`
- `src/models/manager.py:31`
- `src/models/manager.py:36`
- `src/models/manager.py:41`
- `src/models/manager.py:131`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string:  not implemented...
**Occurrences**: 8

- `src/models/manager.py:197`
- `src/utils/dataset_manager.py:123`
- `src/utils/dataset_manager.py:169`
- `src/utils/dataset_manager.py:186`
- `src/utils/dataset_manager.py:294`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string:  not supported...
**Occurrences**: 4

- `src/models/manager.py:237`
- `src/utils/dataset_manager.py:79`
- `src/utils/dataset_manager.py:217`
- `src/utils/dataset_manager.py:324`

**Suggested Action**: [TO BE DETERMINED]

### string: trainable_parameters...
**Occurrences**: 4

- `src/models/permutable_model.py:97`
- `src/models/permutable_model.py:344`
- `tests/integration/test_model_integration.py:153`
- `tests/unit/test_model_manager.py:209`

**Suggested Action**: [TO BE DETERMINED]

### string: Convert to dictionary for serialization....
**Occurrences**: 3

- `src/profiler/calibration.py:68`
- `src/profiler/latency.py:53`
- `src/profiler/ncu.py:76`

**Suggested Action**: [TO BE DETERMINED]

### string: system_info...
**Occurrences**: 6

- `src/profiler/calibration.py:77`
- `src/profiler/calibration.py:379`
- `src/profiler/calibration.py:297`
- `tests/unit/test_profiler_calibration.py:268`
- `tests/unit/test_profiler_calibration.py:316`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: error_message...
**Occurrences**: 4

- `src/profiler/calibration.py:78`
- `src/profiler/ncu.py:89`
- `tests/unit/test_profiler_calibration.py:101`
- `tests/unit/test_profiler_ncu.py:96`

**Suggested Action**: [TO BE DETERMINED]

### string: baseline_mean_ms...
**Occurrences**: 3

- `src/profiler/latency.py:504`
- `tests/integration/test_profiler_integration.py:196`
- `tests/unit/test_profiler_latency.py:349`

**Suggested Action**: [TO BE DETERMINED]

### string: optimized_mean_ms...
**Occurrences**: 3

- `src/profiler/latency.py:505`
- `tests/integration/test_profiler_integration.py:197`
- `tests/unit/test_profiler_latency.py:350`

**Suggested Action**: [TO BE DETERMINED]

### string: statistically_significant...
**Occurrences**: 3

- `src/profiler/latency.py:509`
- `tests/integration/test_profiler_integration.py:194`
- `tests/unit/test_profiler_latency.py:352`

**Suggested Action**: [TO BE DETERMINED]

### string: cuda_events...
**Occurrences**: 6

- `src/profiler/latency.py:345`
- `tests/unit/test_profiler_latency.py:96`
- `tests/unit/test_profiler_latency.py:66`
- `tests/unit/test_profiler_latency.py:89`
- `tests/unit/test_profiler_latency.py:324`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: perf_counter...
**Occurrences**: 5

- `src/profiler/latency.py:345`
- `tests/unit/test_profiler_calibration.py:247`
- `tests/unit/test_profiler_calibration.py:291`
- `tests/unit/test_profiler_latency.py:181`
- `tests/unit/test_profiler_latency.py:416`

**Suggested Action**: [TO BE DETERMINED]

### string: dram_bandwidth_gb_s...
**Occurrences**: 3

- `src/profiler/ncu.py:82`
- `src/utils/profiler.py:233`
- `tests/unit/test_profiler_ncu.py:95`

**Suggested Action**: [TO BE DETERMINED]

### string: dram__bytes_read.sum...
**Occurrences**: 9

- `src/profiler/ncu.py:466`
- `src/profiler/ncu.py:33`
- `src/utils/config.py:185`
- `src/utils/profiler.py:227`
- `tests/integration/test_profiler_integration.py:228`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: dram__bytes_write.sum...
**Occurrences**: 5

- `src/profiler/ncu.py:472`
- `src/profiler/ncu.py:34`
- `src/utils/config.py:186`
- `src/utils/profiler.py:228`
- `tests/unit/test_profiler_ncu.py:202`

**Suggested Action**: [TO BE DETERMINED]

### string: sm__warps_active.avg.pct_of_peak_sustained_active...
**Occurrences**: 3

- `src/profiler/ncu.py:486`
- `src/profiler/ncu.py:35`
- `src/utils/config.py:187`

**Suggested Action**: [TO BE DETERMINED]

### string: gpu__time_duration.sum...
**Occurrences**: 3

- `src/profiler/ncu.py:498`
- `src/profiler/ncu.py:38`
- `tests/unit/test_profiler_ncu.py:36`

**Suggested Action**: [TO BE DETERMINED]

### string: nsight_compute...
**Occurrences**: 6

- `src/utils/config.py:181`
- `src/utils/config.py:194`
- `src/utils/profiler.py:37`
- `src/utils/profiler.py:46`
- `src/utils/profiler.py:88`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string:  not supported. Supported: ...
**Occurrences**: 11

- `src/utils/config.py:21`
- `src/utils/config.py:28`
- `src/utils/config.py:45`
- `src/utils/config.py:79`
- `src/utils/config.py:95`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: 

Suggestions:...
**Occurrences**: 6

- `src/utils/exceptions.py:133`
- `src/utils/exceptions.py:139`
- `src/utils/exceptions.py:146`
- `src/utils/exceptions.py:152`
- `src/utils/exceptions.py:158`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: Setup test fixtures....
**Occurrences**: 37

- `tests/integration/test_cli_correlation.py:15`
- `tests/integration/test_cli_full_pipeline.py:84`
- `tests/integration/test_cli_full_pipeline.py:427`
- `tests/integration/test_cli_full_pipeline.py:487`
- `tests/integration/test_experiment_pipeline.py:29`
- ... and 32 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: Cleanup test fixtures....
**Occurrences**: 3

- `tests/integration/test_cli_full_pipeline.py:145`
- `tests/integration/test_cli_full_pipeline.py:431`
- `tests/integration/test_cli_full_pipeline.py:496`

**Suggested Action**: [TO BE DETERMINED]

### string: src.models.manager.ModelManager.load_model...
**Occurrences**: 11

- `tests/integration/test_cli_full_pipeline.py:167`
- `tests/integration/test_cli_full_pipeline.py:193`
- `tests/integration/test_cli_full_pipeline.py:212`
- `tests/integration/test_cli_full_pipeline.py:230`
- `tests/integration/test_cli_full_pipeline.py:249`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: src.utils.dataset_manager.DatasetManager.get_datal...
**Occurrences**: 11

- `tests/integration/test_cli_full_pipeline.py:168`
- `tests/integration/test_cli_full_pipeline.py:194`
- `tests/integration/test_cli_full_pipeline.py:213`
- `tests/integration/test_cli_full_pipeline.py:231`
- `tests/integration/test_cli_full_pipeline.py:250`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: nonexistent.layer...
**Occurrences**: 3

- `tests/integration/test_cli_full_pipeline.py:320`
- `tests/unit/test_iasp.py:427`
- `tests/unit/test_model_manager.py:92`

**Suggested Action**: [TO BE DETERMINED]

### string: test_config.yaml...
**Occurrences**: 3

- `tests/integration/test_cli_full_pipeline.py:455`
- `tests/integration/test_experiment_pipeline.py:448`
- `tests/integration/test_model_integration.py:58`

**Suggested Action**: [TO BE DETERMINED]

### string: scripts.run_experiment.ModelManager...
**Occurrences**: 9

- `tests/integration/test_experiment_pipeline.py:138`
- `tests/integration/test_experiment_pipeline.py:166`
- `tests/integration/test_experiment_pipeline.py:199`
- `tests/integration/test_experiment_pipeline.py:229`
- `tests/integration/test_experiment_pipeline.py:269`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: scripts.run_experiment.DatasetManager...
**Occurrences**: 9

- `tests/integration/test_experiment_pipeline.py:139`
- `tests/integration/test_experiment_pipeline.py:167`
- `tests/integration/test_experiment_pipeline.py:200`
- `tests/integration/test_experiment_pipeline.py:230`
- `tests/integration/test_experiment_pipeline.py:270`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: scripts.run_experiment.IASPPermutationOptimizer...
**Occurrences**: 4

- `tests/integration/test_experiment_pipeline.py:168`
- `tests/integration/test_experiment_pipeline.py:232`
- `tests/integration/test_experiment_pipeline.py:272`
- `tests/integration/test_experiment_pipeline.py:318`

**Suggested Action**: [TO BE DETERMINED]

### string: scripts.run_experiment.apply_hds_to_model...
**Occurrences**: 3

- `tests/integration/test_experiment_pipeline.py:201`
- `tests/integration/test_experiment_pipeline.py:231`
- `tests/integration/test_experiment_pipeline.py:271`

**Suggested Action**: [TO BE DETERMINED]

### string: nonexistent_layer...
**Occurrences**: 3

- `tests/integration/test_iasp_integration.py:349`
- `tests/integration/test_permutation_integration.py:206`
- `tests/unit/test_apply.py:390`

**Suggested Action**: [TO BE DETERMINED]

### string: torch_geometric not available...
**Occurrences**: 3

- `tests/integration/test_model_integration.py:105`
- `tests/integration/test_model_integration.py:140`
- `tests/integration/test_model_integration.py:158`

**Suggested Action**: [TO BE DETERMINED]

### string: % deviation...
**Occurrences**: 3

- `tests/integration/test_profiler_integration.py:331`
- `tests/integration/test_profiler_integration.py:367`
- `tests/integration/test_profiler_integration.py:369`

**Suggested Action**: [TO BE DETERMINED]

### string: Modularity: ...
**Occurrences**: 6

- `tests/integration/test_real_models.py:322`
- `tests/integration/test_real_models.py:199`
- `tests/integration/test_real_models.py:251`
- `tests/integration/test_real_models.py:375`
- `tests/integration/test_real_models.py:434`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: execution_time_ms...
**Occurrences**: 4

- `tests/performance/test_benchmarks.py:78`
- `tests/performance/test_benchmarks.py:326`
- `tests/performance/test_benchmarks.py:107`
- `tests/performance/test_benchmarks.py:315`

**Suggested Action**: [TO BE DETERMINED]

### string: linear_scaling_exponent...
**Occurrences**: 3

- `tests/performance/test_benchmarks.py:437`
- `tests/performance/test_benchmarks.py:672`
- `tests/performance/test_benchmarks.py:678`

**Suggested Action**: [TO BE DETERMINED]

### string: cubic_scaling_exponent...
**Occurrences**: 3

- `tests/performance/test_benchmarks.py:438`
- `tests/performance/test_benchmarks.py:673`
- `tests/performance/test_benchmarks.py:682`

**Suggested Action**: [TO BE DETERMINED]

### string: memory_reduction...
**Occurrences**: 3

- `tests/performance/test_benchmarks.py:336`
- `tests/performance/test_benchmarks.py:696`
- `tests/performance/test_benchmarks.py:697`

**Suggested Action**: [TO BE DETERMINED]

### string: mean_memory_mb...
**Occurrences**: 8

- `tests/performance/test_benchmarks.py:125`
- `tests/performance/test_benchmarks.py:486`
- `tests/performance/test_benchmarks.py:487`
- `tests/performance/test_benchmarks.py:578`
- `tests/performance/test_benchmarks.py:579`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: warnings.warn...
**Occurrences**: 3

- `tests/unit/test_apply.py:468`
- `tests/unit/test_spectral.py:174`
- `tests/unit/test_spectral.py:221`

**Suggested Action**: [TO BE DETERMINED]

### string: Test configuration validation....
**Occurrences**: 5

- `tests/unit/test_config.py:41`
- `tests/unit/test_iasp.py:46`
- `tests/unit/test_profiler_calibration.py:51`
- `tests/unit/test_profiler_latency.py:120`
- `tests/unit/test_profiler_ncu.py:48`

**Suggested Action**: [TO BE DETERMINED]

### string: src.utils.text_dataset.WikiTextDataset...
**Occurrences**: 9

- `tests/unit/test_dataset_manager.py:28`
- `tests/unit/test_dataset_manager.py:130`
- `tests/unit/test_dataset_manager.py:158`
- `tests/unit/test_dataset_manager.py:187`
- `tests/unit/test_dataset_manager.py:207`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: Test utility functions....
**Occurrences**: 6

- `tests/unit/test_hds.py:263`
- `tests/unit/test_iasp.py:349`
- `tests/unit/test_profiler_calibration.py:374`
- `tests/unit/test_profiler_latency.py:238`
- `tests/unit/test_profiler_ncu.py:315`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: Test default configuration values....
**Occurrences**: 6

- `tests/unit/test_hds.py:21`
- `tests/unit/test_iasp.py:22`
- `tests/unit/test_profiler_calibration.py:24`
- `tests/unit/test_profiler_latency.py:23`
- `tests/unit/test_profiler_ncu.py:25`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: Test custom configuration values....
**Occurrences**: 6

- `tests/unit/test_hds.py:32`
- `tests/unit/test_iasp.py:32`
- `tests/unit/test_profiler_calibration.py:35`
- `tests/unit/test_profiler_latency.py:33`
- `tests/unit/test_profiler_ncu.py:35`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### string: Test forward pass....
**Occurrences**: 4

- `tests/unit/test_hds.py:118`
- `tests/unit/test_hds.py:164`
- `tests/unit/test_model_manager.py:131`
- `tests/unit/test_ptq.py:140`

**Suggested Action**: [TO BE DETERMINED]

### string: Test error handling and edge cases....
**Occurrences**: 3

- `tests/unit/test_iasp.py:422`
- `tests/unit/test_profiler_latency.py:356`
- `tests/unit/test_profiler_ncu.py:367`

**Suggested Action**: [TO BE DETERMINED]

### string: Test converting results to dictionary....
**Occurrences**: 4

- `tests/unit/test_iasp.py:88`
- `tests/unit/test_profiler_calibration.py:91`
- `tests/unit/test_profiler_latency.py:74`
- `tests/unit/test_profiler_ncu.py:85`

**Suggested Action**: [TO BE DETERMINED]

### string: torch.cuda.is_available...
**Occurrences**: 4

- `tests/unit/test_profiler_calibration.py:127`
- `tests/unit/test_profiler_calibration.py:133`
- `tests/unit/test_profiler_calibration.py:167`
- `tests/unit/test_profiler_calibration.py:175`

**Suggested Action**: [TO BE DETERMINED]

### string: _run_calibration_benchmark...
**Occurrences**: 3

- `tests/unit/test_profiler_calibration.py:223`
- `tests/unit/test_profiler_calibration.py:259`
- `tests/unit/test_profiler_calibration.py:301`

**Suggested Action**: [TO BE DETERMINED]

### string: calibrate_system...
**Occurrences**: 3

- `tests/unit/test_profiler_calibration.py:376`
- `tests/unit/test_profiler_calibration.py:392`
- `tests/unit/test_profiler_calibration.py:403`

**Suggested Action**: [TO BE DETERMINED]

### string: System overloaded...
**Occurrences**: 3

- `tests/unit/test_profiler_calibration.py:101`
- `tests/unit/test_profiler_calibration.py:94`
- `tests/unit/test_profiler_calibration.py:408`

**Suggested Action**: [TO BE DETERMINED]

### string: subprocess.run...
**Occurrences**: 5

- `tests/unit/test_profiler_ncu.py:117`
- `tests/unit/test_profiler_ncu.py:125`
- `tests/unit/test_profiler_ncu.py:211`
- `tests/unit/test_profiler_ncu.py:246`
- `tests/unit/test_profiler_ncu.py:275`

**Suggested Action**: [TO BE DETERMINED]

### number: 0.8
**Occurrences**: 24

- `audit_code_duplication.py:33`
- `test_basic_functionality.py:21`
- `test_basic_functionality.py:22`
- `scripts/check_coverage.py:125`
- `scripts/generate_figures.py:36`
- ... and 19 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 50
**Occurrences**: 29

- `audit_code_duplication.py:390`
- `audit_code_duplication.py:277`
- `check_syntax.py:28`
- `check_syntax.py:50`
- `test_basic_functionality.py:156`
- ... and 24 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 30
**Occurrences**: 7

- `audit_code_duplication.py:471`
- `scripts/implementation_demo.py:20`
- `tests/integration/test_cli_full_pipeline.py:441`
- `tests/integration/test_cli_full_pipeline.py:472`
- `tests/integration/test_profiler_integration.py:428`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 3
**Occurrences**: 193

- `audit_code_duplication.py:87`
- `audit_code_duplication.py:301`
- `test_basic_functionality.py:37`
- `test_basic_functionality.py:77`
- `test_basic_functionality.py:129`
- ... and 188 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 5
**Occurrences**: 66

- `audit_code_duplication.py:252`
- `audit_code_duplication.py:435`
- `audit_code_duplication.py:417`
- `audit_code_duplication.py:436`
- `scripts/generate_figures.py:448`
- ... and 61 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 20
**Occurrences**: 13

- `audit_code_duplication.py:392`
- `audit_code_duplication.py:240`
- `scripts/generate_figures.py:342`
- `scripts/generate_figures.py:348`
- `scripts/implementation_demo.py:131`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 60
**Occurrences**: 14

- `audit_code_duplication.py:445`
- `main.py:275`
- `main.py:283`
- `main.py:273`
- `scripts/check_coverage.py:142`
- ... and 9 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 15
**Occurrences**: 6

- `audit_code_duplication.py:473`
- `scripts/generate_figures.py:381`
- `scripts/generate_figures.py:350`
- `scripts/generate_figures.py:458`
- `scripts/generate_figures.py:451`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 10
**Occurrences**: 91

- `audit_code_duplication.py:253`
- `audit_code_duplication.py:276`
- `scripts/generate_correlation_matrix.py:111`
- `scripts/generate_figures.py:298`
- `scripts/generate_figures.py:143`
- ... and 86 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 200
**Occurrences**: 4

- `audit_code_duplication.py:125`
- `audit_code_duplication.py:125`
- `tests/integration/test_model_integration.py:39`
- `tests/unit/test_dataset_manager.py:60`

**Suggested Action**: [TO BE DETERMINED]

### number: 100
**Occurrences**: 55

- `audit_code_duplication.py:244`
- `audit_code_duplication.py:244`
- `scripts/check_coverage.py:89`
- `scripts/check_coverage.py:95`
- `scripts/check_coverage.py:128`
- ... and 50 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 4
**Occurrences**: 202

- `main.py:151`
- `test_basic_functionality.py:36`
- `test_basic_functionality.py:83`
- `test_basic_functionality.py:58`
- `test_basic_functionality.py:109`
- ... and 197 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 8
**Occurrences**: 108

- `main.py:151`
- `scripts/generate_figures.py:457`
- `scripts/generate_figures.py:215`
- `scripts/performance_report.py:60`
- `scripts/performance_report.py:67`
- ... and 103 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 6
**Occurrences**: 20

- `test_basic_functionality.py:121`
- `test_basic_functionality.py:121`
- `test_basic_functionality.py:137`
- `test_basic_functionality.py:109`
- `scripts/generate_figures.py:143`
- ... and 15 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 42
**Occurrences**: 33

- `test_basic_functionality.py:28`
- `test_basic_functionality.py:127`
- `scripts/generate_figures.py:286`
- `scripts/generate_figures.py:340`
- `scripts/quick_start_demo.py:61`
- ... and 28 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.5
**Occurrences**: 46

- `test_basic_functionality.py:32`
- `scripts/generate_figures.py:163`
- `scripts/performance_report.py:253`
- `scripts/performance_report.py:254`
- `scripts/quick_start_demo.py:50`
- ... and 41 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.1
**Occurrences**: 57

- `test_basic_functionality.py:129`
- `test_basic_functionality.py:21`
- `test_basic_functionality.py:22`
- `test_basic_functionality.py:23`
- `test_basic_functionality.py:24`
- ... and 52 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.2
**Occurrences**: 26

- `test_basic_functionality.py:21`
- `test_basic_functionality.py:22`
- `test_basic_functionality.py:23`
- `test_basic_functionality.py:24`
- `tests/integration/test_cli_full_pipeline.py:529`
- ... and 21 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.9
**Occurrences**: 13

- `test_basic_functionality.py:23`
- `test_basic_functionality.py:24`
- `scripts/generate_figures.py:342`
- `scripts/generate_figures.py:366`
- `scripts/generate_figures.py:256`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 40
**Occurrences**: 9

- `scripts/check_coverage.py:215`
- `scripts/check_coverage.py:232`
- `scripts/implementation_demo.py:59`
- `scripts/quick_start_demo.py:97`
- `scripts/quick_start_demo.py:136`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1000
**Occurrences**: 22

- `scripts/generate_correlation_matrix.py:31`
- `scripts/generate_figures.py:451`
- `src/co_design/correlation.py:45`
- `src/co_design/iasp.py:57`
- `src/co_design/ptq.py:32`
- ... and 17 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 4096
**Occurrences**: 7

- `scripts/generate_correlation_matrix.py:32`
- `scripts/generate_figures.py:442`
- `src/co_design/spectral.py:66`
- `src/utils/config.py:36`
- `tests/performance/test_benchmarks.py:599`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 12
**Occurrences**: 16

- `scripts/generate_figures.py:26`
- `scripts/generate_figures.py:31`
- `scripts/generate_figures.py:32`
- `scripts/generate_figures.py:33`
- `scripts/performance_report.py:32`
- ... and 11 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 16
**Occurrences**: 76

- `scripts/generate_figures.py:30`
- `scripts/generate_figures.py:404`
- `scripts/performance_report.py:52`
- `scripts/performance_report.py:107`
- `scripts/performance_report.py:142`
- ... and 71 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1.2
**Occurrences**: 4

- `scripts/generate_figures.py:35`
- `scripts/generate_figures.py:38`
- `scripts/generate_figures.py:39`
- `scripts/generate_figures.py:40`

**Suggested Action**: [TO BE DETERMINED]

### number: 150
**Occurrences**: 7

- `scripts/generate_figures.py:41`
- `scripts/performance_report.py:38`
- `scripts/performance_report.py:100`
- `scripts/performance_report.py:135`
- `scripts/performance_report.py:202`
- ... and 2 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 300
**Occurrences**: 6

- `scripts/generate_figures.py:42`
- `scripts/run_experiment.py:118`
- `src/co_design/spectral.py:224`
- `src/profiler/ncu.py:41`
- `src/utils/profiler.py:135`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 95
**Occurrences**: 4

- `scripts/generate_figures.py:381`
- `scripts/generate_figures.py:350`
- `src/profiler/latency.py:338`
- `src/utils/profiler.py:405`

**Suggested Action**: [TO BE DETERMINED]

### number: 128
**Occurrences**: 40

- `scripts/generate_figures.py:442`
- `scripts/generate_figures.py:447`
- `scripts/generate_figures.py:460`
- `src/co_design/ptq.py:85`
- `src/models/gcn_model.py:18`
- ... and 35 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 256
**Occurrences**: 20

- `scripts/generate_figures.py:442`
- `src/models/gcn_model.py:19`
- `src/profiler/calibration.py:222`
- `src/profiler/calibration.py:223`
- `src/utils/dataset_manager.py:280`
- ... and 15 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 512
**Occurrences**: 27

- `scripts/generate_figures.py:442`
- `scripts/quick_start_demo.py:33`
- `scripts/run_experiment.py:294`
- `src/profiler/calibration.py:250`
- `src/profiler/calibration.py:222`
- ... and 22 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1024
**Occurrences**: 51

- `scripts/generate_figures.py:442`
- `src/co_design/correlation.py:156`
- `src/co_design/correlation.py:165`
- `src/co_design/correlation.py:168`
- `src/co_design/correlation.py:366`
- ... and 46 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 2048
**Occurrences**: 12

- `scripts/generate_figures.py:442`
- `src/co_design/spectral.py:67`
- `tests/integration/test_real_models.py:559`
- `tests/integration/test_real_models.py:576`
- `tests/integration/test_real_models.py:545`
- ... and 7 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.3
**Occurrences**: 28

- `scripts/generate_figures.py:158`
- `scripts/generate_figures.py:255`
- `scripts/generate_figures.py:376`
- `scripts/generate_figures.py:467`
- `scripts/generate_figures.py:439`
- ... and 23 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.7
**Occurrences**: 20

- `scripts/generate_figures.py:249`
- `scripts/generate_figures.py:360`
- `scripts/generate_figures.py:458`
- `scripts/generate_figures.py:460`
- `scripts/generate_figures.py:168`
- ... and 15 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1.5
**Occurrences**: 6

- `scripts/generate_figures.py:235`
- `tests/performance/test_benchmarks.py:679`
- `tests/unit/test_iasp.py:85`
- `tests/unit/test_iasp.py:79`
- `tests/unit/test_profiler_latency.py:44`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 16.5
**Occurrences**: 5

- `scripts/generate_figures.py:206`
- `tests/unit/test_profiler_calibration.py:281`
- `tests/unit/test_profiler_calibration.py:284`
- `tests/unit/test_profiler_calibration.py:286`
- `tests/unit/test_profiler_calibration.py:287`

**Suggested Action**: [TO BE DETERMINED]

### number: 1e-05
**Occurrences**: 6

- `scripts/quick_start_demo.py:47`
- `src/co_design/hds.py:42`
- `src/utils/config.py:86`
- `tests/integration/test_experiment_pipeline.py:58`
- `tests/integration/test_real_models.py:313`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 15.0
**Occurrences**: 11

- `scripts/run_experiment.py:125`
- `src/profiler/calibration.py:28`
- `tests/unit/test_profiler_calibration.py:27`
- `tests/unit/test_profiler_calibration.py:254`
- `tests/unit/test_profiler_calibration.py:255`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1.1
**Occurrences**: 11

- `src/co_design/correlation.py:417`
- `src/co_design/correlation.py:417`
- `tests/integration/test_iasp_integration.py:265`
- `tests/integration/test_iasp_integration.py:264`
- `tests/unit/test_correlation.py:114`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.001
**Occurrences**: 11

- `src/co_design/correlation.py:409`
- `src/co_design/correlation.py:413`
- `tests/integration/test_hds_ptq_integration.py:46`
- `tests/integration/test_iasp_integration.py:258`
- `tests/integration/test_iasp_integration.py:261`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.0001
**Occurrences**: 4

- `src/co_design/hds.py:32`
- `src/co_design/hds.py:43`
- `tests/integration/test_cli_full_pipeline.py:121`
- `tests/unit/test_hds.py:26`

**Suggested Action**: [TO BE DETERMINED]

### number: 32
**Occurrences**: 65

- `src/co_design/hds.py:49`
- `src/co_design/ptq.py:33`
- `src/profiler/calibration.py:225`
- `tests/integration/test_cli_full_pipeline.py:157`
- `tests/integration/test_cli_full_pipeline.py:98`
- ... and 60 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1e-08
**Occurrences**: 13

- `src/co_design/hds.py:171`
- `src/co_design/hds.py:766`
- `src/co_design/hds.py:104`
- `src/co_design/hds.py:104`
- `src/co_design/hds.py:779`
- ... and 8 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 64
**Occurrences**: 68

- `src/co_design/iasp.py:56`
- `src/co_design/iasp.py:276`
- `src/co_design/iasp.py:339`
- `src/profiler/calibration.py:224`
- `src/profiler/calibration.py:225`
- ... and 63 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 127
**Occurrences**: 4

- `src/co_design/ptq.py:86`
- `src/co_design/ptq.py:85`
- `src/co_design/ptq.py:89`
- `tests/unit/test_ptq.py:61`

**Suggested Action**: [TO BE DETERMINED]

### number: 1e-06
**Occurrences**: 11

- `src/co_design/spectral.py:181`
- `tests/integration/test_cli_full_pipeline.py:418`
- `tests/integration/test_cli_full_pipeline.py:419`
- `tests/integration/test_cli_full_pipeline.py:420`
- `tests/integration/test_iasp_integration.py:269`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 5.0
**Occurrences**: 5

- `src/profiler/calibration.py:32`
- `tests/unit/test_profiler_calibration.py:31`
- `tests/unit/test_profiler_latency.py:318`
- `tests/unit/test_profiler_latency.py:335`
- `tests/unit/test_ptq.py:98`

**Suggested Action**: [TO BE DETERMINED]

### number: 10.0
**Occurrences**: 11

- `src/profiler/latency.py:369`
- `tests/unit/test_profiler_calibration.py:53`
- `tests/unit/test_profiler_latency.py:55`
- `tests/unit/test_profiler_latency.py:78`
- `tests/unit/test_profiler_latency.py:424`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 224
**Occurrences**: 6

- `src/utils/dataset_manager.py:273`
- `src/utils/dataset_manager.py:281`
- `tests/integration/test_real_models.py:338`
- `tests/integration/test_real_models.py:338`
- `tests/integration/test_real_models.py:337`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 0.75
**Occurrences**: 8

- `tests/integration/test_experiment_pipeline.py:178`
- `tests/integration/test_experiment_pipeline.py:247`
- `tests/integration/test_experiment_pipeline.py:287`
- `tests/integration/test_experiment_pipeline.py:333`
- `tests/unit/test_hds.py:41`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 5000
**Occurrences**: 14

- `tests/integration/test_permutation_integration.py:180`
- `tests/integration/test_permutation_integration.py:180`
- `tests/integration/test_permutation_integration.py:193`
- `tests/integration/test_permutation_integration.py:194`
- `tests/integration/test_real_models.py:443`
- ... and 9 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 7
**Occurrences**: 5

- `tests/integration/test_permutation_integration.py:230`
- `tests/integration/test_permutation_integration.py:274`
- `tests/unit/test_apply.py:451`
- `tests/unit/test_profiler_calibration.py:189`
- `tests/unit/test_profiler_calibration.py:179`

**Suggested Action**: [TO BE DETERMINED]

### number: 87.5
**Occurrences**: 6

- `tests/integration/test_profiler_integration.py:264`
- `tests/integration/test_profiler_integration.py:242`
- `tests/unit/test_profiler_ncu.py:207`
- `tests/unit/test_profiler_ncu.py:336`
- `tests/unit/test_profiler_ncu.py:200`
- ... and 1 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 1000000
**Occurrences**: 5

- `tests/integration/test_profiler_integration.py:265`
- `tests/integration/test_profiler_integration.py:243`
- `tests/unit/test_profiler_ncu.py:81`
- `tests/unit/test_profiler_ncu.py:74`
- `tests/unit/test_profiler_ncu.py:201`

**Suggested Action**: [TO BE DETERMINED]

### number: 20.0
**Occurrences**: 11

- `tests/integration/test_profiler_integration.py:292`
- `tests/unit/test_profiler_calibration.py:44`
- `tests/unit/test_profiler_calibration.py:123`
- `tests/unit/test_profiler_calibration.py:37`
- `tests/unit/test_profiler_calibration.py:112`
- ... and 6 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 10.5
**Occurrences**: 10

- `tests/unit/test_profiler_calibration.py:85`
- `tests/unit/test_profiler_calibration.py:77`
- `tests/unit/test_profiler_latency.py:69`
- `tests/unit/test_profiler_latency.py:94`
- `tests/unit/test_profiler_latency.py:53`
- ... and 5 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 11.0
**Occurrences**: 12

- `tests/unit/test_profiler_calibration.py:86`
- `tests/unit/test_profiler_calibration.py:78`
- `tests/unit/test_profiler_latency.py:56`
- `tests/unit/test_profiler_latency.py:59`
- `tests/unit/test_profiler_latency.py:79`
- ... and 7 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 4.8
**Occurrences**: 4

- `tests/unit/test_profiler_calibration.py:87`
- `tests/unit/test_profiler_calibration.py:79`
- `tests/unit/test_profiler_latency.py:60`
- `tests/unit/test_profiler_latency.py:83`

**Suggested Action**: [TO BE DETERMINED]

### number: 16.0
**Occurrences**: 9

- `tests/unit/test_profiler_calibration.py:298`
- `tests/unit/test_profiler_calibration.py:278`
- `tests/unit/test_profiler_calibration.py:282`
- `tests/unit/test_profiler_calibration.py:243`
- `tests/unit/test_profiler_calibration.py:286`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 15.5
**Occurrences**: 9

- `tests/unit/test_profiler_calibration.py:209`
- `tests/unit/test_profiler_calibration.py:218`
- `tests/unit/test_profiler_calibration.py:237`
- `tests/unit/test_profiler_calibration.py:240`
- `tests/unit/test_profiler_calibration.py:280`
- ... and 4 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 21.0
**Occurrences**: 4

- `tests/unit/test_profiler_latency.py:314`
- `tests/unit/test_profiler_latency.py:317`
- `tests/unit/test_profiler_latency.py:319`
- `tests/unit/test_profiler_latency.py:320`

**Suggested Action**: [TO BE DETERMINED]

### number: 12.0
**Occurrences**: 8

- `tests/unit/test_profiler_latency.py:62`
- `tests/unit/test_profiler_latency.py:85`
- `tests/unit/test_profiler_latency.py:406`
- `tests/unit/test_profiler_latency.py:409`
- `tests/unit/test_profiler_latency.py:421`
- ... and 3 more occurrences

**Suggested Action**: [TO BE DETERMINED]

### number: 89.5
**Occurrences**: 4

- `tests/unit/test_profiler_ncu.py:80`
- `tests/unit/test_profiler_ncu.py:94`
- `tests/unit/test_profiler_ncu.py:73`
- `tests/unit/test_profiler_ncu.py:87`

**Suggested Action**: [TO BE DETERMINED]
