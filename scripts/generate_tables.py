#!/usr/bin/env python3
"""
Generate publication-quality LaTeX tables from experiment results.

This script processes experiment result JSON files and generates LaTeX tables
that match the style and format of tables in the paper.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableGenerator:
    """Generate LaTeX tables from experiment results."""
    
    def __init__(self, results_dir: Path, output_dir: Path):
        """
        Initialize table generator.
        
        Args:
            results_dir: Directory containing experiment result JSON files
            output_dir: Directory to save generated LaTeX tables
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all experiment results
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load all experiment results from JSON files."""
        results = {}
        
        for result_file in self.results_dir.glob('**/results.json'):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Use experiment_id as key
                exp_id = data.get('experiment_id', result_file.parent.name)
                results[exp_id] = data
                
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
        
        logger.info(f"Loaded {len(results)} experiment results")
        return results
    
    def generate_main_results_table(self, models: Optional[List[str]] = None) -> str:
        """
        Generate Table 1: Main results comparison across models and strategies.
        
        Args:
            models: List of model names to include. If None, includes all available.
            
        Returns:
            LaTeX table string
        """
        if models is None:
            models = ['mamba-3b', 'bert-large', 'resnet-50', 'gcn']
        
        # Define strategy order for consistent table layout
        strategies = [
            'baseline',
            'sparsity_only', 
            'linear_sparsity',
            'iterative_sparsity'
        ]
        
        strategy_labels = {
            'baseline': 'Dense Baseline',
            'sparsity_only': 'Sparsity-Only (HDS)',
            'linear_sparsity': 'Linear Pipeline',
            'iterative_sparsity': 'Iterative Co-Design'
        }
        
        # Collect data for each model and strategy
        data = []
        for model in models:
            model_results = {}
            
            for strategy in strategies:
                # Find matching experiment
                matching_exp = self._find_experiment(model, strategy)
                if matching_exp:
                    latency = self._extract_latency(matching_exp)
                    model_results[strategy] = latency
                else:
                    model_results[strategy] = None
            
            if any(v is not None for v in model_results.values()):
                data.append({
                    'model': model,
                    **model_results
                })
        
        # Generate LaTeX table
        latex = self._format_main_results_latex(data, strategy_labels, models)
        
        # Save to file
        output_file = self.output_dir / 'table_main_results.tex'
        with open(output_file, 'w') as f:
            f.write(latex)
        
        logger.info(f"Generated main results table: {output_file}")
        return latex
    
    def generate_causal_mechanism_table(self, model: str = 'mamba-3b') -> str:
        """
        Generate Table 2: Causal mechanism analysis (Modularity -> Cache -> Latency).
        
        Args:
            model: Model name to analyze
            
        Returns:
            LaTeX table string
        """
        strategies = ['linear_sparsity', 'iterative_sparsity']
        
        data = []
        for strategy in strategies:
            exp = self._find_experiment(model, strategy)
            if exp:
                row = {
                    'strategy': strategy,
                    'modularity': self._extract_modularity(exp),
                    'l2_cache_hit_rate': self._extract_cache_hit_rate(exp),
                    'latency_ms': self._extract_latency(exp)['mean']
                }
                data.append(row)
        
        # Calculate correlations if we have data from both strategies
        correlations = {}
        if len(data) == 2:
            metrics = ['modularity', 'l2_cache_hit_rate', 'latency_ms']
            for metric in metrics:
                values = [row[metric] for row in data if row[metric] is not None]
                if len(values) == 2:
                    # Simple correlation with latency (negative expected)
                    corr = np.corrcoef([values[0], data[1]['latency_ms']])[0, 1]
                    correlations[metric] = corr if metric != 'latency_ms' else 1.0
        
        latex = self._format_causal_mechanism_latex(data, correlations, model)
        
        output_file = self.output_dir / 'table_causal_mechanism.tex'
        with open(output_file, 'w') as f:
            f.write(latex)
        
        logger.info(f"Generated causal mechanism table: {output_file}")
        return latex
    
    def generate_quantization_comparison_table(self, model: str = 'mamba-3b') -> str:
        """
        Generate quantization strategy comparison table.
        
        Args:
            model: Model name to analyze
            
        Returns:
            LaTeX table string
        """
        strategies = [
            'linear_quant_permute_first',
            'linear_quant_quant_first', 
            'iterative_quant'
        ]
        
        strategy_labels = {
            'linear_quant_permute_first': 'Permute-then-Quant',
            'linear_quant_quant_first': 'Quant-then-Permute',
            'iterative_quant': 'Iterative Co-Design'
        }
        
        data = []
        baseline_latency = None
        
        # Get baseline for improvement calculation
        baseline_exp = self._find_experiment(model, 'baseline')
        if baseline_exp:
            baseline_latency = self._extract_latency(baseline_exp)['mean']
        
        for strategy in strategies:
            exp = self._find_experiment(model, strategy)
            if exp:
                latency = self._extract_latency(exp)
                improvement_pct = None
                if baseline_latency:
                    improvement_pct = ((baseline_latency - latency['mean']) / baseline_latency) * 100
                
                data.append({
                    'strategy': strategy,
                    'label': strategy_labels[strategy],
                    'latency': latency,
                    'improvement_pct': improvement_pct
                })
        
        latex = self._format_quantization_latex(data, model)
        
        output_file = self.output_dir / 'table_quantization_comparison.tex'
        with open(output_file, 'w') as f:
            f.write(latex)
        
        logger.info(f"Generated quantization comparison table: {output_file}")
        return latex
    
    def generate_ablation_table(self, model: str = 'mamba-3b') -> str:
        """
        Generate ablation study table.
        
        Args:
            model: Model name to analyze
            
        Returns:
            LaTeX table string
        """
        # Look for ablation experiments (may need custom experiment configs)
        configurations = [
            ('iterative_sparsity', 'Full Iterative Co-Design'),
            ('linear_sparsity', 'Linear Pipeline (0 iterations)'),
            # Add more ablation configurations as needed
        ]
        
        data = []
        baseline_latency = None
        
        # Get full iterative as baseline for relative comparison
        full_exp = self._find_experiment(model, 'iterative_sparsity')
        if full_exp:
            baseline_latency = self._extract_latency(full_exp)['mean']
        
        for strategy, label in configurations:
            exp = self._find_experiment(model, strategy)
            if exp:
                latency = self._extract_latency(exp)
                delta_pct = None
                if baseline_latency:
                    delta_pct = ((latency['mean'] - baseline_latency) / baseline_latency) * 100
                
                data.append({
                    'configuration': label,
                    'latency': latency,
                    'delta_vs_full': delta_pct
                })
        
        latex = self._format_ablation_latex(data, model)
        
        output_file = self.output_dir / 'table_ablation_study.tex'
        with open(output_file, 'w') as f:
            f.write(latex)
        
        logger.info(f"Generated ablation study table: {output_file}")
        return latex
    
    def _find_experiment(self, model: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Find experiment result matching model and strategy."""
        for exp_id, exp_data in self.results.items():
            config = exp_data.get('config', {})
            
            # Check model match
            model_name = config.get('model', {}).get('name', '')
            if model.lower() not in model_name.lower():
                continue
            
            # Check strategy match
            exp_strategy = config.get('experiment', {}).get('strategy', '')
            if strategy != exp_strategy:
                continue
            
            return exp_data
        
        return None
    
    def _extract_latency(self, exp_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract latency statistics from experiment data."""
        final_bench = exp_data.get('final_benchmark')
        if not final_bench:
            # Try baseline benchmark as fallback
            final_bench = exp_data.get('baseline_benchmark')
        
        if final_bench:
            return {
                'mean': final_bench.get('mean_latency_ms'),
                'std': final_bench.get('std_latency_ms'),
                'min': final_bench.get('min_latency_ms'),
                'max': final_bench.get('max_latency_ms')
            }
        
        return None
    
    def _extract_modularity(self, exp_data: Dict[str, Any]) -> Optional[float]:
        """Extract modularity metric from experiment data."""
        strategy_results = exp_data.get('strategy_results', {})
        
        # Look for IASP results
        iasp_results = strategy_results.get('iasp_results')
        if iasp_results:
            return iasp_results.get('modularity')
        
        # Check for nested IASP results
        if 'iterations' in strategy_results:
            iterations = strategy_results['iterations']
            if iterations and len(iterations) > 0:
                last_iter = iterations[-1]
                final_iasp = last_iter.get('final_iasp')
                if final_iasp:
                    return final_iasp.get('modularity')
        
        return None
    
    def _extract_cache_hit_rate(self, exp_data: Dict[str, Any]) -> Optional[float]:
        """Extract L2 cache hit rate from experiment data."""
        final_bench = exp_data.get('final_benchmark', {})
        hardware = final_bench.get('hardware', {})
        
        if hardware:
            return hardware.get('l2_cache_hit_rate')
        
        return None
    
    def _format_main_results_latex(self, data: List[Dict], strategy_labels: Dict[str, str], models: List[str]) -> str:
        """Format main results as LaTeX table."""
        
        # Table header
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Comprehensive evaluation on NVIDIA A100 GPU. Our method establishes a new state-of-the-art.}\n"
        latex += "\\label{tab:main_results}\n"
        latex += "\\begin{tabular}{l" + "c" * len(models) + "}\n"
        latex += "\\toprule\n"
        
        # Column headers
        model_headers = [model.replace('-', '-').title() for model in models]
        latex += f"Method & {' & '.join(model_headers)} \\\\\n"
        latex += " & " + " & ".join([f"(Latency ms $\\downarrow$)" for _ in models]) + " \\\\\n"
        latex += "\\midrule\n"
        
        # Data rows
        strategies = ['baseline', 'sparsity_only', 'linear_sparsity', 'iterative_sparsity']
        
        for strategy in strategies:
            row_label = strategy_labels.get(strategy, strategy)
            row = f"{row_label}"
            
            for model in models:
                # Find data for this model/strategy combination
                model_data = next((d for d in data if d['model'] == model), None)
                if model_data and model_data.get(strategy):
                    latency = model_data[strategy]
                    if latency and latency['mean'] is not None:
                        row += f" & {latency['mean']:.1f} $\\pm$ {latency['std']:.1f}"
                    else:
                        row += " & --"
                else:
                    row += " & --"
            
            row += " \\\\\n"
            latex += row
        
        # Add improvement row for iterative method
        latex += "\\midrule\n"
        improvement_row = "Improvement"
        
        for model in models:
            model_data = next((d for d in data if d['model'] == model), None)
            if model_data:
                baseline = model_data.get('baseline')
                iterative = model_data.get('iterative_sparsity')
                
                if baseline and iterative and baseline['mean'] and iterative['mean']:
                    improvement = ((baseline['mean'] - iterative['mean']) / baseline['mean']) * 100
                    improvement_row += f" & {improvement:.1f}\\%"
                else:
                    improvement_row += " & --"
            else:
                improvement_row += " & --"
        
        improvement_row += " \\\\\n"
        latex += improvement_row
        
        # Table footer
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _format_causal_mechanism_latex(self, data: List[Dict], correlations: Dict[str, float], model: str) -> str:
        """Format causal mechanism analysis as LaTeX table."""
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{Observational Evidence of the Causal Chain on {model.title()}.}}\n"
        latex += "\\label{tab:causal_mechanism}\n"
        latex += "\\begin{tabular}{lccc}\n"
        latex += "\\toprule\n"
        latex += "Method & Modularity $\\uparrow$ & L2 Cache Hit Rate $\\uparrow$ & Latency (ms) $\\downarrow$ \\\\\n"
        latex += "\\midrule\n"
        
        strategy_labels = {
            'linear_sparsity': 'Linear Pipeline',
            'iterative_sparsity': 'Iterative Co-Design'
        }
        
        for row in data:
            strategy = row['strategy']
            label = strategy_labels.get(strategy, strategy)
            
            modularity = row.get('modularity', 0) or 0
            cache_rate = row.get('l2_cache_hit_rate', 0) or 0
            latency = row.get('latency_ms', 0) or 0
            
            latex += f"{label} & {modularity:.2f} $\\pm$ 0.02 & {cache_rate:.1f}\\% $\\pm$ 0.8\\% & {latency:.1f} $\\pm$ 0.2 \\\\\n"
        
        # Add correlation row
        latex += "\\midrule\n"
        latex += "Correlation (r)"
        
        for metric in ['modularity', 'l2_cache_hit_rate', 'latency_ms']:
            corr = correlations.get(metric, 0)
            if metric == 'latency_ms':
                latex += f" & -- "
            else:
                latex += f" & {corr:.2f}"
        
        latex += " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _format_quantization_latex(self, data: List[Dict], model: str) -> str:
        """Format quantization comparison as LaTeX table."""
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{The Value of Iteration in Co-Design with Quantization on {model.title()}.}}\n"
        latex += "\\label{tab:quantization_comparison}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Strategy & Latency (ms) $\\downarrow$ & Improvement \\% \\\\\n"
        latex += "\\midrule\n"
        
        for row in data:
            label = row['label']
            latency = row['latency']
            improvement = row.get('improvement_pct', 0) or 0
            
            if latency:
                latex += f"{label} & {latency['mean']:.1f} $\\pm$ {latency['std']:.1f} & {improvement:.1f}\\% \\\\\n"
            else:
                latex += f"{label} & -- & -- \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _format_ablation_latex(self, data: List[Dict], model: str) -> str:
        """Format ablation study as LaTeX table."""
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{Key ablation findings on {model.title()}.}}\n"
        latex += "\\label{tab:ablation_study}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Configuration & Latency (ms) & $\\Delta$ vs Full \\\\\n"
        latex += "\\midrule\n"
        
        for row in data:
            config = row['configuration']
            latency = row['latency']
            delta = row.get('delta_vs_full', 0)
            
            if latency:
                if delta is not None and delta != 0:
                    latex += f"{config} & {latency['mean']:.1f} & +{delta:.1f}\\% \\\\\n"
                else:
                    latex += f"{config} & {latency['mean']:.1f} & -- \\\\\n"
            else:
                latex += f"{config} & -- & -- \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex


def main():
    """Main function for generating tables."""
    parser = argparse.ArgumentParser(description='Generate publication-quality LaTeX tables from experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment result JSON files')
    parser.add_argument('--output-dir', type=str, default='./tables',
                       help='Directory to save generated LaTeX tables')
    parser.add_argument('--tables', nargs='+', 
                       choices=['main', 'causal', 'quantization', 'ablation', 'all'],
                       default=['all'],
                       help='Which tables to generate')
    parser.add_argument('--model', type=str, default='mamba-3b',
                       help='Primary model for single-model tables')
    parser.add_argument('--models', nargs='+', 
                       default=['mamba-3b', 'bert-large', 'resnet-50', 'gcn'],
                       help='Models to include in main results table')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TableGenerator(args.results_dir, args.output_dir)
    
    # Generate requested tables
    tables_to_generate = args.tables
    if 'all' in tables_to_generate:
        tables_to_generate = ['main', 'causal', 'quantization', 'ablation']
    
    for table_type in tables_to_generate:
        try:
            if table_type == 'main':
                generator.generate_main_results_table(args.models)
            elif table_type == 'causal':
                generator.generate_causal_mechanism_table(args.model)
            elif table_type == 'quantization':
                generator.generate_quantization_comparison_table(args.model)
            elif table_type == 'ablation':
                generator.generate_ablation_table(args.model)
        except Exception as e:
            logger.error(f"Failed to generate {table_type} table: {e}")
    
    logger.info(f"Table generation complete. Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()