#!/usr/bin/env python3
"""
Performance report generation for T-004 benchmarks.

This script generates comprehensive performance reports with charts,
regression detection, and trend analysis for the iterative co-design framework.
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceReportGenerator:
    """Generate comprehensive performance reports and visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib for better output
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.dpi': 150
        })
    
    def generate_scaling_analysis_chart(self, results: Dict, title: str = "Scaling Analysis") -> str:
        """
        Generate scaling analysis visualization.
        
        Args:
            results: Benchmark results with size -> timing data
            title: Chart title
            
        Returns:
            str: Path to generated chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract data
        sizes = sorted(results.keys())
        times = [results[size]['mean_time_ms'] for size in sizes]
        stds = [results[size].get('std_time_ms', 0) for size in sizes]
        
        # Linear scale plot
        ax1.errorbar(sizes, times, yerr=stds, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'{title} - Linear Scale')
        ax1.grid(True, alpha=0.3)
        
        # Log-log scale plot for scaling analysis
        ax2.loglog(sizes, times, marker='s', linewidth=2, markersize=8, label='Actual')
        
        # Fit theoretical scaling curves
        if len(sizes) >= 3:
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            coeffs = np.polyfit(log_sizes, log_times, 1)
            scaling_exp = coeffs[0]
            
            # Generate theoretical curves
            size_range = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
            
            # Quadratic scaling
            quad_times = (size_range / sizes[0]) ** 2 * times[0]
            ax2.loglog(size_range, quad_times, '--', alpha=0.7, label='O(n²)')
            
            # Cubic scaling
            cubic_times = (size_range / sizes[0]) ** 3 * times[0]
            ax2.loglog(size_range, cubic_times, '--', alpha=0.7, label='O(n³)')
            
            # Fitted curve
            fitted_times = np.exp(coeffs[1]) * size_range ** scaling_exp
            ax2.loglog(size_range, fitted_times, ':', linewidth=3, 
                      label=f'Fitted O(n^{scaling_exp:.2f})')
        
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title(f'{title} - Log-Log Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / f'scaling_analysis_{title.lower().replace(" ", "_")}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def generate_memory_analysis_chart(self, results: Dict) -> str:
        """Generate memory usage analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sizes = sorted(results.keys())
        peak_memory = [results[size].get('peak_memory_mb', 0) for size in sizes]
        theoretical_memory = [results[size].get('theoretical_memory_mb', 0) for size in sizes]
        efficiency = [results[size].get('memory_efficiency', 1) for size in sizes]
        
        # Memory usage comparison
        ax1.plot(sizes, peak_memory, 'o-', linewidth=2, markersize=8, label='Actual Peak')
        ax1.plot(sizes, theoretical_memory, 's--', linewidth=2, markersize=8, 
                label='Theoretical Minimum', alpha=0.7)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage vs Matrix Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory efficiency
        ax2.plot(sizes, efficiency, 'd-', linewidth=2, markersize=8, color='orange')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Theoretical Minimum')
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Memory Efficiency Factor')
        ax2.set_title('Memory Efficiency (Actual / Theoretical)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / 'memory_analysis.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def generate_comparative_analysis_chart(self, results: Dict) -> str:
        """Generate comparative performance analysis chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Spectral vs Random comparison
        if 'spectral_vs_random' in results:
            data = results['spectral_vs_random']
            sizes = sorted(data.keys())
            overhead_factors = [data[size]['overhead_factor'] for size in sizes]
            
            ax1.semilogy(sizes, overhead_factors, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Matrix Size')
            ax1.set_ylabel('Overhead Factor (log scale)')
            ax1.set_title('Spectral vs Random Permutation Overhead')
            ax1.grid(True, alpha=0.3)
        
        # Dense vs Sparse comparison
        if 'dense_vs_sparse' in results:
            data = results['dense_vs_sparse']
            sparsities = sorted(data.keys())
            speedups = [data[sp]['speedup_factor'] for sp in sparsities]
            
            ax2.plot(sparsities, speedups, 's-', linewidth=2, markersize=8, color='green')
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            ax2.set_xlabel('Sparsity Ratio')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Dense vs Sparse Operations Speedup')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # CPU vs GPU comparison
        if 'cpu_vs_gpu' in results:
            data = results['cpu_vs_gpu']
            sizes = sorted(data.keys())
            gpu_speedups = [data[size]['gpu_speedup'] for size in sizes]
            
            ax3.plot(sizes, gpu_speedups, '^-', linewidth=2, markersize=8, color='purple')
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            ax3.set_xlabel('Matrix Size')
            ax3.set_ylabel('GPU Speedup Factor')
            ax3.set_title('CPU vs GPU Performance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Block approximation benefits
        if 'block_approximation' in results:
            data = results['block_approximation']
            sizes = sorted(data.keys())
            speedups = [data[size].get('speedup_factor', 1) for size in sizes if data[size].get('speedup_factor')]
            valid_sizes = [size for size in sizes if data[size].get('speedup_factor')]
            
            if valid_sizes:
                ax4.bar(range(len(valid_sizes)), speedups, alpha=0.7, color='coral')
                ax4.set_xticks(range(len(valid_sizes)))
                ax4.set_xticklabels(valid_sizes)
                ax4.set_xlabel('Matrix Size')
                ax4.set_ylabel('Speedup Factor')
                ax4.set_title('Block Approximation Benefits')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / 'comparative_analysis.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def generate_regression_analysis(self, current_results: Dict, baseline_results: Dict) -> str:
        """Generate performance regression analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract comparable metrics
        regressions = []
        improvements = []
        metrics = []
        
        def compare_recursively(current, baseline, path=""):
            if isinstance(current, dict) and isinstance(baseline, dict):
                for key in current:
                    if key in baseline and key.endswith('_time_ms'):
                        current_val = current[key]
                        baseline_val = baseline[key]
                        
                        change_pct = (current_val - baseline_val) / baseline_val * 100
                        metric_name = f"{path}.{key}" if path else key
                        
                        if change_pct > 5:  # >5% slower
                            regressions.append((metric_name, change_pct))
                        elif change_pct < -5:  # >5% faster
                            improvements.append((metric_name, abs(change_pct)))
                        
                        metrics.append((metric_name, change_pct))
                    elif isinstance(current[key], dict):
                        compare_recursively(current[key], baseline.get(key, {}), 
                                          f"{path}.{key}" if path else key)
        
        compare_recursively(current_results, baseline_results)
        
        # Create regression chart
        if metrics:
            metric_names = [m[0] for m in metrics]
            changes = [m[1] for m in metrics]
            
            colors = ['red' if c > 5 else 'green' if c < -5 else 'blue' for c in changes]
            
            y_pos = np.arange(len(metric_names))
            bars = ax.barh(y_pos, changes, color=colors, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metric_names, fontsize=8)
            ax.set_xlabel('Performance Change (%)')
            ax.set_title('Performance Regression Analysis\n(Red: Regression, Green: Improvement, Blue: Stable)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='5% regression threshold')
            ax.axvline(x=-5, color='green', linestyle='--', alpha=0.5, label='5% improvement threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / 'regression_analysis.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def generate_summary_report(self, all_results: Dict, baseline_results: Optional[Dict] = None) -> str:
        """Generate comprehensive summary report."""
        report_lines = [
            "# Performance Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Analyze overall performance characteristics
        total_tests = 0
        passed_tests = 0
        
        def analyze_results(results, section_name=""):
            nonlocal total_tests, passed_tests
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key.endswith('_time_ms') and isinstance(value, (int, float)):
                        total_tests += 1
                        if value < 10000:  # 10s threshold
                            passed_tests += 1
                    elif isinstance(value, dict):
                        analyze_results(value, f"{section_name}.{key}" if section_name else key)
        
        analyze_results(all_results)
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            report_lines.extend([
                f"- **Total performance tests**: {total_tests}",
                f"- **Tests meeting timing requirements**: {passed_tests} ({success_rate:.1f}%)",
                ""
            ])
        
        # Algorithm scaling analysis
        if 'algorithmic_benchmarks' in all_results:
            report_lines.extend([
                "## Algorithmic Performance Analysis",
                "",
                "### Spectral Clustering Scaling",
                ""
            ])
            
            scaling_data = all_results['algorithmic_benchmarks'].get('spectral_scaling', {})
            if scaling_data:
                sizes = sorted(scaling_data.keys())
                times = [scaling_data[size]['mean_time_ms'] for size in sizes]
                
                # Analyze scaling trend
                if len(sizes) >= 2:
                    log_sizes = np.log(sizes)
                    log_times = np.log(times)
                    scaling_exp = np.polyfit(log_sizes, log_times, 1)[0]
                    
                    report_lines.extend([
                        f"- **Scaling exponent**: {scaling_exp:.2f}",
                        f"- **Scaling classification**: {'Good (≤2)' if scaling_exp <= 2 else 'Concerning (>2)'}",
                        ""
                    ])
        
        # Memory analysis
        if 'memory_analysis' in all_results:
            report_lines.extend([
                "### Memory Usage Analysis",
                ""
            ])
            
            memory_data = all_results['memory_analysis']
            if memory_data:
                avg_efficiency = np.mean([data.get('memory_efficiency', 1) for data in memory_data.values()])
                report_lines.extend([
                    f"- **Average memory efficiency**: {avg_efficiency:.2f}x theoretical",
                    f"- **Memory efficiency rating**: {'Good (<3x)' if avg_efficiency < 3 else 'Needs optimization (≥3x)'}",
                    ""
                ])
        
        # Comparative analysis summary
        if 'comparative_analysis' in all_results:
            comp_data = all_results['comparative_analysis']
            report_lines.extend([
                "## Comparative Analysis Summary",
                ""
            ])
            
            if 'spectral_vs_random' in comp_data:
                overhead_factors = [data['overhead_factor'] for data in comp_data['spectral_vs_random'].values()]
                avg_overhead = np.mean(overhead_factors)
                report_lines.extend([
                    f"- **Spectral vs Random overhead**: {avg_overhead:.1f}x average",
                    ""
                ])
            
            if 'cpu_vs_gpu' in comp_data:
                speedups = [data['gpu_speedup'] for data in comp_data['cpu_vs_gpu'].values()]
                avg_speedup = np.mean(speedups)
                report_lines.extend([
                    f"- **GPU vs CPU speedup**: {avg_speedup:.1f}x average",
                    ""
                ])
        
        # Regression analysis
        if baseline_results:
            report_lines.extend([
                "## Regression Analysis",
                ""
            ])
            
            # Count regressions and improvements
            regressions = 0
            improvements = 0
            
            def count_changes(current, baseline, path=""):
                nonlocal regressions, improvements
                if isinstance(current, dict) and isinstance(baseline, dict):
                    for key in current:
                        if key in baseline and key.endswith('_time_ms'):
                            change_pct = (current[key] - baseline[key]) / baseline[key] * 100
                            if change_pct > 5:
                                regressions += 1
                            elif change_pct < -5:
                                improvements += 1
                        elif isinstance(current[key], dict):
                            count_changes(current[key], baseline.get(key, {}))
            
            count_changes(all_results, baseline_results)
            
            report_lines.extend([
                f"- **Performance regressions detected**: {regressions}",
                f"- **Performance improvements detected**: {improvements}",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if total_tests > 0 and passed_tests / total_tests < 0.8:
            report_lines.append("- ⚠️  Consider optimizing algorithms with poor timing performance")
        
        if 'memory_analysis' in all_results:
            memory_data = all_results['memory_analysis']
            if any(data.get('memory_efficiency', 1) > 5 for data in memory_data.values()):
                report_lines.append("- ⚠️  High memory overhead detected - consider memory optimizations")
        
        if regressions > improvements:
            report_lines.append("- ⚠️  More regressions than improvements - investigate recent changes")
        
        report_lines.extend([
            "- ✅ Continue monitoring performance trends",
            "- ✅ Update baselines after verified improvements",
            ""
        ])
        
        # Write report
        report_path = self.output_dir / 'performance_summary.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return str(report_path)


def main():
    """Main function for performance report generation."""
    parser = argparse.ArgumentParser(description='Generate performance benchmark reports')
    parser.add_argument('--results-file', type=str, required=True,
                       help='JSON file containing benchmark results')
    parser.add_argument('--baseline-file', type=str,
                       help='JSON file containing baseline results for regression analysis')
    parser.add_argument('--output-dir', type=str, default='performance_reports',
                       help='Output directory for reports and charts')
    parser.add_argument('--charts', nargs='+', 
                       choices=['scaling', 'memory', 'comparative', 'regression', 'all'],
                       default=['all'], help='Types of charts to generate')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    baseline_results = None
    if args.baseline_file:
        try:
            with open(args.baseline_file, 'r') as f:
                baseline_results = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Baseline file {args.baseline_file} not found")
    
    # Generate reports
    generator = PerformanceReportGenerator(args.output_dir)
    
    generated_files = []
    
    # Generate requested charts
    if 'all' in args.charts or 'scaling' in args.charts:
        if 'algorithmic_benchmarks' in results:
            scaling_data = results['algorithmic_benchmarks'].get('spectral_scaling', {})
            if scaling_data:
                chart_path = generator.generate_scaling_analysis_chart(
                    scaling_data, "Spectral Clustering Performance"
                )
                generated_files.append(chart_path)
    
    if 'all' in args.charts or 'memory' in args.charts:
        if 'memory_analysis' in results:
            chart_path = generator.generate_memory_analysis_chart(results['memory_analysis'])
            generated_files.append(chart_path)
    
    if 'all' in args.charts or 'comparative' in args.charts:
        if 'comparative_analysis' in results:
            chart_path = generator.generate_comparative_analysis_chart(results['comparative_analysis'])
            generated_files.append(chart_path)
    
    if ('all' in args.charts or 'regression' in args.charts) and baseline_results:
        chart_path = generator.generate_regression_analysis(results, baseline_results)
        generated_files.append(chart_path)
    
    # Generate summary report
    report_path = generator.generate_summary_report(results, baseline_results)
    generated_files.append(report_path)
    
    print("Performance report generation completed!")
    print(f"Generated {len(generated_files)} files:")
    for file_path in generated_files:
        print(f"  - {file_path}")


if __name__ == '__main__':
    main()