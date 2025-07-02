#!/usr/bin/env python
"""
Sequence Length Comparison Script

This script runs quick evaluations with different min_seq_len values
and generates a comparison table to find the optimal bias-variance tradeoff.

Usage:
    python scripts/compare_seq_lengths.py

The script will:
1. Run evaluations with different min_seq_len values
2. Generate a table showing:
   - min_seq_len
   - pad_shorter_sequences (True/False)
   - sample yield (count/percentage)
   - average padding percentage
   - perplexity

This helps identify the configuration with optimal bias-variance tradeoff.
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime

# Configuration for the tests
TEST_CONFIGS = [
    {"min_seq_len": 64, "pad_shorter_sequences": True},
    {"min_seq_len": 128, "pad_shorter_sequences": True},
    {"min_seq_len": 256, "pad_shorter_sequences": True},
    {"min_seq_len": 384, "pad_shorter_sequences": True},
    {"min_seq_len": 64, "pad_shorter_sequences": False},
    {"min_seq_len": 128, "pad_shorter_sequences": False},
    {"min_seq_len": 256, "pad_shorter_sequences": False},
    {"min_seq_len": 512, "pad_shorter_sequences": False},  # Only full-length sequences
]

def run_test(min_seq_len, pad_shorter):
    """Run a single test with the given config and return the results."""
    print(f"\n\n{'='*50}")
    print(f"Running test: min_seq_len={min_seq_len}, pad_shorter={pad_shorter}")
    print(f"{'='*50}")
    
    # Use a unique output directory for each test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"outputs/seq_len_test_{min_seq_len}_{pad_shorter}_{timestamp}"
    
    # Build the command
    cmd = [
        "python", "scripts/run_experiment.py",
        f"dataset=wikitext103_quick",
        "method=dense",
        f"dataset.min_seq_len={min_seq_len}",
        f"dataset.pad_shorter_sequences={str(pad_shorter).lower()}",
        f"hydra.run.dir={outdir}"
    ]
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running test: {result.stderr}")
        return None
    
    # Find the results file
    results_file = Path(outdir) / "dense_metrics.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    # Load and return the results
    with open(results_file, "r") as f:
        return json.load(f)

def extract_metrics(results):
    """Extract relevant metrics from the results dictionary."""
    # Default values if results are None or incomplete
    metrics = {
        "perplexity": None,
        "eval_steps": 0,
        "padded_samples": 0,
        "full_length_samples": 0,
        "padding_percentage": 0,
        "avg_padding_ratio": 0,
        "total_samples": 0  # Ensure this is always present
    }
    
    # Early return if results are completely missing
    if results is None:
        return metrics
    
    # Update with actual values if available
    metrics.update({
        "perplexity": results.get("perplexity"),
        "eval_steps": results.get("eval_steps", 0),
    })
    
    # Extract padding stats if available
    padding_stats = results.get("padding_stats", {})
    padded_samples = padding_stats.get("padded_samples", 0)
    full_length_samples = padding_stats.get("full_length_samples", 0)
    
    metrics.update({
        "padded_samples": padded_samples,
        "full_length_samples": full_length_samples,
        "padding_percentage": padding_stats.get("padding_percentage", 0),
        "avg_padding_ratio": padding_stats.get("avg_padding_ratio", 0),
        "min_seq_len": padding_stats.get("min_seq_len", 0),
    })
    
    # Calculate total samples
    metrics["total_samples"] = padded_samples + full_length_samples
    
    return metrics

def main():
    """Run the tests and generate the comparison table."""
    results = []
    
    for config in TEST_CONFIGS:
        # Run the test
        min_seq_len = config["min_seq_len"]
        pad_shorter = config["pad_shorter_sequences"]
        
        test_results = run_test(min_seq_len, pad_shorter)
        metrics = extract_metrics(test_results)
        
        # Add config parameters to metrics
        metrics.update({
            "min_seq_len": min_seq_len,
            "pad_shorter_sequences": pad_shorter,
        })
        
        results.append(metrics)
    
    # Convert to DataFrame for easy formatting
    df = pd.DataFrame(results)
    
    # Calculate sample yield as percentage of total possible
    max_samples = max([r["total_samples"] for r in results]) if results else 0
    if max_samples > 0:
        df["sample_yield"] = (df["total_samples"] / max_samples * 100).round(1)
    else:
        df["sample_yield"] = 0
        
    # Format the table
    table_df = df[[
        "min_seq_len", 
        "pad_shorter_sequences", 
        "total_samples", 
        "sample_yield", 
        "padding_percentage", 
        "avg_padding_ratio", 
        "perplexity"
    ]].copy()
    
    # Rename columns for display
    table_df.columns = [
        "min_seq_len", 
        "pad_shorter", 
        "sample_count", 
        "sample_yield_%", 
        "padded_samples_%", 
        "avg_padding_%", 
        "PPL"
    ]
    
    # Save as CSV
    results_file = Path("sequence_length_comparison.csv")
    table_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Print the table
    print("\n" + "="*80)
    print("SEQUENCE LENGTH COMPARISON RESULTS")
    print("="*80)
    print(table_df.to_string(index=False))
    print("="*80)
    print("\nRecommendations:")
    
    # Find the configuration with lowest perplexity among those with reasonable padding
    reasonable_padding = table_df[table_df["avg_padding_%"] < 50]
    if not reasonable_padding.empty:
        best_ppl_config = reasonable_padding.loc[reasonable_padding["PPL"].idxmin()]
        print(f"Best perplexity with reasonable padding (<50%): min_seq_len={best_ppl_config['min_seq_len']}, "
              f"pad_shorter={best_ppl_config['pad_shorter']}, PPL={best_ppl_config['PPL']:.2f}")
    
    # Find the best balance between sample yield and padding
    if not table_df.empty:
        # Define a simple score: higher sample yield, lower padding percentage
        table_df["balance_score"] = table_df["sample_yield_%"] * (100 - table_df["avg_padding_%"]) / 100
        best_balance = table_df.loc[table_df["balance_score"].idxmax()]
        print(f"Best bias-variance balance: min_seq_len={best_balance['min_seq_len']}, "
              f"pad_shorter={best_balance['pad_shorter']}, PPL={best_balance['PPL']:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 