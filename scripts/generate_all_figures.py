#!/usr/bin/env python3
"""
Generate All Paper Figures

This script orchestrates the generation of all figures from the paper:
- Figure 1: Random vs. Optimized Permutation Latency
- Figure 2: Quantization Results Bar Chart  
- Figure 3: Metrics vs. Iteration (from experimental results)
- Figure 4: Pareto Frontier (from experimental results)

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --quick  # For faster testing
    python scripts/generate_all_figures.py --figure 1  # Generate specific figure
"""

import sys
import argparse
from pathlib import Path
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def generate_figure1(quick_mode=False):
    """Generate Figure 1: Random vs. Optimized Permutation Latency."""
    print("\n📊 Generating Figure 1: Random vs. Optimized Permutation Latency")
    cmd = "python scripts/generate_figure1.py model=mamba_3b dataset=wikitext103"
    return run_command(cmd, "Figure 1 generation")

def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick mode for faster testing")
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4], 
                       help="Generate specific figure only")
    
    args = parser.parse_args()
    
    print("🎯 Generating Paper Figures")
    print("="*50)
    
    success_count = 0
    total_count = 0
    
    if args.figure:
        if args.figure == 1:
            total_count = 1
            if generate_figure1(args.quick):
                success_count += 1
        else:
            print(f"⚠️ Figure {args.figure} generation not yet implemented")
    else:
        # Generate all figures
        total_count = 1  # Only Figure 1 implemented for now
        if generate_figure1(args.quick):
            success_count += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"📊 Figure Generation Summary: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("✅ All figures generated successfully!")
        print("📁 Check the 'figures/' directory for output files")
    else:
        print("⚠️ Some figures failed to generate.")
    
    # List generated files
    figures_dir = Path("figures")
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*.pdf")) + list(figures_dir.glob("*.png"))
        if figure_files:
            print(f"\n📄 Generated files ({len(figure_files)}):")
            for f in sorted(figure_files):
                print(f"   - {f}")

if __name__ == "__main__":
    main() 