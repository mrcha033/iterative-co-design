#!/usr/bin/env python3
"""
Paper Figure Generation Suite

This script generates all figures from the paper "The Orthogonality Fallacy:
Iterative Co-Design as a First-Class Principle for Efficient AI":

- Figure 1: Random vs. Optimized Permutation Latency
- Figure 2: Quantization Co-Design Strategies
- Figure 3: Metrics vs. Iteration (Causal Chain)
- Figure 4: Pareto Frontier

Usage:
    python scripts/generate_all_figures.py                    # All figures
    python scripts/generate_all_figures.py --figure 1         # Specific figure
    python scripts/generate_all_figures.py --quick            # Fast mode
"""

import warnings
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd
from tqdm import tqdm
import subprocess

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from utils.profiler import LatencyProfiler  # noqa: E402
from co_design.iasp import find_optimal_permutation  # noqa: E402
from models.wrapper import ModelWrapper  # noqa: E402

# ================================
# Figure Generation Functions
# ================================


def generate_random_permutations(model_size: int, n_samples: int = 20) -> list:
    """Generate multiple random permutations for comparison."""
    permutations = []
    for _ in range(n_samples):
        perm = list(range(model_size))
        np.random.shuffle(perm)
        permutations.append(perm)
    return permutations


def measure_permutation_latency(model, permutation, dummy_input_dict, profiler):
    """Measure latency for a specific permutation."""
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()

    # Apply permutation
    wrapped_model.permute_model_weights(permutation)

    # Measure latency
    latency = profiler.measure_latency(wrapped_model, dummy_input_dict)
    return latency


def setup_model_and_data(
    model_name="state-spaces/mamba-2.8b-hf",
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    sample_size=100,
):
    """Common setup for model and data loading."""
    print("📥 Loading model and data...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_config)
    sample_dataset = dataset["validation"].select(range(sample_size))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_dataset = sample_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=4)

    if torch.cuda.is_available():
        model.cuda()

    return model, tokenizer, data_loader


def generate_figure1(quick_mode=False):
    """
    Figure 1: Random vs. Optimized Permutation Latency
    Shows the impact of memory layout optimization on inference latency.
    """
    print("\n📊 Generating Figure 1: Random vs. Optimized Permutation Latency")

    try:
        # Setup
        model, tokenizer, data_loader = setup_model_and_data()
        profiler = LatencyProfiler()

        # Create dummy input
        dummy_input = torch.randint(0, 50277, (1, 512))  # Mamba vocab size
        dummy_input_dict = {"input_ids": dummy_input}
        if torch.cuda.is_available():
            dummy_input_dict = {k: v.cuda() for k, v in dummy_input_dict.items()}

        model_size = model.config.hidden_size
        n_samples = 10 if quick_mode else 20

        # 1. Generate random permutations and measure latency
        print("🎲 Testing random permutations...")
        random_permutations = generate_random_permutations(model_size, n_samples)
        random_latencies = []

        # Measure baseline (identity) latency
        identity_permutation = list(range(model_size))
        baseline_latency = measure_permutation_latency(
            model, identity_permutation, dummy_input_dict, profiler
        )
        print(f"   📊 Baseline latency: {baseline_latency:.2f} ms")

        for perm in tqdm(random_permutations, desc="Testing random permutations"):
            latency = measure_permutation_latency(
                model, perm, dummy_input_dict, profiler
            )
            random_latencies.append(latency)

        # 2. Find optimal permutation using IASP
        print("🔍 Finding optimal permutation...")
        wrapped_model = ModelWrapper(model)
        if torch.cuda.is_available():
            wrapped_model.cuda()

        optimal_permutation = find_optimal_permutation(
            model=wrapped_model,
            data_loader=data_loader,
            target_layer_name="layers.20.mixer.out_proj",  # Mamba target layer
            cluster_size_range=(32, 128),
        )

        # 3. Measure optimal permutation latency
        print("⚡ Measuring optimized latency...")
        optimal_latency = measure_permutation_latency(
            model, optimal_permutation, dummy_input_dict, profiler
        )
        print(f"   📊 Optimized latency: {optimal_latency:.2f} ms")

        # 4. Create visualization
        improvement_vs_baseline = (
            (baseline_latency - optimal_latency) / baseline_latency
        ) * 100

        plt.figure(figsize=(12, 8))

        # Plot data
        x_random = ["Random"] * len(random_latencies)
        plt.scatter(
            x_random,
            random_latencies,
            alpha=0.6,
            color="lightcoral",
            s=60,
            label="Random Permutations",
        )

        plt.bar(
            ["Random (Mean)"],
            [np.mean(random_latencies)],
            color="red",
            alpha=0.7,
            width=0.4,
            label="Random (Average)",
        )
        plt.bar(
            ["Baseline (Identity)"],
            [baseline_latency],
            color="orange",
            alpha=0.7,
            width=0.4,
            label="Baseline (Identity)",
        )
        plt.bar(
            ["Optimized (IASP)"],
            [optimal_latency],
            color="green",
            alpha=0.7,
            width=0.4,
            label="Optimized (IASP)",
        )

        # Add improvement annotation
        plt.annotate(
            f"-{improvement_vs_baseline:.1f}%",
            xy=("Optimized (IASP)", optimal_latency),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color="green",
            arrowprops=dict(arrowstyle="->", color="green"),
        )

        plt.ylabel("Latency (ms)", fontsize=14)
        plt.xlabel("Permutation Type", fontsize=14)
        plt.title(
            "Figure 1: Latency of Mamba Layer with Random vs. Optimized Memory Permutations\n"
            f"Optimized layout reduces latency by {improvement_vs_baseline:.1f}% vs baseline",
            fontsize=16,
            pad=20,
        )
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        sns.set_style("whitegrid")
        plt.tight_layout()

        # Save figure
        output_dir = Path("figures")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / "figure1_mamba_latency_scan_vs_perm.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "figure1_mamba_latency_scan_vs_perm.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save data
        results = {
            "baseline_latency": baseline_latency,
            "random_latencies": random_latencies,
            "random_mean": np.mean(random_latencies),
            "random_std": np.std(random_latencies),
            "optimal_latency": optimal_latency,
            "improvement_vs_baseline_percent": improvement_vs_baseline,
        }

        with open(output_dir / "figure1_data.json", "w") as f:
            json.dump(results, f, indent=4)

        print("✅ Figure 1 saved successfully")
        print(f"   • Improvement: {improvement_vs_baseline:.1f}%")
        return True

    except Exception as e:
        print(f"❌ Figure 1 generation failed: {e}")
        return False


def generate_figure2(non_interactive=False):
    """
    Figure 2: Quantization Co-Design Strategies
    Bar chart comparing different quantization approaches.
    """
    print("\n📊 Generating Figure 2: Quantization Co-Design Strategies")

    try:
        # Check if quantization results exist
        quant_dir = Path("outputs") / "quantization"
        if not quant_dir.exists():
            print("⚠️ WARNING: Figure 2 requires quantization experiment results.")
            print(
                "   This will run experiments which may take 10-30 minutes per method."
            )
            print("   The following experiments will be executed:")
            print("   - python scripts/run_quant_test.py method=quant_then_permute")
            print("   - python scripts/run_quant_test.py method=permute_then_quant")
            print(
                "   - python scripts/run_quant_test.py method=permute_quant_repermute"
            )
            print()

            if non_interactive:
                print("🚀 Non-interactive mode: Auto-proceeding with experiments...")
                response = "y"
            else:
                response = (
                    input("Do you want to run these experiments? (y/N): ")
                    .strip()
                    .lower()
                )

            if response != "y":
                print(
                    "❌ Figure 2 generation cancelled. Run experiments manually if needed."
                )
                return False

            print("🚀 Running quantization experiments...")
            # Run quantization experiments
            commands = [
                "python scripts/run_quant_test.py method=quant_then_permute",
                "python scripts/run_quant_test.py method=permute_then_quant",
                "python scripts/run_quant_test.py method=permute_quant_repermute",
            ]

            for cmd in commands:
                try:
                    subprocess.run(
                        cmd, shell=True, check=True, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError:
                    print(f"❌ Failed to run: {cmd}")
                    return False

        # Load quantization results
        results = {}
        method_labels = {
            "quant_then_permute": "Quant-then-Permute",
            "permute_then_quant": "Permute-then-Quant",
            "permute_quant_repermute": "Permute-Quant-RePermute\n(Ours)",
        }

        for method_file in quant_dir.glob("*_metrics.json"):
            with open(method_file) as f:
                data = json.load(f)
                method_name = method_file.stem.replace("_metrics", "")
                if method_name in method_labels:
                    results[method_labels[method_name]] = data.get("latency", 0)

        if not results:
            print("❌ No quantization results data found")
            return False

        # Create Figure 2
        plt.figure(figsize=(10, 6))
        methods = list(results.keys())
        latencies = list(results.values())
        colors = ["lightcoral", "orange", "green"]

        bars = plt.bar(methods, latencies, color=colors, alpha=0.8)

        # Add value labels on bars
        for bar, latency in zip(bars, latencies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{latency:.1f} ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add improvement annotation
        if len(latencies) >= 3:
            baseline = latencies[1]  # permute_then_quant
            improvement = ((baseline - latencies[2]) / baseline) * 100

            plt.annotate(
                f"-{improvement:.1f}%",
                xy=(methods[2], latencies[2]),
                xytext=(0, -30),
                textcoords="offset points",
                fontsize=14,
                fontweight="bold",
                color="green",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
            )

        plt.ylabel("Latency (ms)", fontsize=14)
        plt.xlabel("Optimization Strategy", fontsize=14)
        plt.title(
            "Figure 2: The Value of Iteration in Co-Design with Quantization\n"
            "Iterative approach outperforms linear pipelines",
            fontsize=16,
            pad=20,
        )
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # Save figure
        output_dir = Path("figures")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / "figure2_quantization_results.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "figure2_quantization_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("✅ Figure 2 saved successfully")
        return True

    except Exception as e:
        print(f"❌ Figure 2 generation failed: {e}")
        return False


def generate_figure3():
    """
    Figure 3: The Causal Chain in Action
    Shows latency, modularity, and cache hit rate vs. iterations.
    """
    print("\n📊 Generating Figure 3: Metrics vs. Iteration (Causal Chain)")

    try:
        # Load experimental results
        results_files = list(Path("outputs").glob("**/*_metrics.json"))

        # Find iterative results
        iterative_data = None
        for f in results_files:
            with open(f) as fp:
                data = json.load(fp)
                if "iteration_metrics" in data:
                    iterative_data = data
                    break

        if not iterative_data or "iteration_metrics" not in iterative_data:
            print("⚠️ No iterative experimental results found.")
            print("   Run: python scripts/run_experiment.py method=iterative")
            return False

        # Extract iteration metrics
        iter_metrics = iterative_data["iteration_metrics"]
        iterations = range(1, len(iter_metrics["latency"]) + 1)

        # Create Figure 3
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot latency (left y-axis, inverted)
        color1 = "tab:blue"
        ax1.set_xlabel("Co-Design Iteration", fontsize=14)
        ax1.set_ylabel("Latency (ms)", color=color1, fontsize=14)
        line1 = ax1.plot(
            iterations,
            iter_metrics["latency"],
            "b-o",
            linewidth=3,
            markersize=8,
            label="Latency",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.invert_yaxis()  # Lower latency is better

        # Plot modularity and cache hit rate (right y-axis)
        ax2 = ax1.twinx()
        color2 = "tab:orange"
        ax2.set_ylabel("Modularity / Cache Hit Rate (%)", color=color2, fontsize=14)

        line2 = ax2.plot(
            iterations,
            iter_metrics["modularity"],
            "o-",
            color="orange",
            linewidth=3,
            markersize=8,
            label="Modularity",
        )
        line3 = ax2.plot(
            iterations,
            iter_metrics["l2_cache_hit_rate"],
            "s-",
            color="green",
            linewidth=3,
            markersize=8,
            label="L2 Cache Hit Rate",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        # Combine legends
        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="center right", fontsize=12)

        plt.title(
            "Figure 3: The Causal Chain in Action\n"
            "Latency, Modularity, and Cache Hit Rate vs. Co-Design Iterations",
            fontsize=16,
            pad=20,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        output_dir = Path("figures")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / "figure3_metrics_vs_iteration.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "figure3_metrics_vs_iteration.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("✅ Figure 3 saved successfully")
        return True

    except Exception as e:
        print(f"❌ Figure 3 generation failed: {e}")
        return False


def generate_figure4():
    """
    Figure 4: Pushing the Pareto Frontier
    Scatter plot of latency vs. perplexity for all methods.
    """
    print("\n📊 Generating Figure 4: Pareto Frontier")

    try:
        # Load all experimental results
        results_files = list(Path("outputs").glob("**/*_metrics.json"))
        if not results_files:
            print("⚠️ No experimental results found.")
            print("   Run experiments first using: python scripts/run_experiment.py")
            return False

        records = []
        for f in results_files:
            with open(f) as fp:
                data = json.load(fp)
            method = Path(f).stem.replace("_metrics", "")
            if "perplexity" in data and "latency_ms" in data:
                records.append(
                    {
                        "method": method,
                        "perplexity": data["perplexity"],
                        "latency_ms": data["latency_ms"],
                    }
                )

        if not records:
            print(
                "❌ No valid experimental data found with both perplexity and latency"
            )
            return False

        df = pd.DataFrame(records)

        # Create Figure 4
        plt.figure(figsize=(10, 8))

        # Define colors and markers for different methods
        method_styles = {
            "dense": {"color": "red", "marker": "o", "size": 100},
            "sparsity_only": {"color": "orange", "marker": "s", "size": 100},
            "permute_only": {"color": "blue", "marker": "^", "size": 100},
            "linear_pipeline": {"color": "purple", "marker": "D", "size": 100},
            "iterative": {"color": "green", "marker": "*", "size": 300},  # Our method
        }

        # Plot each method
        for _, row in df.iterrows():
            method = row["method"]
            style = method_styles.get(
                method, {"color": "gray", "marker": "o", "size": 100}
            )

            label = f"{method} (Ours)" if method == "iterative" else method

            plt.scatter(
                row["perplexity"],
                row["latency_ms"],
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                label=label,
                alpha=0.8,
                edgecolors="black",
                linewidth=1,
            )

            # Add method labels
            plt.annotate(
                method.replace("_", " ").title(),
                (row["perplexity"], row["latency_ms"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        plt.xlabel("Perplexity (lower is better)", fontsize=14)
        plt.ylabel("Latency (ms, lower is better)", fontsize=14)
        plt.title(
            "Figure 4: Pushing the Pareto Frontier\n"
            "Iterative Co-Design establishes new state-of-the-art",
            fontsize=16,
            pad=20,
        )
        plt.legend(fontsize=11, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        output_dir = Path("figures")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / "figure4_pareto_frontier.pdf", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "figure4_pareto_frontier.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("✅ Figure 4 saved successfully")
        return True

    except Exception as e:
        print(f"❌ Figure 4 generation failed: {e}")
        return False


# ================================
# Main Function
# ================================


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--figure",
        type=int,
        choices=[1, 2, 3, 4],
        help="Generate specific figure only (1-4)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode for faster testing"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm all prompts for non-interactive execution",
    )

    args = parser.parse_args()

    print("🎯 Paper Figure Generation Suite")
    print("=" * 60)

    # Define figure generators
    generators = {
        1: generate_figure1,
        2: generate_figure2,
        3: generate_figure3,
        4: generate_figure4,
    }

    success_count = 0
    total_count = 0

    if args.figure:
        # Generate specific figure
        total_count = 1
        print(f"\n🎨 Generating Figure {args.figure} only...")
        if args.figure == 1:
            success = generators[1](quick_mode=args.quick)
        elif args.figure == 2:
            success = generators[2](non_interactive=args.yes)
        else:
            success = generators[args.figure]()

        if success:
            success_count = 1
    else:
        # Generate all figures
        total_count = 4
        print("\n🎨 Generating all paper figures...")

        for fig_num, generator in generators.items():
            if fig_num == 1:
                success = generator(quick_mode=args.quick)
            elif fig_num == 2:
                success = generator(non_interactive=args.yes)
            else:
                success = generator()

            if success:
                success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Generation Summary: {success_count}/{total_count} figures successful")

    if success_count == total_count:
        print("✅ All figures generated successfully!")
    elif success_count > 0:
        print("⚠️ Some figures generated successfully, others failed")
    else:
        print("❌ No figures were generated successfully")

    # List generated files
    figures_dir = Path("figures")
    if figures_dir.exists():
        figure_files = sorted(
            list(figures_dir.glob("*.pdf")) + list(figures_dir.glob("*.png"))
        )
        if figure_files:
            print(f"\n📄 Generated files ({len(figure_files)}):")
            for f in figure_files:
                print(f"   • {f.name}")

        print(f"\n📁 All figures saved to: {figures_dir.absolute()}")
        print("\n💡 Usage tips:")
        print("   • Use PDF files for publication")
        print("   • Use PNG files for presentations/web")
        print("   • Check JSON data files for raw results")


if __name__ == "__main__":
    main()
