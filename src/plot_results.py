#!/usr/bin/env python
"""
Plot and visualize results from previous runs of the analysis.

This script loads cached results from previous runs and creates
visualizations to show the key findings.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from visualizations import (
    plot_helix_3d,
    visualize_circular_structure,
    visualize_clock_algorithm_for_numbers,
    create_summary_figure,
    plot_activation_patching_comparison
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot results from previous runs")
    
    parser.add_argument("--cache_dir", type=str, default="../data/cache",
                        help="Directory with cached results (default: ../data/cache)")
    parser.add_argument("--results_dir", type=str, default="../results/plots",
                        help="Directory to save plots (default: ../results/plots)")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model name to load results for (default: gpt2)")
    parser.add_argument("--numbers", type=str, default="0-100",
                        help="Range of numbers (format: min-max, default: 0-100)")
    
    return parser.parse_args()

def plot_helix_representations(helix_fits, numbers, results_dir):
    """Plot helix representations for different layers and periods."""
    print("Plotting helix representations...")
    
    # Create directory for helix plots
    helix_dir = os.path.join(results_dir, "helices")
    os.makedirs(helix_dir, exist_ok=True)
    
    # Find best layer based on R² score
    best_layer = max(helix_fits.keys(), key=lambda l: helix_fits[l]['r2_score'])
    
    # For each layer, plot the helix for different periods
    for layer in sorted(helix_fits.keys()):
        is_best = layer == best_layer
        label = "best_" if is_best else ""
        
        # For each period, create a visualization
        for period in helix_fits[layer]['periods']:
            plot_path = os.path.join(helix_dir, f"{label}layer{layer}_T{period}.png")
            plot_helix_3d(
                numbers=numbers,
                helix_fit=helix_fits[layer],
                periods=[period],
                plot_path=plot_path
            )
            print(f"  Saved helix plot for layer {layer}, period {period} to {plot_path}")
    
    # Plot circular structure
    vis_path = os.path.join(helix_dir, "circular_structure.png")
    visualize_circular_structure(
        numbers=numbers[:20],  # Use subset for clarity
        periods=helix_fits[best_layer]['periods'],
        plot_path=vis_path
    )
    print(f"  Saved circular structure visualization to {vis_path}")
    
    return best_layer

def plot_clock_algorithm_demos(numbers, periods, results_dir):
    """Plot demonstrations of the Clock algorithm for different number pairs."""
    print("Plotting Clock algorithm demonstrations...")
    
    # Create directory for clock plots
    clock_dir = os.path.join(results_dir, "clock")
    os.makedirs(clock_dir, exist_ok=True)
    
    # Pairs to demonstrate
    demo_pairs = [
        (3, 4),    # Simple addition
        (7, 5),    # Addition with mod 10 = 2, showcasing carry
        (12, 9),   # Double digit first operand
        (23, 45),  # Double digit both operands
        (99, 1)    # Addition crossing century boundary
    ]
    
    for a, b in demo_pairs:
        plot_path = os.path.join(clock_dir, f"clock_{a}_plus_{b}.png")
        visualize_clock_algorithm_for_numbers(
            a=a,
            b=b,
            periods=periods,
            plot_path=plot_path
        )
        print(f"  Saved Clock algorithm demo for {a}+{b} to {plot_path}")
    
    return demo_pairs

def plot_patching_results(patching_results, best_layer, results_dir):
    """Plot results from activation patching experiments."""
    print("Plotting activation patching results...")
    
    # Create directory for patching plots
    patching_dir = os.path.join(results_dir, "patching")
    os.makedirs(patching_dir, exist_ok=True)
    
    # Plot comparison of patching methods
    comparison_path = os.path.join(patching_dir, "patching_comparison.png")
    plot_activation_patching_comparison(
        patching_results=patching_results,
        best_layer=best_layer,
        plot_path=comparison_path
    )
    print(f"  Saved patching methods comparison to {comparison_path}")

def create_overall_summary(numbers, helix_fits, best_layer, patching_results, demo_pairs, results_dir):
    """Create an overall summary figure of key findings."""
    print("Creating summary figure...")
    
    summary_path = os.path.join(results_dir, "summary_figure.png")
    create_summary_figure(
        numbers=numbers,
        helix_fit=helix_fits[best_layer],
        patching_results=patching_results,
        sample_a=demo_pairs[1][0],
        sample_b=demo_pairs[1][1],
        plot_path=summary_path
    )
    print(f"Saved summary figure to {summary_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Parse number range
    min_num, max_num = map(int, args.numbers.split('-'))
    numbers = list(range(min_num, max_num + 1))
    
    # Create results directory
    model_name_safe = args.model_name.replace("/", "_")
    results_dir = Path(args.results_dir) / model_name_safe
    os.makedirs(results_dir, exist_ok=True)
    
    # Cache directory
    cache_dir = Path(args.cache_dir) / model_name_safe
    
    print(f"Loading results for model {args.model_name}, numbers {min_num}-{max_num}")
    print(f"Results will be saved to {results_dir}")
    
    # Load helix fits
    helix_fits_path = cache_dir / f"helix_fits_{min_num}_{max_num}.pt"
    if not helix_fits_path.exists():
        print(f"ERROR: Could not find cached helix fits at {helix_fits_path}")
        print("Please run the main analysis first to generate these files.")
        return
    
    helix_fits = torch.load(helix_fits_path)
    print(f"Loaded helix fits for {len(helix_fits)} layers")
    
    # Load patching results
    patching_path = cache_dir / "patching_results.pt"
    if not patching_path.exists():
        print(f"WARNING: Could not find patching results at {patching_path}")
        print("Proceeding without patching results. Some visualizations will be skipped.")
        patching_results = None
    else:
        patching_results = torch.load(patching_path)
        print(f"Loaded patching results for {len(patching_results['layers'])} layers")
    
    # Plot helix representations
    best_layer = plot_helix_representations(helix_fits, numbers, results_dir)
    
    # Plot Clock algorithm demos
    periods = helix_fits[best_layer]['periods']
    demo_pairs = plot_clock_algorithm_demos(numbers, periods, results_dir)
    
    # Plot patching results if available
    if patching_results is not None:
        plot_patching_results(patching_results, best_layer, results_dir)
        
        # Create overall summary
        create_overall_summary(numbers, helix_fits, best_layer, patching_results, demo_pairs, results_dir)
    
    print("\nAll plots created successfully.")
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main() 