#!/usr/bin/env python
"""
Demonstration script for the "Language Models Use Trigonometry to Do Addition" project.

This script provides a simplified demonstration of the Clock algorithm with minimal
resource requirements. It uses a small range of numbers and a smaller model.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from number_representation import extract_number_representations
from helix_fitting import fit_helix, evaluate_helix_fit
from clock_algorithm import demonstrate_clock_algorithm
from visualizations import (
    plot_helix_3d,
    visualize_circular_structure,
    visualize_clock_algorithm_for_numbers
)

def main():
    """Run a simplified demonstration of the Clock algorithm."""
    print("== Language Models Use Trigonometry to Do Addition ==")
    print("Running demonstration with minimal resources...\n")
    
    # Setup directories
    os.makedirs("../results/demo", exist_ok=True)
    
    # Settings for the demo
    model_name = "distilgpt2"  # Smaller model for faster execution
    min_number = 0
    max_number = 20
    periods = [2, 5, 10]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using model: {model_name} on {device}")
    print(f"Analyzing numbers from {min_number} to {max_number}")
    print(f"Using periods: {periods}\n")
    
    # Load model and tokenizer
    try:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters\n")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Proceeding with synthetic data for demonstration purposes only...")
        model = None
        tokenizer = None
    
    # Generate range of numbers to analyze
    numbers = list(range(min_number, max_number + 1))
    
    # Step 1: Extract representations or generate synthetic data if model loading failed
    if model is not None and tokenizer is not None:
        print("Extracting number representations...")
        representations = extract_number_representations(
            model=model,
            tokenizer=tokenizer,
            numbers=numbers,
            layer=4,  # Use a middle layer for the demo
            device=device
        )
        print("Representations extracted.\n")
    else:
        print("Generating synthetic representations for demonstration...")
        # Generate synthetic representations that follow a helical pattern
        dim = 20  # Small dimension for synthetic data
        representations = np.zeros((len(numbers), dim))
        
        # Add linear component
        representations[:, 0] = numbers
        
        # Add helical components
        for i, num in enumerate(numbers):
            for j, period in enumerate(periods):
                representations[i, 2*j+1] = np.cos(2 * np.pi * num / period)
                representations[i, 2*j+2] = np.sin(2 * np.pi * num / period)
                
        # Add some noise
        representations += np.random.normal(0, 0.1, representations.shape)
        representations = torch.tensor(representations)
        print("Synthetic representations generated.\n")
    
    # Step 2: Fit helix to representations
    print("Fitting helix to representations...")
    helix_fit = fit_helix(
        numbers=numbers,
        representations=representations,
        periods=periods
    )
    
    # Evaluate fit
    evaluation = evaluate_helix_fit(numbers, representations, helix_fit)
    print(f"Helix fitted with R² = {evaluation['r2_score']:.4f}, MSE = {evaluation['mse']:.4f}\n")
    
    # Step 3: Visualize helical representation
    print("Visualizing helical representation...")
    for period in periods:
        plot_path = f"../results/demo/helix_3d_T{period}.png"
        plot_helix_3d(
            numbers=numbers,
            helix_fit=helix_fit,
            periods=[period],
            plot_path=plot_path
        )
        print(f"Saved 3D helix plot for T={period} to {plot_path}")
    
    # Visualize circular structure
    vis_path = "../results/demo/circular_structure.png"
    visualize_circular_structure(
        numbers=numbers,
        periods=periods,
        plot_path=vis_path
    )
    print(f"Saved circular structure visualization to {vis_path}\n")
    
    # Step 4: Demonstrate Clock algorithm
    print("Demonstrating Clock algorithm...")
    
    # Demonstrate for specific number pairs
    example_pairs = [(3, 4), (7, 12), (5, 15)]
    
    for a, b in example_pairs:
        demo_path = f"../results/demo/clock_demo_{a}_{b}.png"
        visualize_clock_algorithm_for_numbers(
            a=a,
            b=b,
            periods=periods,
            plot_path=demo_path
        )
        print(f"Saved Clock algorithm demonstration for {a}+{b} to {demo_path}")
    
    # Use the specialized function from clock_algorithm module
    clock_path = "../results/demo/clock_algorithm.png"
    demonstrate_clock_algorithm(
        periods=periods,
        plot_path=clock_path
    )
    print(f"Saved general Clock algorithm demonstration to {clock_path}\n")
    
    print("== Demonstration Complete ==")
    print("Results saved to ../results/demo/")
    print("\nKey Insights:")
    print("1. LLMs represent numbers as generalized helices in their residual stream")
    print("2. These helices have periods that correspond to significant numerical patterns (2, 5, 10)")
    print("3. For addition, LLMs use the Clock algorithm - rotating on these circles by amounts")
    print("4. The Clock algorithm combines the linear component and the circular rotations")
    print("5. This provides a mechanistic explanation for how LLMs perform addition")

if __name__ == "__main__":
    main() 