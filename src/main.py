#!/usr/bin/env python
"""
Main script for the "Language Models Use Trigonometry to Do Addition" project.

This script demonstrates the full pipeline, from extracting representations to
visualizing the Clock algorithm and causal interventions.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Import project modules
from number_representation import extract_number_representations
from helix_fitting import fit_helix, project_to_helix_subspace, evaluate_helix_fit
from causal_interventions import patch_across_layers, plot_patching_results
from clock_algorithm import extract_addition_representations, fit_addition_helices, verify_clock_algorithm
from visualizations import (
    plot_helix_3d, 
    visualize_circular_structure,
    visualize_clock_algorithm_for_numbers,
    create_summary_figure,
    plot_activation_patching_comparison
)
import utils

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Language Models Use Trigonometry to Do Addition")
    
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Model to use (default: gpt2)")
    parser.add_argument("--min_number", type=int, default=0, 
                        help="Minimum number to analyze (default: 0)")
    parser.add_argument("--max_number", type=int, default=100, 
                        help="Maximum number to analyze (default: 100)")
    parser.add_argument("--periods", nargs="+", type=int, default=[2, 5, 10, 100],
                        help="Periods to use for helix fitting (default: 2 5 10 100)")
    parser.add_argument("--layers", nargs="+", type=int, 
                        help="Layers to analyze (default: all layers)")
    parser.add_argument("--token_type", type=str, default="number", 
                        choices=["number", "equals", "prompt", "a", "b"],
                        help="Token type to analyze (default: number)")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory to save results (default: ../results)")
    parser.add_argument("--cache_dir", type=str, default="../data/cache",
                        help="Directory to cache intermediate results (default: ../data/cache)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (reduced computation)")
    
    return parser.parse_args()

def setup_directories(args):
    """Setup directories for results and cache."""
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create model-specific subdirectories
    model_result_dir = os.path.join(args.results_dir, args.model_name.replace("/", "_"))
    model_cache_dir = os.path.join(args.cache_dir, args.model_name.replace("/", "_"))
    
    os.makedirs(model_result_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    
    return model_result_dir, model_cache_dir

def load_model(model_name, device):
    """Load model and tokenizer."""
    print(f"Loading model {model_name} on {device}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded: {model_name}")
        print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def analyze_numbers(args, model, tokenizer, result_dir, cache_dir):
    """Analyze number representations and fit helices."""
    print("\n== Analyzing Number Representations ==")
    
    # Generate range of numbers to analyze
    numbers = list(range(args.min_number, args.max_number + 1))
    
    # Extract representations
    number_reps = extract_number_representations(
        model=model,
        tokenizer=tokenizer,
        numbers=numbers,
        layer=None,  # All layers
        token_type=args.token_type,
        cache_file=os.path.join(cache_dir, f"number_reps_{args.min_number}_{args.max_number}.pt"),
        device=args.device
    )
    
    print(f"Extracted representations for {len(numbers)} numbers across {len(number_reps['layer_to_reps'])} layers")
    
    # Fit helices for each layer
    helix_fits = {}
    for layer, reps in number_reps['layer_to_reps'].items():
        if args.layers is not None and layer not in args.layers:
            continue
            
        print(f"\nFitting helix for layer {layer}...")
        helix_fit = fit_helix(
            numbers=numbers,
            representations=reps,
            periods=args.periods
        )
        
        # Evaluate fit
        evaluation = evaluate_helix_fit(numbers, reps, helix_fit)
        helix_fits[layer] = {**helix_fit, **evaluation}
        
        print(f"Layer {layer}: R² = {evaluation['r2_score']:.4f}, MSE = {evaluation['mse']:.4f}")
    
    # Save helix fits
    torch.save(helix_fits, os.path.join(cache_dir, f"helix_fits_{args.min_number}_{args.max_number}.pt"))
    
    # Find best layer based on R² score
    best_layer = max(helix_fits.keys(), key=lambda l: helix_fits[l]['r2_score'])
    print(f"\nBest layer: {best_layer} with R² = {helix_fits[best_layer]['r2_score']:.4f}")
    
    # Visualize helix for best layer
    print("\nVisualizing helix for best layer...")
    for period in args.periods:
        plot_path = os.path.join(result_dir, f"helix_3d_layer{best_layer}_T{period}.png")
        plot_helix_3d(
            numbers=numbers,
            helix_fit=helix_fits[best_layer],
            periods=[period],
            plot_path=plot_path
        )
        print(f"Saved 3D helix plot to {plot_path}")
    
    # Visualize circular structure
    print("\nVisualizing circular structure...")
    vis_path = os.path.join(result_dir, f"circular_structure.png")
    visualize_circular_structure(
        numbers=numbers[:20],  # Use subset for clarity
        periods=args.periods,
        plot_path=vis_path
    )
    print(f"Saved circular structure visualization to {vis_path}")
    
    return helix_fits, best_layer

def analyze_addition(args, model, tokenizer, helix_fits, best_layer, result_dir, cache_dir):
    """Analyze how the model performs addition using helical representations."""
    print("\n== Analyzing Addition ==")
    
    # Define range of values for a and b
    a_range = list(range(args.min_number, min(args.max_number, 20)))
    b_range = list(range(args.min_number, min(args.max_number, 20)))
    
    # Extract addition representations
    addition_reps = extract_addition_representations(
        model=model,
        tokenizer=tokenizer,
        a_values=a_range,
        b_values=b_range,
        layer=best_layer,
        token_type="token",  # Extract 'a', 'b', and 'equals' tokens
        device=args.device,
        model_name=args.model_name,
        cache_file=os.path.join(cache_dir, f"addition_reps_layer{best_layer}.pt")
    )
    
    print(f"Extracted addition representations for {len(a_range)}×{len(b_range)} problems")
    
    # Fit helices to addition representations
    addition_helix_fits = fit_addition_helices(
        addition_reps=addition_reps,
        periods=args.periods,
        result_path=os.path.join(result_dir, "addition_helix_fits.png")
    )
    
    print(f"Fitted helices for addition representations")
    
    # Verify Clock algorithm
    clock_results = verify_clock_algorithm(
        addition_reps=addition_reps,
        helix_fits=addition_helix_fits,
        result_path=os.path.join(result_dir, "clock_verification.png")
    )
    
    print(f"Clock algorithm verification:")
    print(f"- Clock Algorithm R² = {clock_results['clock_r2']:.4f}")
    print(f"- Vector Addition R² = {clock_results['vector_r2']:.4f}")
    print(f"- Linear Addition R² = {clock_results['linear_r2']:.4f}")
    
    # Visualize helix addition
    vis_path = os.path.join(result_dir, "helix_addition.png")
    addition_helix_fits['visualize_helix_addition'](
        addition_reps=addition_reps,
        plot_path=vis_path
    )
    print(f"Saved helix addition visualization to {vis_path}")
    
    # Demonstrate Clock algorithm for sample numbers
    sample_a, sample_b = 3, 63
    demo_path = os.path.join(result_dir, f"clock_demo_{sample_a}_{sample_b}.png")
    visualize_clock_algorithm_for_numbers(
        a=sample_a,
        b=sample_b,
        periods=args.periods,
        plot_path=demo_path
    )
    print(f"Saved Clock algorithm demonstration to {demo_path}")
    
    return addition_reps, addition_helix_fits, clock_results

def run_causal_interventions(args, model, tokenizer, helix_fits, best_layer, result_dir, cache_dir):
    """Run causal intervention experiments (activation patching)."""
    print("\n== Running Causal Interventions ==")
    
    # Define range of values for a and b
    a_range = list(range(args.min_number, min(args.max_number, 20)))
    b_range = list(range(args.min_number, min(args.max_number, 20)))
    
    # Generate clean and corrupted pairs for patching
    num_pairs = 25  # Number of pairs to use
    
    # Use a subset of layers for patching
    if args.layers is None:
        model_layers = model.config.num_hidden_layers
        patching_layers = list(range(0, model_layers, max(1, model_layers // 10)))
    else:
        patching_layers = args.layers
    
    # Run patching across layers
    patching_results = patch_across_layers(
        model=model,
        tokenizer=tokenizer,
        a_range=a_range,
        b_range=b_range,
        num_pairs=num_pairs,
        layers=patching_layers,
        helix_fit=helix_fits[best_layer],
        cache_file=os.path.join(cache_dir, f"patching_results.pt"),
        device=args.device
    )
    
    print(f"Completed activation patching across {len(patching_layers)} layers")
    
    # Plot patching results
    plot_path = os.path.join(result_dir, "patching_results.png")
    plot_patching_results(
        patching_results=patching_results,
        plot_path=plot_path
    )
    print(f"Saved patching results plot to {plot_path}")
    
    # Plot comparison of patching methods
    comparison_path = os.path.join(result_dir, "patching_comparison.png")
    plot_activation_patching_comparison(
        patching_results=patching_results,
        best_layer=best_layer if best_layer in patching_layers else None,
        plot_path=comparison_path
    )
    print(f"Saved patching methods comparison to {comparison_path}")
    
    return patching_results

def create_final_summary(args, numbers, helix_fits, best_layer, patching_results, result_dir):
    """Create a final summary figure of the key findings."""
    print("\n== Creating Summary Figure ==")
    
    summary_path = os.path.join(result_dir, "summary_figure.png")
    create_summary_figure(
        numbers=numbers,
        helix_fit=helix_fits[best_layer],
        patching_results=patching_results,
        sample_a=3,
        sample_b=63,
        plot_path=summary_path
    )
    print(f"Saved summary figure to {summary_path}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    result_dir, cache_dir = setup_directories(args)
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, args.device)
    
    # Generate range of numbers to analyze
    numbers = list(range(args.min_number, args.max_number + 1))
    
    # Analyze number representations and fit helices
    helix_fits, best_layer = analyze_numbers(args, model, tokenizer, result_dir, cache_dir)
    
    # Analyze addition using helical representations
    addition_reps, addition_helix_fits, clock_results = analyze_addition(
        args, model, tokenizer, helix_fits, best_layer, result_dir, cache_dir
    )
    
    # Run causal interventions
    patching_results = run_causal_interventions(
        args, model, tokenizer, helix_fits, best_layer, result_dir, cache_dir
    )
    
    # Create final summary
    create_final_summary(args, numbers, helix_fits, best_layer, patching_results, result_dir)
    
    print("\n== Analysis Complete ==")
    print(f"All results saved to {result_dir}")

if __name__ == "__main__":
    main() 