"""
Clock algorithm implementation for how LLMs perform addition.

This module demonstrates the Clock algorithm, which explains how language models
manipulate helical representations to perform addition.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Any
import os
from scipy.optimize import minimize

from utils import to_numpy, format_prompt, save_results, load_results
from helix_fitting import generate_helix_basis, fit_helix, project_to_helix, reconstruct_from_helix

def extract_addition_representations(
    model: Any,
    tokenizer: Any,
    a_values: List[int],
    b_values: List[int],
    layer: int = 0,
    token_type: str = "equals",
    device: str = "cuda",
    model_name: str = "gpt-j",
    cache_file: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract representations for addition problems from a language model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        a_values: List of 'a' values
        b_values: List of 'b' values
        layer: Layer to extract representations from
        token_type: Type of token to extract ('a', 'b', 'equals', or 'all')
        device: Device to run the model on
        model_name: Model name for prompt formatting
        cache_file: Optional path to cache results
        
    Returns:
        dict: Dictionary with representations for a, b, and equals tokens
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached representations from {cache_file}")
        return np.load(cache_file, allow_pickle=True).item()
    
    print(f"Extracting representations for {len(a_values) * len(b_values)} addition problems...")
    
    # Initialize storage for representations
    a_reps = []
    b_reps = []
    equals_reps = []
    all_reps = []
    problems = []
    
    for a in a_values:
        for b in b_values:
            # Skip if sum is too large (might not be a single token)
            if a + b >= 200:
                continue
                
            # Format prompt
            prompt = format_prompt(a, b, model_name)
            problems.append((a, b))
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get hidden states at specified layer
                hidden_states = outputs.hidden_states[layer][0]
                
                # Extract token positions
                input_text = tokenizer.decode(inputs.input_ids[0])
                a_pos = 0  # First token is usually the 'a' value
                
                # Find position of '+' to locate 'b'
                plus_text = '+'
                plus_tokens = tokenizer.encode(plus_text, add_special_tokens=False)
                plus_pos = None
                
                for i in range(len(inputs.input_ids[0])):
                    if inputs.input_ids[0, i].item() in plus_tokens:
                        plus_pos = i
                        break
                
                if plus_pos is not None:
                    b_pos = plus_pos + 1
                else:
                    # Fall back to simple position if '+' not found
                    b_pos = 2
                
                # Last position is the '=' token
                equals_pos = inputs.input_ids.shape[1] - 1
                
                # Extract representations
                a_rep = hidden_states[a_pos].clone()
                b_rep = hidden_states[b_pos].clone()
                equals_rep = hidden_states[equals_pos].clone()
                
                a_reps.append(to_numpy(a_rep))
                b_reps.append(to_numpy(b_rep))
                equals_reps.append(to_numpy(equals_rep))
                all_reps.append({
                    'a': to_numpy(a_rep),
                    'b': to_numpy(b_rep),
                    'equals': to_numpy(equals_rep),
                    'prompt': prompt,
                    'a_value': a,
                    'b_value': b,
                    'sum': a + b
                })
    
    # Stack into matrices
    a_matrix = np.stack(a_reps, axis=0)
    b_matrix = np.stack(b_reps, axis=0)
    equals_matrix = np.stack(equals_reps, axis=0)
    
    # Create result dictionary
    result = {
        'a_matrix': a_matrix,
        'b_matrix': b_matrix,
        'equals_matrix': equals_matrix,
        'all_reps': all_reps,
        'problems': problems,
        'layer': layer
    }
    
    # Cache results if path provided
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, result)
    
    return result

def fit_addition_helices(
    addition_reps: Dict[str, Any],
    periods: List[int] = [2, 5, 10, 100],
    pca_dim: int = 100,
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fit helices to the a, b, and equals token representations.
    
    Args:
        addition_reps: Dictionary with addition representations
        periods: List of periods for the Fourier features
        pca_dim: Dimension to reduce to with PCA before fitting
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with helix fits for a, b, and equals tokens
    """
    print(f"Fitting helices with periods {periods}...")
    
    # Extract matrices and problems
    a_matrix = addition_reps['a_matrix']
    b_matrix = addition_reps['b_matrix']
    equals_matrix = addition_reps['equals_matrix']
    problems = addition_reps['problems']
    
    # Extract a, b, and sum values
    a_values = [a for a, _ in problems]
    b_values = [b for _, b in problems]
    sum_values = [a + b for a, b in problems]
    
    # Fit helices
    a_helix = fit_helix(a_matrix, a_values, periods, pca_dim)
    b_helix = fit_helix(b_matrix, b_values, periods, pca_dim)
    equals_helix = fit_helix(equals_matrix, sum_values, periods, pca_dim)
    
    # Create result dictionary
    result = {
        'a_helix': a_helix,
        'b_helix': b_helix,
        'equals_helix': equals_helix,
        'periods': periods,
        'problems': problems
    }
    
    if result_path:
        save_results(result, result_path)
    
    return result

def verify_clock_algorithm(
    addition_reps: Dict[str, Any],
    helix_fits: Dict[str, Any],
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify that the equals token representation is well modeled by helix(a+b).
    
    Args:
        addition_reps: Dictionary with addition representations
        helix_fits: Dictionary with helix fits
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with verification results
    """
    print("Verifying Clock algorithm...")
    
    # Extract matrices and problems
    equals_matrix = addition_reps['equals_matrix']
    problems = addition_reps['problems']
    
    # Extract helix fits
    a_helix = helix_fits['a_helix']
    b_helix = helix_fits['b_helix']
    equals_helix = helix_fits['equals_helix']
    periods = helix_fits['periods']
    
    # Extract a, b, and sum values
    a_values = [a for a, _ in problems]
    b_values = [b for _, b in problems]
    sum_values = [a + b for a, b in problems]
    
    # Compute equals token predictions using different methods
    results = {
        'n_samples': len(problems),
        'periods': periods,
        'reconstruction_errors': {},
        'r2_scores': {}
    }
    
    # 1. Direct helix(a+b) fit
    # This is already computed in equals_helix
    
    # 2. Linear combination of a and b helices
    # Create a synthetic helix(a+b) by fitting a linear model to a_helix and b_helix
    X = np.zeros((len(problems), 2))
    for i, (a, b) in enumerate(problems):
        X[i, 0] = a
        X[i, 1] = b
    
    # Generate helix basis for a+b
    basis_sum = generate_helix_basis(sum_values, periods)
    
    # Generate direct linear combination of a and b
    linear_combination = np.column_stack((a_values, b_values))
    
    # Compute PCA of equals_matrix
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(100, equals_matrix.shape[0], equals_matrix.shape[1]))
    equals_pca = pca.fit_transform(equals_matrix)
    
    # Fit different models to the equals token PCA
    from sklearn.linear_model import LinearRegression
    
    # Fit using helix(a+b)
    model_helix_sum = LinearRegression()
    model_helix_sum.fit(basis_sum, equals_pca)
    pred_helix_sum = model_helix_sum.predict(basis_sum)
    pred_full_helix_sum = pred_helix_sum @ pca.components_
    
    # Fit using linear combination of a and b
    model_linear = LinearRegression()
    model_linear.fit(linear_combination, equals_pca)
    pred_linear = model_linear.predict(linear_combination)
    pred_full_linear = pred_linear @ pca.components_
    
    # Calculate reconstruction errors and R² scores
    from sklearn.metrics import mean_squared_error, r2_score
    
    # For helix(a+b)
    mse_helix_sum = mean_squared_error(equals_matrix, pred_full_helix_sum)
    r2_helix_sum = r2_score(equals_matrix, pred_full_helix_sum)
    
    # For linear combination
    mse_linear = mean_squared_error(equals_matrix, pred_full_linear)
    r2_linear = r2_score(equals_matrix, pred_full_linear)
    
    # Store results
    results['reconstruction_errors']['helix_sum'] = mse_helix_sum
    results['reconstruction_errors']['linear'] = mse_linear
    
    results['r2_scores']['helix_sum'] = r2_helix_sum
    results['r2_scores']['linear'] = r2_linear
    
    # Compute additional metrics to compare the models
    results['model_comparison'] = {
        'helix_sum_vs_linear': mse_linear / mse_helix_sum,
        'r2_diff': r2_helix_sum - r2_linear
    }
    
    if result_path:
        save_results(results, result_path)
    
    return results

def demonstrate_clock_algorithm(
    periods: List[int] = [2, 5, 10, 100],
    plot_path: Optional[str] = None
):
    """
    Demonstrate how the Clock algorithm works for addition.
    
    Args:
        periods: List of periods for the Fourier features
        plot_path: Optional path to save the plot
    """
    # Create a simple example
    a = 3
    b = 63
    result = a + b
    
    # Create figure
    fig, axes = plt.subplots(len(periods), 3, figsize=(15, 4*len(periods)))
    
    # For each period, demonstrate how circular addition works
    for i, T in enumerate(periods):
        # Create circle for this period
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        # Calculate positions on the circle for a, b, and a+b (modulo T)
        a_angle = 2 * np.pi * (a % T) / T
        b_angle = 2 * np.pi * (b % T) / T
        result_angle = 2 * np.pi * (result % T) / T
        
        a_x, a_y = np.cos(a_angle), np.sin(a_angle)
        b_x, b_y = np.cos(b_angle), np.sin(b_angle)
        result_x, result_y = np.cos(result_angle), np.sin(result_angle)
        
        # Plot the base circle with a
        axes[i, 0].plot(circle_x, circle_y, 'k--', alpha=0.3)
        axes[i, 0].scatter(a_x, a_y, color='blue', s=100, label=f'a={a}')
        axes[i, 0].scatter(0, 0, color='black', s=30)
        axes[i, 0].set_title(f'Step 1: Position for a={a} (mod {T})')
        axes[i, 0].set_aspect('equal')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(-1.2, 1.2)
        axes[i, 0].set_ylim(-1.2, 1.2)
        axes[i, 0].legend()
        
        # Plot the rotation by b
        axes[i, 1].plot(circle_x, circle_y, 'k--', alpha=0.3)
        axes[i, 1].scatter(a_x, a_y, color='blue', s=100, label=f'a={a}')
        axes[i, 1].scatter(b_x, b_y, color='green', s=100, label=f'b={b}')
        
        # Draw arrow from a to a+b
        axes[i, 1].arrow(a_x, a_y, 
                        (result_x - a_x) * 0.8, 
                        (result_y - a_y) * 0.8,
                        head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', alpha=0.7)
        
        axes[i, 1].scatter(0, 0, color='black', s=30)
        axes[i, 1].set_title(f'Step 2: Rotate by b={b} (mod {T})')
        axes[i, 1].set_aspect('equal')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(-1.2, 1.2)
        axes[i, 1].set_ylim(-1.2, 1.2)
        axes[i, 1].legend()
        
        # Plot the final result
        axes[i, 2].plot(circle_x, circle_y, 'k--', alpha=0.3)
        axes[i, 2].scatter(result_x, result_y, color='red', s=100, label=f'a+b={result}')
        axes[i, 2].scatter(0, 0, color='black', s=30)
        axes[i, 2].set_title(f'Step 3: Result a+b={result} (mod {T})')
        axes[i, 2].set_aspect('equal')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_xlim(-1.2, 1.2)
        axes[i, 2].set_ylim(-1.2, 1.2)
        axes[i, 2].legend()
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_helix_addition(
    helix_fits: Dict[str, Any],
    sample_indices: List[int] = None,
    plot_path: Optional[str] = None
):
    """
    Visualize how addition works in the helix subspace.
    
    Args:
        helix_fits: Dictionary with helix fits
        sample_indices: Indices of problems to visualize
        plot_path: Optional path to save the plot
    """
    # Extract parameters
    problems = helix_fits['problems']
    periods = helix_fits['periods']
    a_helix = helix_fits['a_helix']
    b_helix = helix_fits['b_helix']
    equals_helix = helix_fits['equals_helix']
    
    # Select sample indices if not provided
    if sample_indices is None:
        if len(problems) <= 5:
            sample_indices = list(range(len(problems)))
        else:
            import random
            sample_indices = random.sample(range(len(problems)), 5)
    
    # Create figure
    n_samples = len(sample_indices)
    n_periods = len(periods)
    
    fig, axes = plt.subplots(n_samples, n_periods, figsize=(4*n_periods, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    # For each sample
    for i, idx in enumerate(sample_indices):
        a, b = problems[idx]
        result = a + b
        
        # For each period
        for j, T in enumerate(periods):
            # Calculate positions on the circle for a, b, and a+b (modulo T)
            a_angle = 2 * np.pi * (a % T) / T
            b_angle = 2 * np.pi * (b % T) / T
            result_angle = 2 * np.pi * (result % T) / T
            
            a_x, a_y = np.cos(a_angle), np.sin(a_angle)
            b_x, b_y = np.cos(b_angle), np.sin(b_angle)
            result_x, result_y = np.cos(result_angle), np.sin(result_angle)
            
            # Create circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            
            # Plot
            axes[i][j].plot(circle_x, circle_y, 'k--', alpha=0.3)
            axes[i][j].scatter(a_x, a_y, color='blue', s=100, label=f'a={a}')
            axes[i][j].scatter(b_x, b_y, color='green', s=100, label=f'b={b}')
            axes[i][j].scatter(result_x, result_y, color='red', s=100, label=f'a+b={result}')
            
            # Draw arrows
            axes[i][j].arrow(0, 0, a_x * 0.8, a_y * 0.8,
                           head_width=0.05, head_length=0.1, 
                           fc='blue', ec='blue', alpha=0.5)
            axes[i][j].arrow(0, 0, b_x * 0.8, b_y * 0.8,
                           head_width=0.05, head_length=0.1, 
                           fc='green', ec='green', alpha=0.5)
            axes[i][j].arrow(0, 0, result_x * 0.8, result_y * 0.8,
                           head_width=0.05, head_length=0.1, 
                           fc='red', ec='red', alpha=0.5)
            
            axes[i][j].set_title(f'{a} + {b} = {result} (mod {T})')
            axes[i][j].set_aspect('equal')
            axes[i][j].grid(True, alpha=0.3)
            axes[i][j].set_xlim(-1.2, 1.2)
            axes[i][j].set_ylim(-1.2, 1.2)
            
            if i == 0:
                axes[i][j].legend()
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def simulate_clock_algorithm(
    a: int,
    b: int,
    periods: List[int] = [2, 5, 10, 100],
    plot_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate the Clock algorithm for addition.
    
    Args:
        a: First operand
        b: Second operand
        periods: List of periods for the Fourier features
        plot_path: Optional path to save the plot
        
    Returns:
        dict: Dictionary with simulation results
    """
    result = a + b
    
    # Generate helix basis vectors for a, b, and a+b
    basis_a = generate_helix_basis([a], periods)[0]
    basis_b = generate_helix_basis([b], periods)[0]
    basis_sum = generate_helix_basis([result], periods)[0]
    
    # Simulate calculation
    # In theory, for each period T, the trig identity cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    # and sin(a+b) = sin(a)cos(b) + cos(a)sin(b) applies
    
    # Initialize combined basis
    basis_combined = np.zeros_like(basis_sum)
    basis_combined[0] = basis_a[0] + basis_b[0]  # Linear component adds directly
    
    # For each period, apply trig identities
    for i, T in enumerate(periods):
        cos_a_idx = 2*i + 1
        sin_a_idx = 2*i + 2
        
        cos_a = basis_a[cos_a_idx]
        sin_a = basis_a[sin_a_idx]
        cos_b = basis_b[cos_a_idx]
        sin_b = basis_b[sin_a_idx]
        
        # Apply identities
        cos_sum = cos_a * cos_b - sin_a * sin_b
        sin_sum = sin_a * cos_b + cos_a * sin_b
        
        basis_combined[cos_a_idx] = cos_sum
        basis_combined[sin_a_idx] = sin_sum
    
    # Compute errors between ideal and simulated
    error = np.abs(basis_combined - basis_sum)
    
    # Create result dictionary
    result = {
        'a': a,
        'b': b,
        'sum': a + b,
        'basis_a': basis_a,
        'basis_b': basis_b,
        'basis_sum': basis_sum,
        'basis_combined': basis_combined,
        'error': error,
        'mean_error': np.mean(error),
        'max_error': np.max(error)
    }
    
    # Plot if requested
    if plot_path:
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(basis_sum))
        labels = ['linear']
        for T in periods:
            labels.extend([f'cos(T={T})', f'sin(T={T})'])
        
        plt.bar(x - 0.2, basis_sum, width=0.2, label='Ideal a+b')
        plt.bar(x, basis_combined, width=0.2, label='Simulated a+b')
        plt.bar(x + 0.2, error, width=0.2, label='Error')
        
        plt.xlabel('Component')
        plt.ylabel('Value')
        plt.title(f'Clock Algorithm Simulation: {a} + {b} = {a+b}')
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return result

if __name__ == "__main__":
    print("This module demonstrates the Clock algorithm for addition.")
    print("Import and use these functions in your main script or notebook.") 