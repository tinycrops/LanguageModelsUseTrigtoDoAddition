"""
Helix fitting module for representing numbers in language models.

This module implements the key finding from the paper: that language models
represent numbers as generalized helices with both periodic and linear components.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Any
import os
from scipy.optimize import minimize

from utils import to_numpy, save_results, load_results

def generate_helix_basis(
    numbers: List[int], 
    periods: List[int] = [2, 5, 10, 100]
) -> np.ndarray:
    """
    Generate a basis of helix functions for the given numbers.
    
    Args:
        numbers: List of numbers to generate basis for
        periods: List of periods for the Fourier features
        
    Returns:
        np.ndarray: Matrix of shape [len(numbers), 2*len(periods)+1] containing 
                    the basis functions evaluated at each number
    """
    n = len(numbers)
    k = len(periods)
    basis = np.zeros((n, 2*k + 1))
    
    # Linear component
    basis[:, 0] = numbers
    
    # Periodic components
    for i, T in enumerate(periods):
        basis[:, 2*i + 1] = np.cos(2 * np.pi * np.array(numbers) / T)
        basis[:, 2*i + 2] = np.sin(2 * np.pi * np.array(numbers) / T)
    
    return basis

def fit_helix(
    representations: np.ndarray,
    numbers: List[int],
    periods: List[int] = [2, 5, 10, 100],
    pca_dim: int = 100,
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fit a generalized helix to the number representations.
    
    Args:
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        numbers: List of numbers corresponding to the representations
        periods: List of periods for the Fourier features
        pca_dim: Dimension to reduce to with PCA before fitting
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with helix fitting results
    """
    print(f"Fitting helix with periods {periods}...")
    
    # Generate helix basis
    basis = generate_helix_basis(numbers, periods)
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=min(pca_dim, representations.shape[0], representations.shape[1]))
    pca_result = pca.fit_transform(representations)
    
    # Fit with linear regression
    reg = LinearRegression()
    reg.fit(basis, pca_result)
    
    # Get coefficients and project back to original space
    pca_coef = reg.coef_  # Shape [pca_dim, 2*len(periods)+1]
    full_coef = pca.components_.T @ pca_coef  # Shape [hidden_size, 2*len(periods)+1]
    
    # Calculate the quality of fit in PCA space
    pca_pred = reg.predict(basis)
    r2_pca = reg.score(basis, pca_result)
    
    # Calculate the quality of fit in original space
    full_pred = pca_pred @ pca.components_
    residuals = representations - full_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((representations - np.mean(representations, axis=0))**2)
    r2_full = 1 - (ss_res / ss_tot)
    
    # Save results
    results = {
        'coefficients': full_coef,
        'pca_coefficients': pca_coef,
        'periods': periods,
        'pca_components': pca.components_,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'r2_pca': r2_pca,
        'r2_full': r2_full,
        'basis': basis
    }
    
    if result_path:
        save_results(results, result_path)
    
    return results

def project_to_helix(
    representation: np.ndarray,
    helix_fit: Dict[str, Any]
) -> np.ndarray:
    """
    Project a representation onto the fitted helix.
    
    Args:
        representation: Vector representation to project [hidden_size]
        helix_fit: Dictionary with helix fitting results
        
    Returns:
        np.ndarray: Projection of the representation onto the helix basis
    """
    # Calculate the pseudo-inverse of the coefficient matrix
    coefficients = helix_fit['coefficients']
    pseudo_inv = np.linalg.pinv(coefficients)
    
    # Project the representation onto the helix basis
    projection = pseudo_inv @ representation
    
    return projection

def reconstruct_from_helix(
    projection: np.ndarray,
    helix_fit: Dict[str, Any]
) -> np.ndarray:
    """
    Reconstruct a representation from its helix projection.
    
    Args:
        projection: Helix basis projection [2*len(periods)+1]
        helix_fit: Dictionary with helix fitting results
        
    Returns:
        np.ndarray: Reconstructed representation in the original space
    """
    # Reconstruct using the coefficient matrix
    coefficients = helix_fit['coefficients']
    reconstruction = coefficients @ projection
    
    return reconstruction

def plot_helix_components(
    helix_fit: Dict[str, Any],
    plot_path: Optional[str] = None
):
    """
    Plot the magnitude of the helix components.
    
    Args:
        helix_fit: Dictionary with helix fitting results
        plot_path: Optional path to save the plot
    """
    coefficients = helix_fit['coefficients']
    periods = helix_fit['periods']
    
    # Calculate the magnitude of each component
    component_magnitudes = np.linalg.norm(coefficients, axis=0)
    
    # Create component labels
    labels = ['Linear']
    for period in periods:
        labels.extend([f'cos(T={period})', f'sin(T={period})'])
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, component_magnitudes)
    
    # Color the bars by component type
    colors = ['blue'] + [f'C{i//2}' for i in range(1, len(labels))]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Component')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of Helix Components')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_helix_subspace(
    representations: np.ndarray,
    numbers: List[int],
    helix_fit: Dict[str, Any],
    plot_path: Optional[str] = None
):
    """
    Visualize the helix subspace by projecting representations onto it.
    
    Args:
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        numbers: List of numbers corresponding to the representations
        helix_fit: Dictionary with helix fitting results
        plot_path: Optional path to save the plot
    """
    periods = helix_fit['periods']
    n_periods = len(periods)
    
    # Project all representations onto the helix
    projections = np.zeros((len(numbers), 2*n_periods+1))
    for i, rep in enumerate(representations):
        projections[i] = project_to_helix(rep, helix_fit)
    
    # Create a grid of plots (one for each period + one for linear component)
    fig, axs = plt.subplots(2, n_periods + 1, figsize=(5*(n_periods+1), 10))
    axs = axs.flatten()
    
    # Plot linear component
    axs[0].scatter(numbers, projections[:, 0])
    axs[0].set_title('Linear Component')
    axs[0].set_xlabel('Number')
    axs[0].set_ylabel('Projection')
    
    # Plot periodic components for each period
    for i, T in enumerate(periods):
        # Plot cosine vs sine for this period
        cos_idx = 2*i + 1
        sin_idx = 2*i + 2
        
        # Create color mapping by modulo T
        mod_values = np.array(numbers) % T
        unique_mods = sorted(set(mod_values))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_mods)))
        color_map = {mod: colors[j] for j, mod in enumerate(unique_mods)}
        
        # Scatter plot with colors
        for mod in unique_mods:
            mask = mod_values == mod
            axs[i+1].scatter(
                projections[mask, cos_idx], 
                projections[mask, sin_idx],
                color=color_map[mod],
                label=f'{mod}',
                alpha=0.7
            )
        
        # Plot unit circle as reference
        theta = np.linspace(0, 2*np.pi, 100)
        axs[i+1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        axs[i+1].set_title(f'Period T={T}')
        axs[i+1].set_xlabel(f'cos(2π/T * n)')
        axs[i+1].set_ylabel(f'sin(2π/T * n)')
        axs[i+1].set_aspect('equal')
        
        # Add number labels for a few points
        step = max(1, len(numbers) // 20)  # Label at most ~20 points
        for j in range(0, len(numbers), step):
            axs[i+1].annotate(
                str(numbers[j]), 
                (projections[j, cos_idx], projections[j, sin_idx]),
                fontsize=8
            )
        
        # Plot helical structure in 3D for selected periods
        if i < n_periods:
            ax_3d = fig.add_subplot(2, n_periods + 1, n_periods + 2 + i, projection='3d')
            
            # Sort by number for clean visualization
            sort_idx = np.argsort(numbers)
            sorted_nums = np.array(numbers)[sort_idx]
            sorted_cos = projections[sort_idx, cos_idx]
            sorted_sin = projections[sort_idx, sin_idx]
            
            # Plot the 3D helix
            ax_3d.plot3D(sorted_nums, sorted_cos, sorted_sin, 'b-')
            ax_3d.scatter3D(sorted_nums, sorted_cos, sorted_sin, c=sorted_nums, cmap='viridis')
            
            ax_3d.set_title(f'3D Helix for T={T}')
            ax_3d.set_xlabel('Number')
            ax_3d.set_ylabel(f'cos(2π/{T} * n)')
            ax_3d.set_zlabel(f'sin(2π/{T} * n)')
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def find_optimal_periods(
    representations: np.ndarray,
    numbers: List[int],
    max_periods: int = 4,
    candidate_periods: List[int] = [2, 3, 4, 5, 10, 20, 50, 100],
    pca_dim: int = 50,
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find the optimal set of periods for the helix fit by evaluating all combinations.
    
    Args:
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        numbers: List of numbers corresponding to the representations
        max_periods: Maximum number of periods to include
        candidate_periods: List of candidate periods to choose from
        pca_dim: Dimension to reduce to with PCA before fitting
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with optimal periods and fit results
    """
    from itertools import combinations
    
    print(f"Finding optimal periods from candidates {candidate_periods}...")
    
    best_r2 = -float('inf')
    best_periods = None
    best_fit = None
    
    # Try all combinations of periods up to max_periods
    for k in range(1, max_periods + 1):
        for period_combo in combinations(candidate_periods, k):
            # Fit helix with this set of periods
            fit_result = fit_helix(
                representations, 
                numbers, 
                list(period_combo), 
                pca_dim=pca_dim
            )
            
            # Check if this is the best fit so far
            if fit_result['r2_pca'] > best_r2:
                best_r2 = fit_result['r2_pca']
                best_periods = list(period_combo)
                best_fit = fit_result
                
                print(f"New best: periods={best_periods}, R²={best_r2:.4f}")
    
    results = {
        'optimal_periods': best_periods,
        'optimal_r2': best_r2,
        'all_candidates': candidate_periods,
        'best_fit': best_fit
    }
    
    if result_path:
        save_results(results, result_path)
    
    return results

def evaluate_helix_fit(
    numbers: List[int],
    representations: np.ndarray,
    helix_fit: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate the quality of a helix fit to the number representations.
    
    Args:
        numbers: List of numbers corresponding to the representations
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        helix_fit: Dictionary with helix fitting results from fit_helix
        
    Returns:
        dict: Dictionary with evaluation metrics (r2_score, mse)
    """
    # Generate helix basis
    basis = generate_helix_basis(numbers, helix_fit['periods'])
    
    # Compute predictions using the basis and coefficients
    coefficients = helix_fit['coefficients']
    predictions = basis @ coefficients.T
    
    # Calculate residuals
    residuals = representations - predictions
    
    # Calculate mean squared error
    mse = np.mean(np.sum(residuals**2, axis=1))
    
    # Calculate R² score
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((representations - np.mean(representations, axis=0))**2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'r2_score': r2_score,
        'mse': mse
    }

if __name__ == "__main__":
    print("This module provides functions to fit helical representations to number embeddings.")
    print("Import and use these functions in your main script or notebook.") 