"""
Number representation analysis for language models.

This module provides functions to analyze how language models
represent numbers internally, specifically focusing on the
helix representation described in the paper.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Union, Optional, Any
import os

from utils import to_numpy, fftn_mean, save_results, load_results

def extract_number_representations(
    model: Any, 
    tokenizer: Any, 
    numbers: List[int], 
    layer: int = 0,
    device: str = "cuda",
    cache_file: Optional[str] = None
) -> np.ndarray:
    """
    Extract hidden representations for numbers from a language model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        numbers: List of numbers to extract representations for
        layer: Layer to extract representations from (default 0)
        device: Device to run inference on
        cache_file: Optional path to cache results
        
    Returns:
        np.ndarray: Matrix of shape [len(numbers), hidden_size] with number representations
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached representations from {cache_file}")
        return np.load(cache_file)
    
    print(f"Extracting representations for {len(numbers)} numbers at layer {layer}...")
    representations = []
    
    for number in numbers:
        # Tokenize number
        inputs = tokenizer(str(number), return_tensors="pt").to(device)
        
        # Forward pass with no gradient calculation
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get hidden state for the number token at specified layer
            hidden_states = outputs.hidden_states[layer]
            number_rep = hidden_states[0, 0].clone()  # First token representation
            
            representations.append(to_numpy(number_rep))
    
    # Stack into a matrix
    representations_matrix = np.stack(representations, axis=0)
    
    # Cache results if path provided
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, representations_matrix)
        
    return representations_matrix

def analyze_fourier_structure(
    representations: np.ndarray,
    numbers: List[int],
    max_period: int = 100,
    result_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Analyze the Fourier structure of number representations.
    
    Args:
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        numbers: List of numbers corresponding to the representations
        max_period: Maximum period to analyze
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with Fourier analysis results
    """
    print("Analyzing Fourier structure...")
    
    # Center the representations
    centered = representations - np.mean(representations, axis=0, keepdims=True)
    
    # Compute FFT along the number dimension
    fft_result = np.fft.fftn(centered, axes=[0])
    
    # Calculate magnitude and average across hidden dimensions
    magnitude = np.abs(fft_result)
    mean_magnitude = np.mean(magnitude, axis=1)
    
    # Find peak frequencies
    frequencies = np.fft.fftfreq(len(numbers))
    periods = 1 / frequencies[1:]  # Skip DC component
    
    # Filter for positive periods up to max_period
    valid_indices = np.where((periods > 0) & (periods <= max_period))[0]
    valid_periods = periods[valid_indices]
    valid_magnitudes = mean_magnitude[1:][valid_indices]  # Skip DC component
    
    # Sort by magnitude
    sorted_indices = np.argsort(-valid_magnitudes)
    top_periods = valid_periods[sorted_indices]
    top_magnitudes = valid_magnitudes[sorted_indices]
    
    results = {
        'raw_fft': mean_magnitude,
        'frequencies': frequencies,
        'top_periods': top_periods[:20],  # Top 20 periods
        'top_magnitudes': top_magnitudes[:20],  # Top 20 magnitudes
    }
    
    if result_path:
        save_results(results, result_path)
        
    return results

def plot_fourier_analysis(
    fourier_results: Dict[str, np.ndarray],
    plot_path: Optional[str] = None
):
    """
    Plot the results of Fourier analysis.
    
    Args:
        fourier_results: Dictionary with Fourier analysis results
        plot_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Get positive frequencies only (up to Nyquist frequency)
    freqs = fourier_results['frequencies']
    magnitudes = fourier_results['raw_fft']
    
    pos_indices = np.where(freqs > 0)[0]
    pos_freqs = freqs[pos_indices]
    pos_magnitudes = magnitudes[pos_indices]
    
    # Convert to periods for readability
    periods = 1 / pos_freqs
    
    # Sort by period
    sorted_indices = np.argsort(periods)
    sorted_periods = periods[sorted_indices]
    sorted_magnitudes = pos_magnitudes[sorted_indices]
    
    # Plot
    plt.plot(sorted_periods, sorted_magnitudes)
    
    # Highlight key periods
    for period, magnitude in zip(fourier_results['top_periods'][:5], fourier_results['top_magnitudes'][:5]):
        plt.axvline(x=period, color='r', linestyle='--', alpha=0.5)
        plt.text(period, magnitude, f"T={period:.1f}", 
                 horizontalalignment='center', verticalalignment='bottom')
    
    plt.xlabel('Period')
    plt.ylabel('Magnitude')
    plt.title('Fourier Analysis of Number Representations')
    plt.xlim(0, 20)  # Limit x-axis for readability
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def analyze_linear_structure(
    representations: np.ndarray,
    numbers: List[int],
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the linear structure of number representations using PCA.
    
    Args:
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        numbers: List of numbers corresponding to the representations
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with PCA analysis results
    """
    print("Analyzing linear structure...")
    
    # Perform PCA
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(representations)
    
    # Calculate R² of linear fit to PC1
    pc1 = pca_result[:, 0]
    numbers_array = np.array(numbers)
    
    # Linear regression on PC1 vs numbers
    A = np.vstack([numbers_array, np.ones(len(numbers_array))]).T
    m, c = np.linalg.lstsq(A, pc1, rcond=None)[0]
    
    # Predicted values
    pc1_pred = m * numbers_array + c
    
    # R² calculation
    ss_tot = np.sum((pc1 - np.mean(pc1)) ** 2)
    ss_res = np.sum((pc1 - pc1_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    results = {
        'pca_components': pca_result,
        'explained_variance': pca.explained_variance_ratio_,
        'pc1_linear_coef': m,
        'pc1_linear_intercept': c,
        'pc1_r_squared': r_squared,
    }
    
    if result_path:
        save_results(results, result_path)
        
    return results

def plot_linear_structure(
    pca_results: Dict[str, Any],
    numbers: List[int],
    plot_path: Optional[str] = None
):
    """
    Plot the results of linear structure analysis.
    
    Args:
        pca_results: Dictionary with PCA analysis results
        numbers: List of numbers corresponding to the representations
        plot_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract PC1 values
    pc1 = pca_results['pca_components'][:, 0]
    
    # Sort by number value for clean plotting
    sort_idx = np.argsort(numbers)
    sorted_numbers = np.array(numbers)[sort_idx]
    sorted_pc1 = pc1[sort_idx]
    
    # Plot PC1 vs numbers
    plt.scatter(sorted_numbers, sorted_pc1, alpha=0.7, s=30)
    
    # Plot linear fit
    m = pca_results['pc1_linear_coef']
    c = pca_results['pc1_linear_intercept']
    r2 = pca_results['pc1_r_squared']
    
    x_line = np.array([min(numbers), max(numbers)])
    y_line = m * x_line + c
    
    plt.plot(x_line, y_line, 'r-', linewidth=2, 
             label=f'Linear fit: R² = {r2:.3f}')
    
    plt.xlabel('Number')
    plt.ylabel('PC1')
    plt.title('Linear Structure of Number Representations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_number_similarities(
    representations: np.ndarray,
    numbers: List[int],
    metric: str = 'euclidean',
    plot_path: Optional[str] = None
):
    """
    Visualize similarities between number representations.
    
    Args:
        representations: Matrix of hidden representations [len(numbers), hidden_size]
        numbers: List of numbers corresponding to the representations
        metric: Similarity metric ('euclidean' or 'cosine')
        plot_path: Optional path to save the plot
    """
    n_numbers = len(numbers)
    similarity_matrix = np.zeros((n_numbers, n_numbers))
    
    # Compute similarity matrix
    for i in range(n_numbers):
        for j in range(n_numbers):
            if metric == 'euclidean':
                similarity_matrix[i, j] = np.linalg.norm(
                    representations[i] - representations[j])
            elif metric == 'cosine':
                similarity_matrix[i, j] = np.dot(
                    representations[i], representations[j]) / (
                    np.linalg.norm(representations[i]) * np.linalg.norm(representations[j]))
    
    # Plot the similarity matrix
    plt.figure(figsize=(10, 8))
    
    if metric == 'euclidean':
        # Lower values (blue) indicate similarity for euclidean distance
        plt.imshow(similarity_matrix, cmap='viridis')
        plt.title('Euclidean Distance Between Number Representations')
        plt.colorbar(label='Distance')
    else:
        # Higher values (red) indicate similarity for cosine similarity
        plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Cosine Similarity Between Number Representations')
        plt.colorbar(label='Similarity')
    
    # Only show a subset of ticks for readability if many numbers
    tick_step = max(1, n_numbers // 10)
    tick_indices = range(0, n_numbers, tick_step)
    tick_labels = [numbers[i] for i in tick_indices]
    
    plt.xticks(tick_indices, tick_labels)
    plt.yticks(tick_indices, tick_labels)
    
    plt.xlabel('Number')
    plt.ylabel('Number')
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    print("This module provides functions to analyze number representations in LLMs.")
    print("Import and use these functions in your main script or notebook.") 