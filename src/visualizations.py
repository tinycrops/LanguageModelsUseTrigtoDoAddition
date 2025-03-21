"""
Visualization module for the helical number representations and Clock algorithm.

This module provides functions for creating visualizations of the key findings,
including helical representations, activation patching results, and Clock algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
import os

def plot_accuracy_heatmap(
    a_range: List[int],
    b_range: List[int],
    accuracy_matrix: np.ndarray,
    model_name: str = None,
    plot_path: Optional[str] = None
):
    """
    Plot a heatmap of model accuracy on addition problems.
    
    Args:
        a_range: Range of values for the first operand
        b_range: Range of values for the second operand
        accuracy_matrix: Matrix of model accuracy [len(a_range), len(b_range)]
        model_name: Optional name of the model for the title
        plot_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        accuracy_matrix, 
        cmap='viridis',
        vmin=0, vmax=1,
        xticklabels=b_range[::10] if len(b_range) > 10 else b_range,
        yticklabels=a_range[::10] if len(a_range) > 10 else a_range
    )
    
    # Add labels
    plt.xlabel('b')
    plt.ylabel('a')
    
    if model_name:
        plt.title(f'Addition Accuracy for {model_name}')
    else:
        plt.title('Addition Accuracy')
    
    # Add colorbar label
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Accuracy')
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_helix_3d(
    numbers: List[int],
    helix_fit: Dict[str, Any],
    periods: List[int] = None,
    plot_path: Optional[str] = None
):
    """
    Create 3D plots of the helical structure for different periods.
    
    Args:
        numbers: List of numbers to plot
        helix_fit: Dictionary with helix fitting results
        periods: Optional list of specific periods to plot (defaults to all periods in helix_fit)
        plot_path: Optional path to save the plot
    """
    # Extract periods from helix fit if not provided
    if periods is None:
        periods = helix_fit['periods']
    
    # If only one period, make it a list
    if isinstance(periods, (int, float)):
        periods = [periods]
    
    # Sort numbers for a cleaner visualization
    sorted_indices = np.argsort(numbers)
    sorted_numbers = np.array(numbers)[sorted_indices]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(6 * len(periods), 5))
    
    for i, T in enumerate(periods):
        # Create 3D subplot
        ax = fig.add_subplot(1, len(periods), i+1, projection='3d')
        
        # Generate projected points for this period
        period_idx = helix_fit['periods'].index(T) if T in helix_fit['periods'] else None
        
        if period_idx is not None:
            # Generate helix basis
            basis = np.zeros((len(numbers), 2*len(helix_fit['periods'])+1))
            for j, num in enumerate(numbers):
                basis[j, 0] = num  # Linear component
                for k, period in enumerate(helix_fit['periods']):
                    basis[j, 2*k+1] = np.cos(2 * np.pi * num / period)
                    basis[j, 2*k+2] = np.sin(2 * np.pi * num / period)
            
            # Project to get the 3D helix components
            linear = sorted_numbers
            cos_idx = 2*period_idx + 1
            sin_idx = 2*period_idx + 2
            
            cos_vals = np.cos(2 * np.pi * sorted_numbers / T)
            sin_vals = np.sin(2 * np.pi * sorted_numbers / T)
            
            # Plot the 3D helix
            ax.plot3D(linear, cos_vals, sin_vals, 'b-')
            scatter = ax.scatter3D(
                linear, cos_vals, sin_vals,
                c=sorted_numbers,
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Number Value')
            
            # Label some points
            step = max(1, len(sorted_numbers) // 10)  # Label ~10 points
            for j in range(0, len(sorted_numbers), step):
                ax.text(
                    linear[j],
                    cos_vals[j],
                    sin_vals[j],
                    f"{sorted_numbers[j]}",
                    fontsize=8
                )
            
            # Set labels and title
            ax.set_xlabel('Linear Component (n)')
            ax.set_ylabel(f'cos(2π·n/{T})')
            ax.set_zlabel(f'sin(2π·n/{T})')
            ax.set_title(f'3D Helix Representation (T={T})')
        
        else:
            ax.text(0.5, 0.5, 0.5, f"Period T={T} not in helix fit", 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_circular_structure(
    numbers: List[int],
    periods: List[int] = [2, 5, 10, 100],
    n_samples: int = 100,
    plot_path: Optional[str] = None
):
    """
    Visualize the circular structure of numbers with different periods.
    
    Args:
        numbers: List of numbers to visualize
        periods: List of periods to visualize
        n_samples: Number of samples to use for circles
        plot_path: Optional path to save the plot
    """
    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(5*n_periods, 5))
    
    # If only one period, make axes iterable
    if n_periods == 1:
        axes = [axes]
    
    for i, T in enumerate(periods):
        # Generate circle points
        theta = np.linspace(0, 2*np.pi, n_samples)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        # Plot circle
        axes[i].plot(circle_x, circle_y, 'k--', alpha=0.3)
        
        # Calculate positions for each number
        for num in numbers:
            angle = 2 * np.pi * (num % T) / T
            x, y = np.cos(angle), np.sin(angle)
            
            # Color based on the value modulo T
            color = plt.cm.tab10(num % T / T)
            
            # Plot point
            axes[i].scatter(x, y, color=color, s=100, alpha=0.7)
            
            # Label point
            axes[i].annotate(
                str(num),
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8
            )
        
        # Add modulo class circles in background
        for j in range(T):
            angle = 2 * np.pi * j / T
            x, y = 0.9 * np.cos(angle), 0.9 * np.sin(angle)
            color = plt.cm.tab10(j / T)
            axes[i].add_patch(plt.Circle((x, y), 0.05, color=color, alpha=0.3))
            axes[i].annotate(
                f"{j} mod {T}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha='center',
                fontsize=8
            )
        
        # Set title and adjust plot
        axes[i].set_title(f'Numbers on Circle (T={T})')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-1.2, 1.2)
        axes[i].set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_activation_patching_comparison(
    patching_results: Dict[str, Any],
    best_layer: Optional[int] = None,
    plot_path: Optional[str] = None
):
    """
    Plot a comparison of activation patching methods.
    
    Args:
        patching_results: Dictionary with patching results
        best_layer: Optional layer to highlight
        plot_path: Optional path to save the plot
    """
    layers = patching_results['layers']
    
    # Collect data for different methods
    layer_patch = patching_results['avg_layer_patch']
    helix_patch = patching_results['avg_helix_patch']
    pca_patch = patching_results['avg_pca_patch']
    random_patch = patching_results['avg_random_patch']
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.2
    positions = np.arange(len(layers))
    
    # Plot bars for each method
    plt.bar(positions - 1.5*bar_width, layer_patch, bar_width, 
            label='Layer Patch (Full)', color='blue')
    plt.bar(positions - 0.5*bar_width, helix_patch, bar_width, 
            label='Helix Fit', color='green')
    plt.bar(positions + 0.5*bar_width, pca_patch, bar_width, 
            label='PCA Baseline', color='orange')
    plt.bar(positions + 1.5*bar_width, random_patch, bar_width, 
            label='Random Baseline', color='red')
    
    # Highlight best layer if provided
    if best_layer is not None and best_layer in layers:
        idx = layers.index(best_layer)
        plt.axvline(x=positions[idx], color='gray', linestyle='--', alpha=0.7)
        plt.text(positions[idx], max(layer_patch), f"Best Layer: {best_layer}", 
                 ha='center', va='bottom')
    
    # Add labels and legend
    plt.xlabel('Layer')
    plt.ylabel('Logit Difference')
    plt.title('Comparison of Activation Patching Methods Across Layers')
    plt.xticks(positions, layers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text summarizing method performance
    avg_helix_perf = np.mean(helix_patch) / np.mean(layer_patch) * 100
    avg_pca_perf = np.mean(pca_patch) / np.mean(layer_patch) * 100
    
    plt.figtext(0.02, 0.02, 
                f"Average Helix Performance: {avg_helix_perf:.1f}% of Layer Patch\n"
                f"Average PCA Performance: {avg_pca_perf:.1f}% of Layer Patch",
                ha="left", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_clock_algorithm_for_numbers(
    a: int, 
    b: int, 
    periods: List[int] = [2, 5, 10, 100],
    plot_path: Optional[str] = None
):
    """
    Visualize the Clock algorithm for specific numbers.
    
    Args:
        a: First operand
        b: Second operand
        periods: List of periods for the Fourier features
        plot_path: Optional path to save the plot
    """
    result = a + b
    
    # Create a figure with subplots for each period
    fig, axes = plt.subplots(len(periods), 3, figsize=(15, 5*len(periods)))
    
    # If only one period, make axes iterable
    if len(periods) == 1:
        axes = [axes]
    
    # For each period
    for i, T in enumerate(periods):
        # Compute angles and positions
        a_angle = 2 * np.pi * (a % T) / T
        b_angle = 2 * np.pi * (b % T) / T
        result_angle = 2 * np.pi * (result % T) / T
        
        a_pos = (np.cos(a_angle), np.sin(a_angle))
        b_pos = (np.cos(b_angle), np.sin(b_angle))
        result_pos = (np.cos(result_angle), np.sin(result_angle))
        
        # Create circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x, circle_y = np.cos(theta), np.sin(theta)
        
        # First subplot: a
        axes[i][0].plot(circle_x, circle_y, 'k--', alpha=0.3)
        axes[i][0].scatter(a_pos[0], a_pos[1], s=100, color='blue', label=f'a={a}')
        axes[i][0].scatter(0, 0, s=30, color='black')
        
        # Add clock face numbers
        for j in range(T):
            angle = 2 * np.pi * j / T
            pos = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
            axes[i][0].text(pos[0], pos[1], str(j), ha='center', va='center', fontsize=10)
        
        axes[i][0].set_title(f'Step 1: Position for a={a} (mod {T})')
        axes[i][0].set_aspect('equal')
        axes[i][0].set_xlim(-1.2, 1.2)
        axes[i][0].set_ylim(-1.2, 1.2)
        axes[i][0].grid(True, alpha=0.3)
        axes[i][0].legend()
        
        # Second subplot: adding b
        axes[i][1].plot(circle_x, circle_y, 'k--', alpha=0.3)
        axes[i][1].scatter(a_pos[0], a_pos[1], s=100, color='blue', label=f'a={a}')
        axes[i][1].scatter(b_pos[0], b_pos[1], s=100, color='green', label=f'b={b}')
        axes[i][1].scatter(0, 0, s=30, color='black')
        
        # Add clock face numbers
        for j in range(T):
            angle = 2 * np.pi * j / T
            pos = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
            axes[i][1].text(pos[0], pos[1], str(j), ha='center', va='center', fontsize=10)
        
        # Draw arrow showing rotation
        axes[i][1].arrow(a_pos[0], a_pos[1], 
                        (result_pos[0] - a_pos[0]) * 0.8, 
                        (result_pos[1] - a_pos[1]) * 0.8,
                        head_width=0.05, head_length=0.1, 
                        fc='red', ec='red')
        
        axes[i][1].set_title(f'Step 2: Rotate by b={b} (mod {T})')
        axes[i][1].set_aspect('equal')
        axes[i][1].set_xlim(-1.2, 1.2)
        axes[i][1].set_ylim(-1.2, 1.2)
        axes[i][1].grid(True, alpha=0.3)
        axes[i][1].legend()
        
        # Third subplot: result
        axes[i][2].plot(circle_x, circle_y, 'k--', alpha=0.3)
        axes[i][2].scatter(result_pos[0], result_pos[1], s=100, color='red', label=f'a+b={result}')
        axes[i][2].scatter(0, 0, s=30, color='black')
        
        # Add clock face numbers
        for j in range(T):
            angle = 2 * np.pi * j / T
            pos = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
            axes[i][2].text(pos[0], pos[1], str(j), ha='center', va='center', fontsize=10)
        
        axes[i][2].set_title(f'Step 3: Result a+b={result} (mod {T})')
        axes[i][2].set_aspect('equal')
        axes[i][2].set_xlim(-1.2, 1.2)
        axes[i][2].set_ylim(-1.2, 1.2)
        axes[i][2].grid(True, alpha=0.3)
        axes[i][2].legend()
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_summary_figure(
    numbers: List[int],
    helix_fit: Dict[str, Any],
    patching_results: Dict[str, Any],
    sample_a: int = 3,
    sample_b: int = 63,
    plot_path: Optional[str] = None
):
    """
    Create a summary figure combining key findings.
    
    Args:
        numbers: List of numbers used in analysis
        helix_fit: Dictionary with helix fitting results
        patching_results: Dictionary with patching results
        sample_a: Sample 'a' value for Clock algorithm demonstration
        sample_b: Sample 'b' value for Clock algorithm demonstration
        plot_path: Optional path to save the plot
    """
    periods = helix_fit['periods']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid
    grid = plt.GridSpec(3, 4, figure=fig, wspace=0.4, hspace=0.4)
    
    # 1. Fourier analysis plot (top-left)
    ax1 = fig.add_subplot(grid[0, 0:2])
    
    # Sample basis functions for visualization
    sample_basis = np.zeros((len(numbers), 2*len(periods)+1))
    for i, num in enumerate(numbers):
        sample_basis[i, 0] = num  # Linear component
        for j, period in enumerate(periods):
            sample_basis[i, 2*j+1] = np.cos(2 * np.pi * num / period)
            sample_basis[i, 2*j+2] = np.sin(2 * np.pi * num / period)
    
    # Plot magnitudes of basis components
    component_labels = ['Linear']
    for period in periods:
        component_labels.extend([f'cos(T={period})', f'sin(T={period})'])
    
    component_mags = np.linalg.norm(helix_fit['coefficients'], axis=0)
    
    ax1.bar(range(len(component_mags)), component_mags)
    ax1.set_xticks(range(len(component_mags)))
    ax1.set_xticklabels(component_labels, rotation=45)
    ax1.set_title('Helix Components Magnitude')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Circular representation (top-right)
    ax2 = fig.add_subplot(grid[0, 2:4])
    
    # Create visualization for T=10 (most prominent for base-10 numbers)
    T = 10
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x, circle_y = np.cos(theta), np.sin(theta)
    
    ax2.plot(circle_x, circle_y, 'k--', alpha=0.3)
    
    # Plot points for 0-9
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        angle = 2 * np.pi * i / T
        x, y = np.cos(angle), np.sin(angle)
        ax2.scatter(x, y, s=100, color=colors[i], label=str(i))
        ax2.text(1.1*x, 1.1*y, str(i), ha='center', va='center', fontsize=12)
    
    ax2.set_aspect('equal')
    ax2.set_title('Numbers on Circle (T=10)')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_axis_off()
    
    # 3. Activation patching results (middle row)
    ax3 = fig.add_subplot(grid[1, :])
    
    # Plot patching results
    layers = patching_results['layers']
    layer_patch = patching_results['avg_layer_patch']
    helix_patch = patching_results['avg_helix_patch']
    pca_patch = patching_results['avg_pca_patch']
    
    ax3.plot(layers, layer_patch, 'o-', label='Layer Patch', linewidth=2)
    ax3.plot(layers, helix_patch, 's-', label='Helix Fit', linewidth=2)
    ax3.plot(layers, pca_patch, '^-', label='PCA Baseline', linewidth=2)
    
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Logit Difference')
    ax3.set_title('Activation Patching Results')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Clock algorithm (bottom row)
    T = 10  # Use most interpretable period
    
    # Compute positions
    a = sample_a
    b = sample_b
    result = a + b
    
    a_angle = 2 * np.pi * (a % T) / T
    b_angle = 2 * np.pi * (b % T) / T
    result_angle = 2 * np.pi * (result % T) / T
    
    a_pos = (np.cos(a_angle), np.sin(a_angle))
    b_pos = (np.cos(b_angle), np.sin(b_angle))
    result_pos = (np.cos(result_angle), np.sin(result_angle))
    
    # Step 1: Start position
    ax4 = fig.add_subplot(grid[2, 0])
    ax4.plot(circle_x, circle_y, 'k--', alpha=0.3)
    ax4.scatter(a_pos[0], a_pos[1], s=150, color='blue', label=f'a={a}')
    ax4.scatter(0, 0, s=30, color='black')
    
    # Add clock face
    for i in range(T):
        angle = 2 * np.pi * i / T
        pos = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
        ax4.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=10)
    
    ax4.set_title(f'Step 1: Position for a={a}')
    ax4.set_aspect('equal')
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.legend()
    ax4.set_axis_off()
    
    # Step 2: Rotation
    ax5 = fig.add_subplot(grid[2, 1:3])
    ax5.plot(circle_x, circle_y, 'k--', alpha=0.3)
    ax5.scatter(a_pos[0], a_pos[1], s=150, color='blue', label=f'a={a}')
    ax5.scatter(b_pos[0], b_pos[1], s=150, color='green', label=f'b={b}')
    ax5.scatter(0, 0, s=30, color='black')
    
    # Add clock face
    for i in range(T):
        angle = 2 * np.pi * i / T
        pos = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
        ax5.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=10)
    
    # Draw arrow showing rotation
    ax5.arrow(a_pos[0], a_pos[1], 
              (result_pos[0] - a_pos[0]) * 0.8, 
              (result_pos[1] - a_pos[1]) * 0.8,
              head_width=0.05, head_length=0.1, 
              fc='red', ec='red', alpha=0.7, linewidth=2)
    
    ax5.set_title(f'Step 2: Rotate by b={b}')
    ax5.set_aspect('equal')
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.legend()
    ax5.set_axis_off()
    
    # Step 3: Result
    ax6 = fig.add_subplot(grid[2, 3])
    ax6.plot(circle_x, circle_y, 'k--', alpha=0.3)
    ax6.scatter(result_pos[0], result_pos[1], s=150, color='red', label=f'a+b={result}')
    ax6.scatter(0, 0, s=30, color='black')
    
    # Add clock face
    for i in range(T):
        angle = 2 * np.pi * i / T
        pos = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
        ax6.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=10)
    
    ax6.set_title(f'Step 3: Result a+b={result}')
    ax6.set_aspect('equal')
    ax6.set_xlim(-1.2, 1.2)
    ax6.set_ylim(-1.2, 1.2)
    ax6.legend()
    ax6.set_axis_off()
    
    # Add overall title
    fig.suptitle('Language Models Use Trigonometry to Do Addition', fontsize=20, y=0.98)
    
    # Add explanatory text
    fig.text(0.5, 0.02, 
             "LLMs represent numbers as generalized helices and manipulate these helices using the Clock algorithm to perform addition.",
             ha='center', fontsize=12)
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    print("This module provides visualization functions for the project.")
    print("Import and use these functions in your main script or notebook.") 