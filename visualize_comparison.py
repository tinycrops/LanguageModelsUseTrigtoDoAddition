#!/usr/bin/env python
"""
Visualize the comparison results from advanced training of different model architectures.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results():
    """Load results from the CSV file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results/advanced/comparison")
    results_file = os.path.join(results_dir, "normalized_results.csv")
    
    print(f"Looking for results file at: {results_file}")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    print(f"Results file found! Loading data...")
    return pd.read_csv(results_file)

def create_comparison_plots(df):
    """Create comparison plots for different metrics."""
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create output directory with absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "results/advanced/comparison/plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # 1. Create bar plot for each operation comparing models
    operations = df['Task'].unique()
    metrics = ['MSE_normalized', 'MAE_normalized', 'R2']
    models = df['Model'].unique()
    
    print(f"Operations found: {operations}")
    print(f"Models found: {models}")
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15), sharex=True)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = []
        
        for op in operations:
            for model in models:
                value = df[(df['Task'] == op) & (df['Model'] == model)][metric].values[0]
                metric_data.append({
                    'Operation': op,
                    'Model': model,
                    'Value': value
                })
        
        plot_df = pd.DataFrame(metric_data)
        
        # For R2, standard scale is fine; for normalized MSE and MAE, we're already using normalized values
        sns.barplot(x='Operation', y='Value', hue='Model', data=plot_df, ax=ax)
        ax.set_ylabel(f"{metric}")
        
        ax.set_title(f"{metric} by Operation and Model")
        ax.legend(title="Model")
    
    plt.tight_layout()
    metrics_by_op_file = os.path.join(output_dir, "metrics_by_operation.png")
    plt.savefig(metrics_by_op_file, dpi=300)
    plt.close()
    print(f"Saved bar plots to: {metrics_by_op_file}")
    
    # 2. Create a heatmap for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create a pivot table for the heatmap
        pivot_df = df.pivot(index='Task', columns='Model', values=metric)
        
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
        ax.set_title(f"{metric} Across Operations and Models")
    
    plt.tight_layout()
    heatmap_file = os.path.join(output_dir, "metrics_heatmap.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Saved heatmap to: {heatmap_file}")
    
    # 3. Create a radar chart to compare models across operations
    for metric in metrics:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of operations (categories)
        num_ops = len(operations)
        
        # Compute angle for each category
        angles = np.linspace(0, 2*np.pi, num_ops, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for model in models:
            values = []
            for op in operations:
                val = df[(df['Task'] == op) & (df['Model'] == model)][metric].values[0]
                values.append(val)
            
            # Close the loop
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(operations)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(f"{metric} Comparison Across Operations")
        plt.tight_layout()
        radar_file = os.path.join(output_dir, f"{metric}_radar.png")
        plt.savefig(radar_file, dpi=300)
        plt.close()
        print(f"Saved radar chart for {metric} to: {radar_file}")

def main():
    """Main function to create visualization."""
    print("Starting visualization...")
    df = load_results()
    
    if df is not None:
        print("Data loaded successfully. Creating plots...")
        create_comparison_plots(df)
        print("Visualization complete. Results saved to results/advanced/comparison/plots/")
    else:
        print("No results found to visualize.")

if __name__ == "__main__":
    main() 