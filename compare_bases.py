import numpy as np
import matplotlib.pyplot as plt
from ClockAlgorithm import ClockAlgorithm
import os

# Create output directory for figures
os.makedirs("figures/comparison", exist_ok=True)

# Initialize both base-10 and base-20 Clock Algorithms
base10_periods = [2, 5, 10, 100]
base10_algo = ClockAlgorithm(base=10, periods=base10_periods)

base20_periods = [2, 4, 5, 20, 400]
base20_algo = ClockAlgorithm(base=20, periods=base20_periods)

def compare_accuracy(max_number=19):
    """
    Compare the accuracy of base-10 and base-20 Clock Algorithms
    """
    # Evaluate base-10 accuracy
    base10_results = base10_algo.evaluate_accuracy(max_number)
    
    # Evaluate base-20 accuracy
    base20_results = base20_algo.evaluate_accuracy(max_number)
    
    # Print results
    print("--- Accuracy Comparison ---")
    print(f"Base-10: {base10_results['accuracy'] * 100:.2f}% ({base10_results['correct']}/{base10_results['total_tests']})")
    print(f"Base-20: {base20_results['accuracy'] * 100:.2f}% ({base20_results['correct']}/{base20_results['total_tests']})")
    
    # Create a bar chart to compare accuracies
    fig, ax = plt.subplots(figsize=(8, 6))
    bases = ['Base-10', 'Base-20']
    accuracies = [base10_results['accuracy'] * 100, base20_results['accuracy'] * 100]
    
    ax.bar(bases, accuracies, color=['blue', 'green'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison between Base-10 and Base-20')
    ax.set_ylim(0, 100)
    
    # Add accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.savefig("figures/comparison/accuracy_comparison.png")
    plt.close(fig)
    
    return base10_results, base20_results

def analyze_error_patterns(base10_results, base20_results):
    """
    Analyze and visualize error patterns for both bases
    """
    # Create error maps (matrices) for both bases
    max_num = int(np.sqrt(base10_results['total_tests'])) - 1
    
    base10_error_map = np.zeros((max_num + 1, max_num + 1))
    for a, b, expected, actual in base10_results['errors']:
        base10_error_map[a, b] = 1
    
    base20_error_map = np.zeros((max_num + 1, max_num + 1))
    for a, b, expected, actual in base20_results['errors']:
        base20_error_map[a, b] = 1
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    im1 = ax1.imshow(base10_error_map, cmap='Reds', origin='lower')
    ax1.set_title('Base-10 Addition Errors')
    ax1.set_xlabel('b')
    ax1.set_ylabel('a')
    ax1.set_xticks(np.arange(max_num + 1))
    ax1.set_yticks(np.arange(max_num + 1))
    
    im2 = ax2.imshow(base20_error_map, cmap='Reds', origin='lower')
    ax2.set_title('Base-20 Addition Errors')
    ax2.set_xlabel('b')
    ax2.set_ylabel('a')
    ax2.set_xticks(np.arange(max_num + 1))
    ax2.set_yticks(np.arange(max_num + 1))
    
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("figures/comparison/error_patterns.png")
    plt.close(fig)

def compare_specific_additions():
    """
    Compare specific addition cases between base-10 and base-20
    """
    # Selected test cases
    test_cases = [
        (7, 12),   # From the paper example
        (15, 17),  # Sum > 30
        (19, 19),  # Maximum test numbers
        (8, 8),    # Power of 2
        (5, 15)    # Mixed
    ]
    
    print("\n--- Specific Addition Comparisons ---")
    
    # Compare results
    for a, b in test_cases:
        base10_result = base10_algo.add(a, b)
        base10_expected = (a + b) % 100
        
        base20_result = base20_algo.add(a, b)
        base20_expected = (a + b) % 400
        
        print(f"Addition: {a} + {b}")
        print(f"  Base-10: Result = {base10_result}, Expected = {base10_expected}, Correct = {base10_result == base10_expected}")
        print(f"  Base-20: Result = {base20_result}, Expected = {base20_expected}, Correct = {base20_result == base20_expected}")
        
        # Create a visualization comparing the two
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Base-10 visualization
        base10_helix_a = base10_algo.create_helix(a)
        base10_helix_b = base10_algo.create_helix(b)
        base10_combined = base10_algo.combine_helices_using_clock(base10_helix_a, base10_helix_b)
        
        # Base-20 visualization
        base20_helix_a = base20_algo.create_helix(a)
        base20_helix_b = base20_algo.create_helix(b)
        base20_combined = base20_algo.combine_helices_using_clock(base20_helix_a, base20_helix_b)
        
        # Plot for base-10
        ax1.set_title(f"Base-10: {a} + {b} = {base10_result}")
        
        # Create subplots for each period in base-10
        for i, period in enumerate(base10_periods):
            cos_idx = 1 + i * 2
            sin_idx = 2 + i * 2
            
            # Calculate subplot position
            subplot_rows = len(base10_periods) // 2 + len(base10_periods) % 2
            subplot_cols = 2
            subplot_idx = i + 1
            
            # Create subplot
            ax1_sub = plt.subplot2grid((subplot_rows, 2), (i // 2, i % 2), fig=fig)
            
            # Create circle
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            ax1_sub.plot(x, y, 'k--', alpha=0.3)
            
            # Plot a
            cos_a, sin_a = base10_helix_a[cos_idx], base10_helix_a[sin_idx]
            ax1_sub.scatter(cos_a, sin_a, c='red', s=50, alpha=0.5, label=f'a={a}')
            
            # Plot b
            cos_b, sin_b = base10_helix_b[cos_idx], base10_helix_b[sin_idx]
            ax1_sub.scatter(cos_b, sin_b, c='blue', s=50, alpha=0.5, label=f'b={b}')
            
            # Plot a+b
            cos_sum, sin_sum = base10_combined[cos_idx], base10_combined[sin_idx]
            ax1_sub.scatter(cos_sum, sin_sum, c='green', s=100, label=f'a+b={base10_result}')
            
            ax1_sub.set_title(f"T={period}")
            ax1_sub.set_aspect('equal')
            ax1_sub.set_xlim(-1.2, 1.2)
            ax1_sub.set_ylim(-1.2, 1.2)
            ax1_sub.grid(True)
            
            if i == 0:
                ax1_sub.legend()
        
        # Plot for base-20
        ax2.set_title(f"Base-20: {a} + {b} = {base20_result}")
        
        # Create subplots for each period in base-20
        for i, period in enumerate(base20_periods):
            cos_idx = 1 + i * 2
            sin_idx = 2 + i * 2
            
            # Calculate subplot position
            subplot_rows = len(base20_periods) // 2 + len(base20_periods) % 2
            subplot_cols = 2
            subplot_idx = i + 1
            
            # Create subplot
            ax2_sub = plt.subplot2grid((subplot_rows, 2), (i // 2, i % 2), fig=fig, sharey=ax1)
            
            # Create circle
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            ax2_sub.plot(x, y, 'k--', alpha=0.3)
            
            # Plot a
            cos_a, sin_a = base20_helix_a[cos_idx], base20_helix_a[sin_idx]
            ax2_sub.scatter(cos_a, sin_a, c='red', s=50, alpha=0.5, label=f'a={a}')
            
            # Plot b
            cos_b, sin_b = base20_helix_b[cos_idx], base20_helix_b[sin_idx]
            ax2_sub.scatter(cos_b, sin_b, c='blue', s=50, alpha=0.5, label=f'b={b}')
            
            # Plot a+b
            cos_sum, sin_sum = base20_combined[cos_idx], base20_combined[sin_idx]
            ax2_sub.scatter(cos_sum, sin_sum, c='green', s=100, label=f'a+b={base20_result}')
            
            ax2_sub.set_title(f"T={period}")
            ax2_sub.set_aspect('equal')
            ax2_sub.set_xlim(-1.2, 1.2)
            ax2_sub.set_ylim(-1.2, 1.2)
            ax2_sub.grid(True)
            
            if i == 0:
                ax2_sub.legend()
        
        plt.tight_layout()
        plt.savefig(f"figures/comparison/addition_{a}_{b}_comparison.png")
        plt.close(fig)

def analyze_period_contributions():
    """
    Analyze how different periods contribute to the accuracy of addition
    """
    # Test different period subsets
    base10_period_configs = [
        ([2, 5, 10, 100], "All periods"),
        ([10, 100], "Only T=10,100"),
        ([2, 5], "Only T=2,5"),
        ([100], "Only T=100"),
        ([10], "Only T=10"),
        ([2], "Only T=2")
    ]
    
    base20_period_configs = [
        ([2, 4, 5, 20, 400], "All periods"),
        ([20, 400], "Only T=20,400"),
        ([2, 4, 5], "Only T=2,4,5"),
        ([400], "Only T=400"),
        ([20], "Only T=20"),
        ([2], "Only T=2")
    ]
    
    max_num = 19  # Same as in test_clock_algorithm.py
    
    # Calculate accuracy for each configuration
    base10_accuracies = []
    base20_accuracies = []
    
    print("\n--- Period Contribution Analysis ---")
    print("Base-10 period configurations:")
    
    for periods, label in base10_period_configs:
        algo = ClockAlgorithm(base=10, periods=periods)
        results = algo.evaluate_accuracy(max_num)
        accuracy = results['accuracy'] * 100
        base10_accuracies.append((label, accuracy))
        print(f"  {label}: {accuracy:.2f}%")
    
    print("\nBase-20 period configurations:")
    
    for periods, label in base20_period_configs:
        algo = ClockAlgorithm(base=20, periods=periods)
        results = algo.evaluate_accuracy(max_num)
        accuracy = results['accuracy'] * 100
        base20_accuracies.append((label, accuracy))
        print(f"  {label}: {accuracy:.2f}%")
    
    # Create bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Base-10 chart
    labels10, values10 = zip(*base10_accuracies)
    x10 = range(len(labels10))
    ax1.bar(x10, values10, color='blue', alpha=0.7)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Base-10 Period Contribution to Accuracy')
    ax1.set_xticks(x10)
    ax1.set_xticklabels(labels10, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    
    # Add accuracy values on top of the bars
    for i, v in enumerate(values10):
        ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Base-20 chart
    labels20, values20 = zip(*base20_accuracies)
    x20 = range(len(labels20))
    ax2.bar(x20, values20, color='green', alpha=0.7)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Base-20 Period Contribution to Accuracy')
    ax2.set_xticks(x20)
    ax2.set_xticklabels(labels20, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    
    # Add accuracy values on top of the bars
    for i, v in enumerate(values20):
        ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("figures/comparison/period_contribution.png")
    plt.close(fig)

def main():
    # Compare accuracy
    base10_results, base20_results = compare_accuracy()
    
    # Analyze error patterns
    analyze_error_patterns(base10_results, base20_results)
    
    # Compare specific addition cases
    compare_specific_additions()
    
    # Analyze period contributions
    analyze_period_contributions()
    
    print("\nComparison completed. Check the 'figures/comparison' directory for result visualizations.")

if __name__ == "__main__":
    main() 