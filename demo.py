import numpy as np
import matplotlib.pyplot as plt
from ClockAlgorithm import ClockAlgorithm
from improved_base20 import ImprovedBase20Algorithm
import os

# Create output directory for figures
os.makedirs("figures/demo", exist_ok=True)

def main():
    print("=== CLOCK ALGORITHM DEMONSTRATION ===")
    print("This demo shows how language models use trigonometry to do addition")
    print("Based on the paper by Kantamneni & Tegmark (2025)")
    
    # Initialize algorithms
    base10_algo = ClockAlgorithm(base=10, periods=[2, 5, 10, 100])
    standard_base20_algo = ClockAlgorithm(base=20, periods=[2, 4, 5, 20, 400])
    improved_base20_algo = ImprovedBase20Algorithm()
    
    # Demonstration examples
    examples = [
        (7, 12),   # From the paper
        (19, 42),  # Larger numbers
        (5, 5),    # Same number
        (0, 15)    # Zero case
    ]
    
    # Loop through examples
    for a, b in examples:
        print(f"\n--- Example: {a} + {b} ---")
        
        # Compute expected results
        expected_base10 = (a + b) % base10_algo.modulus
        expected_base20 = (a + b) % standard_base20_algo.modulus
        
        # Compute results using the algorithms
        base10_result = base10_algo.add(a, b)
        standard_base20_result = standard_base20_algo.add(a, b)
        improved_base20_result = improved_base20_algo.add(a, b)
        
        # Print results
        print(f"Base-10 addition: {a} + {b} = {base10_result} (expected {expected_base10})")
        print(f"Standard Base-20 addition: {a} + {b} = {standard_base20_result} (expected {expected_base20})")
        print(f"Improved Base-20 addition: {a} + {b} = {improved_base20_result} (expected {expected_base20})")
        
        # Create step-by-step visualization
        create_step_visualization(a, b, base10_algo, standard_base20_algo, improved_base20_algo)
        
    print("\nDemonstration completed. Check the 'figures/demo' directory for visualizations.")
    print("For more detailed results, run the other scripts in the repository:")
    print("  python test_clock_algorithm.py    # Basic tests")
    print("  python compare_bases.py           # Comparison between base-10 and base-20")
    print("  python stress_test.py             # Performance with larger numbers and noise")
    print("  python improved_base20.py         # Improved base-20 implementation")

def create_step_visualization(a, b, base10_algo, standard_base20_algo, improved_base20_algo):
    """
    Create a visualization showing each step of the Clock Algorithm for all implementations
    """
    # Create helices
    base10_helix_a = base10_algo.create_helix(a)
    base10_helix_b = base10_algo.create_helix(b)
    base10_combined = base10_algo.combine_helices_using_clock(base10_helix_a, base10_helix_b)
    base10_result = base10_algo.decode_helix(base10_combined)
    
    # Create a step-by-step figure
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    # Title
    fig.suptitle(f"Step-by-Step: {a} + {b} using the Clock Algorithm", fontsize=16)
    
    # Row titles
    row_titles = [
        f"1. Create helix({a})",
        f"2. Create helix({b})",
        f"3. Combine helices using trigonometric identities",
        f"4. Decode helix({a} + {b}) = {base10_result}"
    ]
    
    # Column titles
    axes[0, 0].set_title("Base-10", fontsize=14)
    axes[0, 1].set_title("Base-20 (Standard)", fontsize=14)
    axes[0, 2].set_title("Base-20 (Improved)", fontsize=14)
    
    # Add row titles
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, fontsize=12, rotation=0, ha='right', va='center')
    
    # Draw helices for base-10
    draw_helix_step(axes[0, 0], base10_helix_a, base10_algo.periods, "Step 1: Base-10", "red", a)
    draw_helix_step(axes[1, 0], base10_helix_b, base10_algo.periods, "Step 2: Base-10", "blue", b)
    draw_helix_step(axes[2, 0], base10_combined, base10_algo.periods, "Step 3: Base-10", "green", a+b)
    
    # Create final result box for base-10
    axes[3, 0].text(0.5, 0.5, f"{a} + {b} = {base10_result}", 
                    fontsize=14, ha='center', va='center', color='green')
    axes[3, 0].axis('off')
    
    # Draw helices for standard base-20
    std_base20_helix_a = standard_base20_algo.create_helix(a)
    std_base20_helix_b = standard_base20_algo.create_helix(b)
    std_base20_combined = standard_base20_algo.combine_helices_using_clock(std_base20_helix_a, std_base20_helix_b)
    std_base20_result = standard_base20_algo.decode_helix(std_base20_combined)
    
    draw_helix_step(axes[0, 1], std_base20_helix_a, standard_base20_algo.periods, "Step 1: Base-20", "red", a)
    draw_helix_step(axes[1, 1], std_base20_helix_b, standard_base20_algo.periods, "Step 2: Base-20", "blue", b)
    draw_helix_step(axes[2, 1], std_base20_combined, standard_base20_algo.periods, "Step 3: Base-20", "green", a+b)
    
    # Create final result box for standard base-20
    axes[3, 1].text(0.5, 0.5, f"{a} + {b} = {std_base20_result}", 
                    fontsize=14, ha='center', va='center', color='green')
    axes[3, 1].axis('off')
    
    # Draw helices for improved base-20
    imp_base20_helix_a = improved_base20_algo.create_helix(a)
    imp_base20_helix_b = improved_base20_algo.create_helix(b)
    imp_base20_combined = improved_base20_algo.combine_helices_using_clock(imp_base20_helix_a, imp_base20_helix_b)
    imp_base20_result = improved_base20_algo.decode_helix(imp_base20_combined)
    
    draw_helix_step(axes[0, 2], imp_base20_helix_a, improved_base20_algo.periods, "Step 1: Improved Base-20", "red", a)
    draw_helix_step(axes[1, 2], imp_base20_helix_b, improved_base20_algo.periods, "Step 2: Improved Base-20", "blue", b)
    draw_helix_step(axes[2, 2], imp_base20_combined, improved_base20_algo.periods, "Step 3: Improved Base-20", "green", a+b)
    
    # Create final result box for improved base-20
    axes[3, 2].text(0.5, 0.5, f"{a} + {b} = {imp_base20_result}", 
                    fontsize=14, ha='center', va='center', color='green')
    axes[3, 2].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.15)
    
    # Save the figure
    plt.savefig(f"figures/demo/step_by_step_{a}_{b}.png")
    plt.close(fig)
    
    # Create a detailed figure for the main period of each implementation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get the main period for each implementation (largest period)
    base10_main_period = max(base10_algo.periods)
    std_base20_main_period = max(standard_base20_algo.periods)
    imp_base20_main_period = max(improved_base20_algo.periods)
    
    # Get index of the main period
    base10_main_idx = base10_algo.periods.index(base10_main_period)
    std_base20_main_idx = standard_base20_algo.periods.index(std_base20_main_period)
    imp_base20_main_idx = improved_base20_algo.periods.index(imp_base20_main_period)
    
    # Draw detailed view of main period for each implementation
    draw_detailed_period(axes[0], base10_helix_a, base10_helix_b, base10_combined,
                         base10_main_idx, base10_main_period, a, b, base10_result, "Base-10")
    
    draw_detailed_period(axes[1], std_base20_helix_a, std_base20_helix_b, std_base20_combined,
                         std_base20_main_idx, std_base20_main_period, a, b, std_base20_result, "Base-20 (Standard)")
    
    draw_detailed_period(axes[2], imp_base20_helix_a, imp_base20_helix_b, imp_base20_combined,
                         imp_base20_main_idx, imp_base20_main_period, a, b, imp_base20_result, "Base-20 (Improved)")
    
    # Title
    fig.suptitle(f"Main Period Comparison for {a} + {b}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig(f"figures/demo/main_period_{a}_{b}.png")
    plt.close(fig)

def draw_helix_step(ax, helix, periods, title, color, value):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    # Plot origin
    ax.plot(0, 0, 'ko')
    
    # Plot each period's contribution
    for i, period in enumerate(periods):
        angle = 2 * np.pi * value / period
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        
        # Draw line from origin to point
        ax.plot([0, cos_val], [0, sin_val], color=color, linestyle='-')
        
        # Plot the point
        ax.plot(cos_val, sin_val, 'o', color=color)
        
        # Add period label
        ax.text(cos_val*1.1, sin_val*1.1, f"p={period}", fontsize=8, color=color)
    
    # Add helix point
    complex_val = sum(helix)
    ax.plot(complex_val.real, complex_val.imag, 'o', color='blue', markersize=8)
    
    # Draw line from origin to helix point
    ax.plot([0, complex_val.real], [0, complex_val.imag], color='blue', linestyle='-')
    
    # Add value label
    ax.text(complex_val.real*1.1, complex_val.imag*1.1, f"value={value}", fontsize=10, color='blue')
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

def draw_detailed_period(ax, helix_a, helix_b, helix_combined, period_idx, period, a, b, result, title):
    """
    Draw a detailed view of a specific period with all three helices
    """
    # Get the cos and sin indices for this period
    cos_idx = 1 + period_idx * 2
    sin_idx = 2 + period_idx * 2
    
    # Extract values
    cos_a = helix_a[cos_idx]
    sin_a = helix_a[sin_idx]
    
    cos_b = helix_b[cos_idx]
    sin_b = helix_b[sin_idx]
    
    cos_combined = helix_combined[cos_idx]
    sin_combined = helix_combined[sin_idx]
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'k--', alpha=0.3)
    
    # Draw points
    ax.scatter(cos_a, sin_a, c='red', s=100, label=f"a = {a}")
    ax.plot([0, cos_a], [0, sin_a], 'r-')
    
    ax.scatter(cos_b, sin_b, c='blue', s=100, label=f"b = {b}")
    ax.plot([0, cos_b], [0, sin_b], 'b-')
    
    ax.scatter(cos_combined, sin_combined, c='green', s=150, label=f"a+b = {result}")
    ax.plot([0, cos_combined], [0, sin_combined], 'g-', linewidth=2)
    
    # Add labels
    ax.text(cos_a*1.1, sin_a*1.1, str(a), fontsize=12, color='red')
    ax.text(cos_b*1.1, sin_b*1.1, str(b), fontsize=12, color='blue')
    ax.text(cos_combined*1.1, sin_combined*1.1, str(result), fontsize=12, color='green')
    
    # Set properties
    ax.set_title(f"{title}: T = {period}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc='upper right')

if __name__ == "__main__":
    main() 