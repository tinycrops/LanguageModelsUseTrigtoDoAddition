import numpy as np
import matplotlib.pyplot as plt
from ClockAlgorithm import ClockAlgorithm
import os
import random

# Create output directory for figures
os.makedirs("figures/improved_base20", exist_ok=True)

class ImprovedBase20Algorithm(ClockAlgorithm):
    """
    An improved version of the Clock Algorithm optimized for base-20
    with better noise resistance.
    """
    
    def __init__(self, precision_digits=6):
        """
        Initialize with optimized periods for base-20
        """
        # Use a richer set of periods for better redundancy
        # Include more factors of 20 and powers of 2 for better representation
        optimized_periods = [2, 4, 5, 10, 20, 40, 100, 400]
        super().__init__(base=20, periods=optimized_periods, precision_digits=precision_digits)
    
    def decode_helix(self, helix: np.ndarray) -> int:
        """
        Improved decoding for better noise resistance
        """
        # First use the linear component as a baseline
        linear_result = round(helix[0]) % self.modulus
        
        # For small numbers, the linear component is usually accurate
        if linear_result < 20:
            return linear_result
            
        # Calculate results from each period
        period_results = []
        confidences = []
        period_weights = {}
        
        # Pre-compute period weights based on their importance
        # Higher periods get higher weights, with special emphasis on base factors
        for period in self.periods:
            if period == 20 or period == 400:  # Base and modulus
                period_weights[period] = 2.0
            elif period == 4 or period == 5:   # Factors of base
                period_weights[period] = 1.5
            else:
                period_weights[period] = 1.0
        
        for i, period in enumerate(self.periods):
            cos_idx = 1 + i * 2
            sin_idx = 2 + i * 2
            
            cos_val = helix[cos_idx]
            sin_val = helix[sin_idx]
            
            # Extract the angle from the cos and sin values
            angle = np.arctan2(sin_val, cos_val)
            if angle < 0:
                angle += 2 * np.pi
                
            # Convert angle back to number value
            candidate = (angle * period / (2 * np.pi)) % period
            
            # Calculate confidence based on vector magnitude and normalization
            magnitude = np.sqrt(cos_val**2 + sin_val**2)
            # Higher confidence if magnitude is close to 1
            confidence = 1.0 - min(abs(1.0 - magnitude), 0.5) * 2.0
            
            # Apply period weight to confidence
            weighted_confidence = confidence * period_weights[period]
            
            # Only consider if confidence is reasonable
            if confidence > 0.5:  # Lower threshold for more contributions
                period_results.append((round(candidate) % self.modulus, period))
                confidences.append(weighted_confidence)
        
        # If no period gave a confident result, fall back to linear
        if not period_results:
            return linear_result
            
        # Voting mechanism - each period gets a weighted vote
        votes = {}
        for (result, period), confidence in zip(period_results, confidences):
            weight = period * confidence  # Larger periods and higher confidence get more weight
            if result in votes:
                votes[result] += weight
            else:
                votes[result] = weight
                
        # Select the result with the most votes
        if votes:
            # For ties, prefer result closer to linear estimate
            if len(votes) > 1:
                # Sort by vote count (descending) and then by proximity to linear_result (ascending)
                sorted_votes = sorted(votes.items(), key=lambda x: (-x[1], abs(x[0] - linear_result)))
                return sorted_votes[0][0]
            else:
                return max(votes.items(), key=lambda x: x[1])[0]
                
        return linear_result

# Initialize standard and improved algorithms
standard_base20 = ClockAlgorithm(base=20, periods=[2, 4, 5, 20, 400])
improved_base20 = ImprovedBase20Algorithm()

def compare_noise_resistance():
    """
    Compare noise resistance between standard and improved base-20 implementations
    """
    print("\n--- NOISE RESISTANCE COMPARISON ---")
    
    # Test parameters
    max_num = 99
    num_tests = 100
    noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    standard_results = []
    improved_results = []
    
    for noise in noise_levels:
        print(f"\nNoise level: {noise}")
        
        # Generate test cases - use the same test pairs for both algorithms
        test_pairs = []
        for _ in range(num_tests):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            test_pairs.append((a, b))
            
        # Test standard base-20
        standard_correct = 0
        for a, b in test_pairs:
            expected = (a + b) % standard_base20.modulus
            
            # Create helices with noise
            helix_a = standard_base20.create_helix(a)
            helix_b = standard_base20.create_helix(b)
            
            # Use the same random seed for both tests
            np.random.seed(a * 1000 + b)
            noise_a = np.random.normal(0, noise, helix_a.shape)
            noise_b = np.random.normal(0, noise, helix_b.shape)
            
            noisy_helix_a = helix_a + noise_a
            noisy_helix_b = helix_b + noise_b
            
            combined_helix = standard_base20.combine_helices_using_clock(noisy_helix_a, noisy_helix_b)
            standard_result = standard_base20.decode_helix(combined_helix)
            
            if standard_result == expected:
                standard_correct += 1
                
        standard_accuracy = standard_correct / num_tests
        standard_results.append((noise, standard_accuracy))
        print(f"  Standard Base-20: {standard_accuracy * 100:.2f}% ({standard_correct}/{num_tests})")
        
        # Test improved base-20
        improved_correct = 0
        for a, b in test_pairs:
            expected = (a + b) % improved_base20.modulus
            
            # Create helices with noise
            helix_a = improved_base20.create_helix(a)
            helix_b = improved_base20.create_helix(b)
            
            # Use the same random seed for both tests
            np.random.seed(a * 1000 + b)
            noise_a = np.random.normal(0, noise, helix_a.shape)
            noise_b = np.random.normal(0, noise, helix_b.shape)
            
            noisy_helix_a = helix_a + noise_a
            noisy_helix_b = helix_b + noise_b
            
            combined_helix = improved_base20.combine_helices_using_clock(noisy_helix_a, noisy_helix_b)
            improved_result = improved_base20.decode_helix(combined_helix)
            
            if improved_result == expected:
                improved_correct += 1
                
        improved_accuracy = improved_correct / num_tests
        improved_results.append((noise, improved_accuracy))
        print(f"  Improved Base-20: {improved_accuracy * 100:.2f}% ({improved_correct}/{num_tests})")
    
    # Create a line chart comparing noise resistance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    noise_levels_std, std_accs = zip(*standard_results)
    noise_levels_imp, imp_accs = zip(*improved_results)
    
    ax.plot(noise_levels_std, [acc * 100 for acc in std_accs], 'o-', color='green', label='Standard Base-20')
    ax.plot(noise_levels_imp, [acc * 100 for acc in imp_accs], 'o-', color='purple', label='Improved Base-20')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Noise Level (σ)')
    ax.set_title('Noise Resistance Comparison for Base-20 Implementations')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figures/improved_base20/noise_resistance_comparison.png")
    plt.close(fig)

def visualize_period_representation():
    """
    Visualize how the different periods represent numbers in the improved algorithm
    """
    test_number = 42  # A number that demonstrates the periods well
    
    # Create helices for standard and improved algorithms
    standard_helix = standard_base20.create_helix(test_number)
    improved_helix = improved_base20.create_helix(test_number)
    
    # Create separate figures for each implementation to avoid layout issues
    # Standard base-20 figure
    std_fig, std_axes = plt.subplots(1, len(standard_base20.periods), figsize=(15, 5))
    
    for i, period in enumerate(standard_base20.periods):
        cos_idx = 1 + i * 2
        sin_idx = 2 + i * 2
        
        cos_val = standard_helix[cos_idx]
        sin_val = standard_helix[sin_idx]
        
        ax = std_axes[i]
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', alpha=0.3)
        
        # Draw point
        ax.scatter(cos_val, sin_val, c='green', s=100)
        ax.plot([0, cos_val], [0, sin_val], 'g-')
        
        ax.set_title(f'T = {period}')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    
    std_fig.suptitle(f'Standard Base-20 Representation of {test_number}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"figures/improved_base20/standard_periods_{test_number}.png")
    plt.close(std_fig)
    
    # Improved base-20 figure
    imp_fig, imp_axes = plt.subplots(1, len(improved_base20.periods), figsize=(20, 5))
    
    for i, period in enumerate(improved_base20.periods):
        cos_idx = 1 + i * 2
        sin_idx = 2 + i * 2
        
        cos_val = improved_helix[cos_idx]
        sin_val = improved_helix[sin_idx]
        
        ax = imp_axes[i]
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', alpha=0.3)
        
        # Draw point
        ax.scatter(cos_val, sin_val, c='purple', s=100)
        ax.plot([0, cos_val], [0, sin_val], 'm-')
        
        ax.set_title(f'T = {period}')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    
    imp_fig.suptitle(f'Improved Base-20 Representation of {test_number}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"figures/improved_base20/improved_periods_{test_number}.png")
    plt.close(imp_fig)
    
    # Create a comparison of addition result
    test_a = 17
    test_b = 25
    expected = (test_a + test_b) % standard_base20.modulus
    
    # Standard implementation
    std_helix_a = standard_base20.create_helix(test_a)
    std_helix_b = standard_base20.create_helix(test_b)
    std_combined = standard_base20.combine_helices_using_clock(std_helix_a, std_helix_b)
    std_result = standard_base20.decode_helix(std_combined)
    
    # Improved implementation
    imp_helix_a = improved_base20.create_helix(test_a)
    imp_helix_b = improved_base20.create_helix(test_b)
    imp_combined = improved_base20.combine_helices_using_clock(imp_helix_a, imp_helix_b)
    imp_result = improved_base20.decode_helix(imp_combined)
    
    # Create comparison figure
    comp_fig, comp_ax = plt.subplots(figsize=(10, 6))
    
    # Add text showing results
    comp_ax.text(0.5, 0.6, f"Addition: {test_a} + {test_b} = {expected}", 
                fontsize=16, ha='center')
    comp_ax.text(0.5, 0.5, f"Standard Base-20 Result: {std_result}", 
                fontsize=14, ha='center', color='green')
    comp_ax.text(0.5, 0.4, f"Improved Base-20 Result: {imp_result}", 
                fontsize=14, ha='center', color='purple')
    
    # Show the extra periods in the improved version
    extra_periods = [p for p in improved_base20.periods if p not in standard_base20.periods]
    if extra_periods:
        comp_ax.text(0.5, 0.3, f"Additional periods in improved version: {extra_periods}", 
                    fontsize=12, ha='center', color='blue')
    
    # Remove axes
    comp_ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"figures/improved_base20/addition_comparison_{test_a}_{test_b}.png")
    plt.close(comp_fig)

def test_addition_accuracy():
    """
    Test the accuracy of the improved base-20 implementation on addition
    """
    print("\n--- ADDITION ACCURACY TEST ---")
    
    max_num = 99
    
    # Test standard base-20
    standard_correct = 0
    for a in range(max_num + 1):
        for b in range(max_num + 1):
            expected = (a + b) % standard_base20.modulus
            result = standard_base20.add(a, b)
            if result == expected:
                standard_correct += 1
    
    standard_total = (max_num + 1) ** 2
    standard_accuracy = standard_correct / standard_total
    
    # Test improved base-20
    improved_correct = 0
    for a in range(max_num + 1):
        for b in range(max_num + 1):
            expected = (a + b) % improved_base20.modulus
            result = improved_base20.add(a, b)
            if result == expected:
                improved_correct += 1
    
    improved_total = (max_num + 1) ** 2
    improved_accuracy = improved_correct / improved_total
    
    print(f"Standard Base-20: {standard_accuracy * 100:.2f}% ({standard_correct}/{standard_total})")
    print(f"Improved Base-20: {improved_accuracy * 100:.2f}% ({improved_correct}/{improved_total})")
    
    # Create a bar chart comparing accuracies
    fig, ax = plt.subplots(figsize=(8, 6))
    
    implementations = ['Standard Base-20', 'Improved Base-20']
    accuracies = [standard_accuracy * 100, improved_accuracy * 100]
    colors = ['green', 'purple']
    
    ax.bar(implementations, accuracies, color=colors, alpha=0.7)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Addition Accuracy Comparison')
    ax.set_ylim(0, 101)
    
    # Add accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("figures/improved_base20/accuracy_comparison.png")
    plt.close(fig)

def main():
    print("=== IMPROVED BASE-20 EVALUATION ===")
    
    # Compare noise resistance
    compare_noise_resistance()
    
    # Visualize period representation
    visualize_period_representation()
    
    # Test addition accuracy
    test_addition_accuracy()
    
    print("\nEvaluation completed. Check the 'figures/improved_base20' directory for visualizations.")

if __name__ == "__main__":
    main() 