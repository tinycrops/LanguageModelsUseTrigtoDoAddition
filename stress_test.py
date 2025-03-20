import numpy as np
import matplotlib.pyplot as plt
from ClockAlgorithm import ClockAlgorithm
import os
import random

# Create output directory for figures
os.makedirs("figures/stress_test", exist_ok=True)

# Define test parameters
MAX_SMALL = 99  # Maximum for small number tests
MAX_LARGE = 999  # Maximum for large number tests
NUM_SAMPLES = 100  # Number of random samples for large tests

# Initialize algorithms
base10_periods = [2, 5, 10, 100]
base10_algo = ClockAlgorithm(base=10, periods=base10_periods)

base20_periods = [2, 4, 5, 20, 400]
base20_algo = ClockAlgorithm(base=20, periods=base20_periods)

def test_range_accuracy(algo, max_num, num_tests=None):
    """
    Test accuracy for a range of numbers
    
    Args:
        algo: ClockAlgorithm instance
        max_num: Maximum number to test
        num_tests: Number of random tests (if None, test all combinations)
        
    Returns:
        Dictionary with accuracy statistics
    """
    correct = 0
    errors = []
    test_pairs = []
    
    if num_tests is None:
        # Test all combinations up to max_num
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                test_pairs.append((a, b))
    else:
        # Generate random test pairs
        for _ in range(num_tests):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            test_pairs.append((a, b))
    
    for a, b in test_pairs:
        expected = (a + b) % algo.modulus
        actual = algo.add(a, b)
        
        if expected == actual:
            correct += 1
        else:
            errors.append((a, b, expected, actual))
    
    total_tests = len(test_pairs)
    accuracy = correct / total_tests if total_tests > 0 else 0
    
    return {
        "total_tests": total_tests,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors
    }

def test_with_noise(algo, max_num, noise_level=0.01, num_tests=100):
    """
    Test accuracy when adding noise to the helix representations
    
    Args:
        algo: ClockAlgorithm instance
        max_num: Maximum number to test
        noise_level: Standard deviation of Gaussian noise
        num_tests: Number of random tests
        
    Returns:
        Dictionary with accuracy statistics
    """
    correct = 0
    errors = []
    test_pairs = []
    
    # Generate random test pairs
    for _ in range(num_tests):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        test_pairs.append((a, b))
    
    for a, b in test_pairs:
        expected = (a + b) % algo.modulus
        
        # Create helices with noise
        helix_a = algo.create_helix(a)
        helix_b = algo.create_helix(b)
        
        # Add noise to the helix representations
        noise_a = np.random.normal(0, noise_level, helix_a.shape)
        noise_b = np.random.normal(0, noise_level, helix_b.shape)
        
        noisy_helix_a = helix_a + noise_a
        noisy_helix_b = helix_b + noise_b
        
        # Combine helices and decode
        combined_helix = algo.combine_helices_using_clock(noisy_helix_a, noisy_helix_b)
        actual = algo.decode_helix(combined_helix)
        
        if expected == actual:
            correct += 1
        else:
            errors.append((a, b, expected, actual))
    
    total_tests = len(test_pairs)
    accuracy = correct / total_tests if total_tests > 0 else 0
    
    return {
        "total_tests": total_tests,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors
    }

def test_digit_ranges(algo, digits=2):
    """
    Test accuracy for different ranges of digits
    
    Args:
        algo: ClockAlgorithm instance
        digits: Maximum number of digits to test
        
    Returns:
        List of (range, accuracy) pairs
    """
    results = []
    
    for d in range(1, digits + 1):
        max_num = 10**d - 1
        num_tests = min(1000, 10**(2*d-2))  # Scale tests with range
        
        # Run accuracy test for this digit range
        test_results = test_range_accuracy(algo, max_num, num_tests)
        results.append((d, test_results["accuracy"]))
        
        print(f"{d} digit(s): Accuracy = {test_results['accuracy'] * 100:.2f}% ({test_results['correct']}/{test_results['total_tests']})")
        if test_results["errors"]:
            print(f"  First 3 errors: {test_results['errors'][:3]}")
    
    return results

def compare_digit_accuracy():
    """
    Compare accuracy of base-10 and base-20 implementations for different digit ranges
    """
    print("\n--- DIGIT RANGE COMPARISON ---")
    
    print("Base-10 accuracy by digit range:")
    base10_results = test_digit_ranges(base10_algo, digits=3)
    
    print("\nBase-20 accuracy by digit range:")
    base20_results = test_digit_ranges(base20_algo, digits=3)
    
    # Create a bar chart to compare accuracies by digit range
    fig, ax = plt.subplots(figsize=(10, 6))
    
    base10_digits, base10_accs = zip(*base10_results)
    base20_digits, base20_accs = zip(*base20_results)
    
    x = np.arange(len(base10_digits))
    width = 0.35
    
    ax.bar(x - width/2, [acc * 100 for acc in base10_accs], width, label='Base-10', color='blue', alpha=0.7)
    ax.bar(x + width/2, [acc * 100 for acc in base20_accs], width, label='Base-20', color='green', alpha=0.7)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Number of Digits')
    ax.set_title('Accuracy by Number of Digits')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} digits" for d in base10_digits])
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add accuracy values on top of the bars
    for i, v in enumerate(base10_accs):
        ax.text(i - width/2, v * 100 + 1, f"{v*100:.1f}%", ha='center')
    
    for i, v in enumerate(base20_accs):
        ax.text(i + width/2, v * 100 + 1, f"{v*100:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("figures/stress_test/digit_accuracy_comparison.png")
    plt.close(fig)

def compare_noise_resistance():
    """
    Compare noise resistance of base-10 and base-20 implementations
    """
    print("\n--- NOISE RESISTANCE COMPARISON ---")
    
    # Test with different noise levels
    noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
    base10_results = []
    base20_results = []
    
    for noise in noise_levels:
        print(f"\nNoise level: {noise}")
        
        # Base-10
        print("Base-10:")
        base10_test = test_with_noise(base10_algo, MAX_SMALL, noise_level=noise)
        base10_results.append((noise, base10_test["accuracy"]))
        print(f"  Accuracy: {base10_test['accuracy'] * 100:.2f}% ({base10_test['correct']}/{base10_test['total_tests']})")
        
        # Base-20
        print("Base-20:")
        base20_test = test_with_noise(base20_algo, MAX_SMALL, noise_level=noise)
        base20_results.append((noise, base20_test["accuracy"]))
        print(f"  Accuracy: {base20_test['accuracy'] * 100:.2f}% ({base20_test['correct']}/{base20_test['total_tests']})")
    
    # Create a line chart to compare noise resistance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    noise_levels_10, base10_accs = zip(*base10_results)
    noise_levels_20, base20_accs = zip(*base20_results)
    
    ax.plot(noise_levels_10, [acc * 100 for acc in base10_accs], 'o-', color='blue', label='Base-10')
    ax.plot(noise_levels_20, [acc * 100 for acc in base20_accs], 'o-', color='green', label='Base-20')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Noise Level (σ)')
    ax.set_title('Noise Resistance Comparison')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figures/stress_test/noise_resistance_comparison.png")
    plt.close(fig)

def analyze_error_patterns():
    """
    Analyze error patterns for larger numbers
    """
    print("\n--- ERROR PATTERN ANALYSIS ---")
    
    # Test with numbers larger than usually tested
    large_test_num = 50  # Reduced from MAX_LARGE for faster testing
    num_samples = 1000
    
    # Generate random test pairs
    test_pairs = []
    for _ in range(num_samples):
        a = random.randint(0, large_test_num)
        b = random.randint(0, large_test_num)
        test_pairs.append((a, b))
    
    # Test base-10
    base10_errors = []
    for a, b in test_pairs:
        expected = (a + b) % base10_algo.modulus
        actual = base10_algo.add(a, b)
        
        if expected != actual:
            base10_errors.append((a, b, expected, actual, abs(expected - actual)))
    
    # Test base-20
    base20_errors = []
    for a, b in test_pairs:
        expected = (a + b) % base20_algo.modulus
        actual = base20_algo.add(a, b)
        
        if expected != actual:
            base20_errors.append((a, b, expected, actual, abs(expected - actual)))
    
    print(f"Base-10 errors: {len(base10_errors)}/{num_samples} ({len(base10_errors)/num_samples*100:.2f}%)")
    print(f"Base-20 errors: {len(base20_errors)}/{num_samples} ({len(base20_errors)/num_samples*100:.2f}%)")
    
    # Analyze error magnitudes
    if base10_errors:
        base10_error_magnitudes = [e[4] for e in base10_errors]
        print(f"Base-10 error magnitude: avg={np.mean(base10_error_magnitudes):.2f}, max={np.max(base10_error_magnitudes)}")
    
    if base20_errors:
        base20_error_magnitudes = [e[4] for e in base20_errors]
        print(f"Base-20 error magnitude: avg={np.mean(base20_error_magnitudes):.2f}, max={np.max(base20_error_magnitudes)}")
    
    # Create visualization if there are errors
    if base10_errors or base20_errors:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot error distribution for base-10
        if base10_errors:
            a_vals = [e[0] for e in base10_errors]
            b_vals = [e[1] for e in base10_errors]
            error_sizes = [e[4] for e in base10_errors]
            
            scatter1 = ax1.scatter(a_vals, b_vals, c=error_sizes, cmap='Reds', alpha=0.7, s=50)
            ax1.set_title('Base-10 Error Distribution')
            ax1.set_xlabel('a')
            ax1.set_ylabel('b')
            fig.colorbar(scatter1, ax=ax1, label='Error Magnitude')
        else:
            ax1.text(0.5, 0.5, 'No errors', ha='center', va='center', fontsize=14)
            ax1.set_title('Base-10 Error Distribution (No Errors)')
        
        # Plot error distribution for base-20
        if base20_errors:
            a_vals = [e[0] for e in base20_errors]
            b_vals = [e[1] for e in base20_errors]
            error_sizes = [e[4] for e in base20_errors]
            
            scatter2 = ax2.scatter(a_vals, b_vals, c=error_sizes, cmap='Reds', alpha=0.7, s=50)
            ax2.set_title('Base-20 Error Distribution')
            ax2.set_xlabel('a')
            ax2.set_ylabel('b')
            fig.colorbar(scatter2, ax=ax2, label='Error Magnitude')
        else:
            ax2.text(0.5, 0.5, 'No errors', ha='center', va='center', fontsize=14)
            ax2.set_title('Base-20 Error Distribution (No Errors)')
        
        plt.tight_layout()
        plt.savefig("figures/stress_test/error_distribution.png")
        plt.close(fig)

def main():
    print("=== CLOCK ALGORITHM STRESS TEST ===")
    
    # Test 1: Compare accuracy by digit range
    compare_digit_accuracy()
    
    # Test 2: Compare noise resistance
    compare_noise_resistance()
    
    # Test 3: Analyze error patterns
    analyze_error_patterns()
    
    print("\nStress tests completed. Check the 'figures/stress_test' directory for visualizations.")

if __name__ == "__main__":
    main() 