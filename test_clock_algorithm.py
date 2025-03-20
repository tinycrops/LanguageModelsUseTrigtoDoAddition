import numpy as np
import matplotlib.pyplot as plt
from ClockAlgorithm import ClockAlgorithm
import os

# Create output directory for figures
os.makedirs("figures", exist_ok=True)

# Test parameters
MAX_TEST_NUMBER = 19  # As in the paper

# Initialize both base-10 and base-20 Clock Algorithms
# Base-10 configuration (using periods from the paper)
base10_periods = [2, 5, 10, 100]
base10_algo = ClockAlgorithm(base=10, periods=base10_periods)

# Base-20 configuration (adapted periods)
base20_periods = [2, 4, 5, 20, 400]
base20_algo = ClockAlgorithm(base=20, periods=base20_periods)

def run_tests():
    # Test and visualize basic number representations
    for base, algo, name in [(10, base10_algo, "base10"), (20, base20_algo, "base20")]:
        print(f"\n--- Testing {name} Clock Algorithm ---")
        
        # Visualize representation of some example numbers
        test_numbers = [3, 7, 12, 19]
        for num in test_numbers:
            fig = algo.visualize_helix(num)
            plt.savefig(f"figures/{name}_helix_{num}.png")
            plt.close(fig)
            print(f"Created visualization for {num} in {name}")
        
        # Test some example additions
        test_pairs = [(3, 7), (7, 12), (12, 7), (5, 19)]
        for a, b in test_pairs:
            # Calculate the expected result
            expected = (a + b) % algo.modulus
            
            # Calculate using the Clock algorithm
            result = algo.add(a, b)
            
            print(f"{a} + {b} = {result} (expected {expected})")
            
            # Visualize the addition
            fig = algo.visualize_addition(a, b)
            plt.savefig(f"figures/{name}_addition_{a}_{b}.png")
            plt.close(fig)
        
        # Evaluate accuracy on all numbers up to MAX_TEST_NUMBER
        accuracy_results = algo.evaluate_accuracy(MAX_TEST_NUMBER)
        print(f"Accuracy: {accuracy_results['accuracy'] * 100:.2f}% ({accuracy_results['correct']}/{accuracy_results['total_tests']})")
        
        if accuracy_results['errors']:
            print(f"First 5 errors: {accuracy_results['errors'][:5]}")

def verify_algorithm_implementation():
    """
    Verify the Clock algorithm implementation using the example from the paper
    """
    print("\n--- Verifying Clock Algorithm Implementation ---")
    
    # Use the example from the paper: a=7, b=12
    a, b = 7, 12
    
    # Base-10
    base10_result = base10_algo.add(a, b)
    base10_expected = (a + b) % 100
    print(f"Base-10: {a} + {b} = {base10_result} (expected {base10_expected})")
    
    # Base-20
    base20_result = base20_algo.add(a, b)
    base20_expected = (a + b) % 400
    print(f"Base-20: {a} + {b} = {base20_result} (expected {base20_expected})")
    
    # Visualize the algorithm steps for base-10
    fig = base10_algo.visualize_addition(a, b)
    plt.savefig("figures/verification_base10.png")
    plt.close(fig)
    
    # Visualize the algorithm steps for base-20
    fig = base20_algo.visualize_addition(a, b)
    plt.savefig("figures/verification_base20.png")
    plt.close(fig)

def main():
    # Verify the implementation matches the paper example
    verify_algorithm_implementation()
    
    # Run comprehensive tests
    run_tests()
    
    print("\nAll tests completed. Check the 'figures' directory for visualizations.")

if __name__ == "__main__":
    main() 