import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class ClockAlgorithm:
    def __init__(self, base: int, periods: List[int], precision_digits: int = 6):
        """
        Initialize the Clock Algorithm with specified base and periods
        
        Args:
            base: Number base (e.g., 10 for decimal, 20 for vigesimal)
            periods: List of periods to use for the generalized helix
            precision_digits: Precision for floating point calculations
        """
        self.base = base
        self.periods = periods
        self.precision_digits = precision_digits
        # Determine the modulus (largest period) for answer reconstruction
        self.modulus = max(periods)
        # Number of Fourier features (cos, sin pairs for each period)
        self.num_features = len(periods)
        # Dimension of the helix representation (1 linear + 2 per period)
        self.helix_dim = 1 + 2 * self.num_features
        
    def create_helix(self, number: int) -> np.ndarray:
        """
        Create a generalized helix representation for a number
        
        Args:
            number: The number to represent as a helix
            
        Returns:
            A vector representing the number as a generalized helix
        """
        # Initialize the helix vector with the linear component
        helix = [number]
        
        # Add Fourier features for each period
        for period in self.periods:
            angle = 2 * np.pi * number / period
            # Add cos and sin components for this period
            helix.append(np.cos(angle))
            helix.append(np.sin(angle))
            
        return np.array(helix)
    
    def combine_helices_using_clock(self, helix_a: np.ndarray, helix_b: np.ndarray) -> np.ndarray:
        """
        Combine two helices to perform addition using the Clock algorithm
        
        Args:
            helix_a: Helix representation of the first number
            helix_b: Helix representation of the second number
            
        Returns:
            A combined helix representing the sum
        """
        # Extract the linear components
        linear_a = helix_a[0]
        linear_b = helix_b[0]
        linear_sum = (linear_a + linear_b) % self.modulus
        
        # Initialize the combined helix with the linear sum
        combined_helix = [linear_sum]
        
        # For each period, use trigonometric identities to combine the Fourier features
        for i in range(self.num_features):
            # Extract cos and sin components for the current period
            cos_a_idx = 1 + i * 2
            sin_a_idx = 2 + i * 2
            
            cos_a = helix_a[cos_a_idx]
            sin_a = helix_a[sin_a_idx]
            cos_b = helix_b[cos_a_idx]
            sin_b = helix_b[sin_a_idx]
            
            # Use trigonometric identity: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            combined_cos = cos_a * cos_b - sin_a * sin_b
            # Use trigonometric identity: sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
            combined_sin = sin_a * cos_b + cos_a * sin_b
            
            combined_helix.append(combined_cos)
            combined_helix.append(combined_sin)
            
        return np.array(combined_helix)
    
    def decode_helix(self, helix: np.ndarray) -> int:
        """
        Decode a helix representation back to a number
        
        Args:
            helix: The helix representation to decode
            
        Returns:
            The decoded number
        """
        # The simplest approach is to use the linear component
        # which already contains the modular sum
        linear_result = round(helix[0]) % self.modulus
        
        # For small numbers, the linear component is usually accurate
        if linear_result < 20:
            return linear_result
        
        # For larger numbers, we'll use the Fourier components to check
        # This helps handle cases where numerical precision issues affect the linear component
        
        # Calculate results from each period
        period_results = []
        confidences = []
        
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
            
            # Check confidence based on the magnitude of the vector
            # (how close it is to unit circle - should be close to 1.0)
            magnitude = np.sqrt(cos_val**2 + sin_val**2)
            confidence = 1.0 - min(abs(1.0 - magnitude), 0.5) * 2.0  # Scale 0.5-1.5 range to 0-1
            
            # Only consider this period if the confidence is reasonable
            if confidence > 0.7:
                period_results.append((round(candidate) % self.modulus, period))
                confidences.append(confidence)
        
        # If no period gave a confident result, fall back to linear
        if not period_results:
            return linear_result
        
        # Prefer larger periods as they have better resolution
        # but weight by confidence
        weighted_results = {}
        for (result, period), confidence in zip(period_results, confidences):
            # Weight by both period size and confidence
            weight = period * confidence
            if result in weighted_results:
                weighted_results[result] += weight
            else:
                weighted_results[result] = weight
        
        # Find the result with the highest weighted score
        if weighted_results:
            best_result = max(weighted_results.items(), key=lambda x: x[1])[0]
            return best_result
        
        # If all else fails, return the linear result
        return linear_result
    
    def add(self, a: int, b: int) -> int:
        """
        Add two numbers using the Clock algorithm
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The sum a + b (modulo the largest period)
        """
        # Create helices for the two numbers
        helix_a = self.create_helix(a)
        helix_b = self.create_helix(b)
        
        # Combine helices using the Clock algorithm
        combined_helix = self.combine_helices_using_clock(helix_a, helix_b)
        
        # Decode the combined helix to get the result
        result = self.decode_helix(combined_helix)
        
        return result
    
    def visualize_helix(self, number: int, title: str = None):
        """
        Visualize the helix representation of a number
        
        Args:
            number: The number to visualize
            title: Optional title for the plot
        """
        helix = self.create_helix(number)
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots for each period
        for i, period in enumerate(self.periods):
            # Get the cos and sin components for this period
            cos_idx = 1 + i * 2
            sin_idx = 2 + i * 2
            
            # Create a subplot for each period
            ax = fig.add_subplot(2, len(self.periods), i + 1)
            
            # Create points for the circle
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Plot the unit circle
            ax.plot(x, y, 'k--', alpha=0.3)
            
            # Plot the point corresponding to this number
            angle = 2 * np.pi * number / period
            x_point = np.cos(angle)
            y_point = np.sin(angle)
            
            ax.scatter(x_point, y_point, c='red', s=100)
            
            # Draw a line from the origin to the point
            ax.plot([0, x_point], [0, y_point], 'r-')
            
            # Set title and labels
            ax.set_title(f'Period T = {period}')
            ax.set_xlabel('cos(2π·n/T)')
            ax.set_ylabel('sin(2π·n/T)')
            ax.set_aspect('equal')
            ax.grid(True)
            
            # Set limits
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
        
        # Plot the linear component
        ax = fig.add_subplot(2, len(self.periods), len(self.periods) + 1)
        ax.scatter(number, 0, c='blue', s=100)
        ax.set_title(f'Linear component')
        ax.set_xlabel('n')
        ax.grid(True)
        
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f'Helix representation of {number} (base {self.base})', fontsize=16)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        return fig
    
    def visualize_addition(self, a: int, b: int):
        """
        Visualize the addition of two numbers using the Clock algorithm
        
        Args:
            a: First number
            b: Second number
        """
        # Create helices
        helix_a = self.create_helix(a)
        helix_b = self.create_helix(b)
        combined_helix = self.combine_helices_using_clock(helix_a, helix_b)
        
        # Decode result
        result = self.decode_helix(combined_helix)
        
        # Create a figure with 3 rows and enough columns for the periods
        num_periods = len(self.periods)
        fig = plt.figure(figsize=(15, 15))
        
        # Plot each period for each number
        for i, period in enumerate(self.periods):
            cos_idx = 1 + i * 2
            sin_idx = 2 + i * 2
            
            # Get the vectors for each helix
            cos_a, sin_a = helix_a[cos_idx], helix_a[sin_idx]
            cos_b, sin_b = helix_b[cos_idx], helix_b[sin_idx]
            cos_sum, sin_sum = combined_helix[cos_idx], combined_helix[sin_idx]
            
            # Plot for a
            ax1 = fig.add_subplot(3, num_periods, i + 1)
            
            # Create points for the circle
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Plot the unit circle
            ax1.plot(x, y, 'k--', alpha=0.3)
            
            # Plot the point for a
            ax1.scatter(cos_a, sin_a, c='red', s=100, label=f'a = {a}')
            ax1.plot([0, cos_a], [0, sin_a], 'r-')
            
            ax1.set_title(f'Period T = {period}')
            ax1.set_xlabel('cos(2π·n/T)')
            ax1.set_ylabel('sin(2π·n/T)')
            ax1.set_aspect('equal')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            
            # Plot for b
            ax2 = fig.add_subplot(3, num_periods, i + 1 + num_periods)
            ax2.plot(x, y, 'k--', alpha=0.3)
            ax2.scatter(cos_b, sin_b, c='blue', s=100, label=f'b = {b}')
            ax2.plot([0, cos_b], [0, sin_b], 'b-')
            
            ax2.set_xlabel('cos(2π·n/T)')
            ax2.set_ylabel('sin(2π·n/T)')
            ax2.set_aspect('equal')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            
            # Plot for the sum
            ax3 = fig.add_subplot(3, num_periods, i + 1 + 2*num_periods)
            ax3.plot(x, y, 'k--', alpha=0.3)
            ax3.scatter(cos_sum, sin_sum, c='green', s=100, label=f'a + b = {result}')
            ax3.plot([0, cos_sum], [0, sin_sum], 'g-')
            
            ax3.set_xlabel('cos(2π·n/T)')
            ax3.set_ylabel('sin(2π·n/T)')
            ax3.set_aspect('equal')
            ax3.legend()
            ax3.grid(True)
            ax3.set_xlim(-1.2, 1.2)
            ax3.set_ylim(-1.2, 1.2)
        
        # Add an extra column for the linear components if there's space
        if num_periods > 1:
            # Plot linear component for a
            ax4 = fig.add_subplot(3, num_periods, num_periods)
            ax4.scatter(a, 0, c='red', s=100, label=f'a = {a}')
            ax4.set_title('Linear component')
            ax4.set_xlabel('n')
            ax4.grid(True)
            ax4.legend()
            
            # Plot linear component for b
            ax5 = fig.add_subplot(3, num_periods, 2*num_periods)
            ax5.scatter(b, 0, c='blue', s=100, label=f'b = {b}')
            ax5.set_xlabel('n')
            ax5.grid(True)
            ax5.legend()
            
            # Plot linear component for the sum
            ax6 = fig.add_subplot(3, num_periods, 3*num_periods)
            ax6.scatter(result, 0, c='green', s=100, label=f'a + b = {result}')
            ax6.set_xlabel('n')
            ax6.grid(True)
            ax6.legend()
        
        fig.suptitle(f'Adding {a} + {b} = {result} using the Clock algorithm (base {self.base})', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        return fig
    
    def evaluate_accuracy(self, max_number: int) -> Dict:
        """
        Evaluate the accuracy of the Clock algorithm up to a maximum number
        
        Args:
            max_number: The maximum number to test
            
        Returns:
            Dictionary with accuracy statistics
        """
        total_tests = 0
        correct = 0
        errors = []
        
        for a in range(max_number + 1):
            for b in range(max_number + 1):
                expected = (a + b) % self.modulus
                actual = self.add(a, b)
                
                total_tests += 1
                if expected == actual:
                    correct += 1
                else:
                    errors.append((a, b, expected, actual))
        
        accuracy = correct / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "correct": correct,
            "accuracy": accuracy,
            "errors": errors
        } 