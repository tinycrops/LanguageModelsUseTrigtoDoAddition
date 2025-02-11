from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

class HelixFitter:
    def __init__(
        self,
        periods: List[int] = [2, 5, 10, 100],
        n_pca_components: int = 100,
        regularization_strength: float = 0.01,
        n_cross_val_folds: int = 5
    ):
        """Initialize the helix fitter.
        
        Args:
            periods: List of periods to use for Fourier features
            n_pca_components: Number of PCA components to use
            regularization_strength: L2 regularization strength
            n_cross_val_folds: Number of cross-validation folds
        """
        self.periods = periods
        self.n_pca_components = n_pca_components
        self.regularization_strength = regularization_strength
        self.n_cross_val_folds = n_cross_val_folds
        self.pca = None
        self.C = None
        self.fit_quality = None
        
    def _create_basis(self, a_values: np.ndarray) -> np.ndarray:
        """Create basis functions for helix fitting.
        
        Args:
            a_values: Array of numbers to create basis for
            
        Returns:
            Array of shape (len(a_values), 2*len(periods) + 1)
        """
        B = [a_values]  # Linear component
        for T in self.periods:
            B.append(np.cos(2 * np.pi * a_values / T))
            B.append(np.sin(2 * np.pi * a_values / T))
        return np.array(B).T
        
    def _compute_fit_quality(
        self,
        B: np.ndarray,
        C: np.ndarray,
        activations_pca: np.ndarray
    ) -> dict:
        """Compute various metrics for fit quality.
        
        Args:
            B: Basis functions
            C: Coefficient matrix
            activations_pca: PCA-reduced activations
            
        Returns:
            Dictionary of fit quality metrics
        """
        pred = B @ C.T
        mse = np.mean((activations_pca - pred) ** 2)
        r2 = 1 - mse / np.var(activations_pca)
        
        # Compute cross-validated metrics
        kf = KFold(n_splits=self.n_cross_val_folds)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(B):
            B_train, B_val = B[train_idx], B[val_idx]
            act_train = activations_pca[train_idx]
            act_val = activations_pca[val_idx]
            
            # Fit on train
            C_cv = np.linalg.solve(
                B_train.T @ B_train + 
                self.regularization_strength * np.eye(B_train.shape[1]),
                B_train.T @ act_train
            )
            
            # Evaluate on val
            pred_val = B_val @ C_cv.T
            mse_val = np.mean((act_val - pred_val) ** 2)
            cv_scores.append(mse_val)
            
        return {
            'mse': mse,
            'r2': r2,
            'cv_mse_mean': np.mean(cv_scores),
            'cv_mse_std': np.std(cv_scores)
        }
        
    def fit(
        self,
        activations: np.ndarray,
        a_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """Fit helix to activations with regularization and validation.
        
        Args:
            activations: Array of shape (n_samples, n_features)
            a_values: Optional array of numbers corresponding to activations
            
        Returns:
            Tuple of (coefficient matrix, fit quality metrics)
        """
        if a_values is None:
            a_values = np.arange(activations.shape[0])
            
        # PCA reduction
        self.pca = PCA(n_components=self.n_pca_components)
        activations_pca = self.pca.fit_transform(activations)
        
        # Create basis
        B = self._create_basis(a_values)
        
        # Fit with regularization
        self.C = np.linalg.solve(
            B.T @ B + self.regularization_strength * np.eye(B.shape[1]),
            B.T @ activations_pca
        )
        
        # Compute fit quality metrics
        self.fit_quality = self._compute_fit_quality(B, self.C, activations_pca)
        
        return self.C, self.fit_quality
        
    def transform(
        self,
        a_values: np.ndarray
    ) -> np.ndarray:
        """Transform new a values using fitted helix.
        
        Args:
            a_values: Array of numbers to transform
            
        Returns:
            Array of transformed values
        """
        if self.C is None:
            raise ValueError("Must fit model before transform")
            
        B = self._create_basis(a_values)
        return B @ self.C.T 