import pytest
import numpy as np
from transformers import AutoTokenizer
from src.data.dataset import AdditionDataset, AdditionExample
from src.analysis.helix_fitting import HelixFitter

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

@pytest.fixture
def dataset(tokenizer):
    return AdditionDataset(tokenizer)

@pytest.fixture
def helix_fitter():
    return HelixFitter()

def test_addition_example_generation(dataset):
    example = dataset.generate_example()
    assert example is not None
    assert isinstance(example, AdditionExample)
    assert dataset.validate_example(example)
    
def test_dataset_generation(dataset):
    train, val = dataset.generate_dataset(100)
    assert len(train) + len(val) == 100
    assert len(val) == int(100 * dataset.validation_split)
    
def test_example_validation(dataset):
    # Valid example
    example = AdditionExample(
        a=5,
        b=3,
        prompt="Output ONLY a number. 5+3=",
        answer="8",
        tokenized_length=len(dataset.tokenizer.encode("Output ONLY a number. 5+3="))
    )
    assert dataset.validate_example(example)
    
    # Invalid range
    invalid_example = AdditionExample(
        a=100,  # Outside valid range
        b=3,
        prompt="Output ONLY a number. 100+3=",
        answer="103",
        tokenized_length=len(dataset.tokenizer.encode("Output ONLY a number. 100+3="))
    )
    assert not dataset.validate_example(invalid_example)

def test_helix_basis_creation(helix_fitter):
    a_values = np.array([0, 1, 2, 3, 4])
    basis = helix_fitter._create_basis(a_values)
    expected_cols = 2 * len(helix_fitter.periods) + 1  # 2 for each period (sin/cos) + 1 for linear
    assert basis.shape == (len(a_values), expected_cols)

def test_helix_fitting(helix_fitter):
    # Create synthetic data that should be well-modeled by a helix
    a_values = np.linspace(0, 10, 100)
    T = 5  # Period
    activations = np.column_stack([
        a_values,  # Linear component
        np.cos(2 * np.pi * a_values / T),  # Cosine component
        np.sin(2 * np.pi * a_values / T)   # Sine component
    ])
    
    C, fit_quality = helix_fitter.fit(activations, a_values)
    
    assert C is not None
    assert isinstance(fit_quality, dict)
    assert 'mse' in fit_quality
    assert 'r2' in fit_quality
    assert fit_quality['r2'] > 0.9  # Should fit well
    
def test_helix_transform(helix_fitter):
    # First fit the model
    a_values = np.linspace(0, 10, 100)
    activations = np.column_stack([
        a_values,
        np.cos(2 * np.pi * a_values / 5),
        np.sin(2 * np.pi * a_values / 5)
    ])
    helix_fitter.fit(activations, a_values)
    
    # Test transform
    new_values = np.array([1.5, 2.5, 3.5])
    transformed = helix_fitter.transform(new_values)
    
    assert transformed.shape[0] == len(new_values)
    assert transformed.shape[1] == helix_fitter.n_pca_components 