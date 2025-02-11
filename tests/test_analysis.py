import pytest
import numpy as np
from nnsight import LanguageModel
from transformers import AutoTokenizer
from src.analysis.attention_analysis import AttentionAnalyzer
from src.analysis.mlp_analysis import MLPAnalyzer
from src.analysis.helix_fitting import HelixFitter

@pytest.fixture
def model():
    return LanguageModel("EleutherAI/gpt-j-6B")

@pytest.fixture
def attention_analyzer(model):
    return AttentionAnalyzer(model)

@pytest.fixture
def mlp_analyzer(model):
    return MLPAnalyzer(model)

@pytest.fixture
def helix_fitter():
    return HelixFitter()

def test_head_effects(attention_analyzer):
    prompt = "Output ONLY a number. 5+3="
    corrupted = "Output ONLY a number. 6+3="
    
    te, de = attention_analyzer.get_head_effects(prompt, corrupted, layer=10, head=0)
    
    assert isinstance(te, float)
    assert isinstance(de, float)
    assert abs(te) >= abs(de)  # Total effect should be larger than direct effect
    
def test_head_analysis(attention_analyzer):
    prompt = "Output ONLY a number. 5+3="
    corrupted = "Output ONLY a number. 6+3="
    
    heads = attention_analyzer.analyze_heads(prompt, corrupted)
    
    assert isinstance(heads, dict)
    assert 'a_b_heads' in heads
    assert 'a_plus_b_heads' in heads
    assert 'mixed_heads' in heads
    
    # Check that heads are properly categorized
    total_heads = len(heads['a_b_heads']) + len(heads['a_plus_b_heads']) + len(heads['mixed_heads'])
    assert total_heads == 20  # Default top_k value
    
def test_head_validation(attention_analyzer, helix_fitter):
    # First fit helix
    a_values = np.linspace(0, 10, 100)
    activations = np.column_stack([
        a_values,
        np.cos(2 * np.pi * a_values / 5),
        np.sin(2 * np.pi * a_values / 5)
    ])
    helix_fitter.fit(activations, a_values)
    
    # Test validation
    prompt = "Output ONLY a number. 5+3="
    corrupted = "Output ONLY a number. 6+3="
    
    heads = attention_analyzer.analyze_heads(prompt, corrupted)
    metrics = attention_analyzer.validate_head_categories(heads, helix_fitter, prompt, 5, 3)
    
    assert isinstance(metrics, dict)
    assert 'ab_head_accuracy' in metrics
    assert 'sum_head_accuracy' in metrics
    assert 0 <= metrics['ab_head_accuracy'] <= 1
    assert 0 <= metrics['sum_head_accuracy'] <= 1
    
def test_mlp_effects(mlp_analyzer):
    prompt = "Output ONLY a number. 5+3="
    corrupted = "Output ONLY a number. 6+3="
    
    te, de = mlp_analyzer.get_mlp_effects(prompt, corrupted, layer=15)
    
    assert isinstance(te, float)
    assert isinstance(de, float)
    assert abs(te) >= abs(de)
    
def test_neuron_analysis(mlp_analyzer):
    prompt = "Output ONLY a number. 5+3="
    
    neurons = mlp_analyzer.analyze_neurons(prompt, layer=15, top_k=10)
    
    assert isinstance(neurons, list)
    assert len(neurons) == 10
    for neuron in neurons:
        assert isinstance(neuron, dict)
        assert 'index' in neuron
        assert 'layer' in neuron
        assert 'activation' in neuron
        assert 'importance' in neuron
        
def test_neuron_pattern_fitting(mlp_analyzer, helix_fitter):
    # First fit helix
    a_values = np.linspace(0, 10, 100)
    activations = np.column_stack([
        a_values,
        np.cos(2 * np.pi * a_values / 5),
        np.sin(2 * np.pi * a_values / 5)
    ])
    helix_fitter.fit(activations, a_values)
    
    # Get neurons and fit patterns
    prompt = "Output ONLY a number. 5+3="
    neurons = mlp_analyzer.analyze_neurons(prompt, layer=15, top_k=10)
    fitted_neurons = mlp_analyzer.fit_neuron_patterns(neurons, 5, 3, helix_fitter)
    
    assert len(fitted_neurons) == len(neurons)
    for neuron in fitted_neurons:
        assert 'a_correlation' in neuron
        assert 'b_correlation' in neuron
        assert 'sum_correlation' in neuron
        assert 'primary_function' in neuron
        assert neuron['primary_function'] in ['a', 'b', 'a+b']
        
def test_mlp_circuit_analysis(mlp_analyzer):
    prompt = "Output ONLY a number. 5+3="
    corrupted = "Output ONLY a number. 6+3="
    
    circuit = mlp_analyzer.analyze_mlp_circuit(prompt, corrupted)
    
    assert isinstance(circuit, dict)
    assert 'builder_mlps' in circuit
    assert 'output_mlps' in circuit
    
    # Check layer assignments
    for mlp in circuit['builder_mlps']:
        assert mlp['layer'] <= 18
    for mlp in circuit['output_mlps']:
        assert mlp['layer'] > 18 