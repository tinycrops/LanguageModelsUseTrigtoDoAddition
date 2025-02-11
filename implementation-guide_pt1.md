# Implementation Guide: "Language Models Use Trigonometry to Do Addition"

## Critical Improvements Needed

1. **Data Handling**
   - The current data generation is too simplistic and needs validation
   - Should include handling for different model tokenizations
   - Need to add input validation for numerical ranges
   - Should include proper dataset splitting for validation

2. **Model Loading**
   - Add proper error handling for model loading
   - Include memory optimization techniques
   - Add model validation steps
   - Consider adding gradient checkpointing

3. **Helical Analysis**
   - The helical fitting needs regularization
   - Should add cross-validation
   - Need proper numerical stability checks
   - Should include confidence metrics for fits

## Detailed Implementation Guide

### I. Environment Setup

```bash
# Create a new conda environment
conda create -n helix python=3.10
conda activate helix

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

requirements.txt (expanded):
```
torch>=2.1.2
transformers>=4.36.2
accelerate>=0.26.1
datasets>=2.16.1
einops>=0.7.0
numpy>=1.26.3
scipy>=1.11.4
scikit-learn>=1.3.2
matplotlib>=3.8.2
tqdm>=4.66.1
nnsight>=0.2.3
wandb>=0.16.2  # For experiment tracking
pytest>=7.4.4  # For testing
black>=23.12.1 # For code formatting
mypy>=1.8.0   # For type checking
```

### II. Project Structure

```
project/
├── configs/
│   ├── model_configs.yaml
│   └── experiment_configs.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── validation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loading.py
│   │   └── validation.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── helix_fitting.py
│   │   ├── activation_patching.py
│   │   └── component_analysis.py
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_analysis.py
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   └── 2_helix_analysis.ipynb
└── README.md
```

### III. Core Implementation Components

#### 1. Improved Data Generation and Validation

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from transformers import PreTrainedTokenizer

@dataclass
class AdditionExample:
    a: int
    b: int
    prompt: str
    answer: str
    tokenized_length: int

class AdditionDataset:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_num: int = 99,
        min_num: int = 0,
        validation_split: float = 0.1
    ):
        self.tokenizer = tokenizer
        self.max_num = max_num
        self.min_num = min_num
        self.validation_split = validation_split
        
    def generate_example(self) -> Optional[AdditionExample]:
        """Generate a single valid addition example."""
        a = np.random.randint(self.min_num, self.max_num + 1)
        b = np.random.randint(self.min_num, self.max_num + 1)
        
        # Validate sum is within single token range
        if a + b > self.max_num:
            return None
            
        prompt = f"Output ONLY a number. {a}+{b}="
        answer = str(a + b)
        
        # Validate tokenization
        tokens = self.tokenizer.encode(prompt)
        answer_tokens = self.tokenizer.encode(answer)
        
        if len(answer_tokens) > 1:
            return None
            
        return AdditionExample(
            a=a,
            b=b,
            prompt=prompt,
            answer=answer,
            tokenized_length=len(tokens)
        )
    
    def generate_dataset(
        self, 
        num_samples: int
    ) -> Tuple[List[AdditionExample], List[AdditionExample]]:
        """Generate train/val split of addition examples."""
        examples = []
        while len(examples) < num_samples:
            example = self.generate_example()
            if example is not None:
                examples.append(example)
                
        # Split into train/val
        split_idx = int(len(examples) * (1 - self.validation_split))
        return examples[:split_idx], examples[split_idx:]

    def validate_example(self, example: AdditionExample) -> bool:
        """Validate a single addition example."""
        # Check numerical ranges
        if not (self.min_num <= example.a <= self.max_num):
            return False
        if not (self.min_num <= example.b <= self.max_num):
            return False
        if not (self.min_num <= (example.a + example.b) <= self.max_num):
            return False
            
        # Validate tokenization
        tokens = self.tokenizer.encode(example.prompt)
        answer_tokens = self.tokenizer.encode(example.answer)
        
        if len(answer_tokens) > 1:
            return False
        if len(tokens) != example.tokenized_length:
            return False
            
        return True
```

#### 2. Enhanced Helix Fitting

```python
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
        self.periods = periods
        self.n_pca_components = n_pca_components
        self.regularization_strength = regularization_strength
        self.n_cross_val_folds = n_cross_val_folds
        self.pca = None
        self.C = None
        self.fit_quality = None
        
    def _create_basis(self, a_values: np.ndarray) -> np.ndarray:
        """Create basis functions for helix fitting."""
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
        """Compute various metrics for fit quality."""
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
        """Fit helix to activations with regularization and validation."""
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
        """Transform new a values using fitted helix."""
        if self.C is None:
            raise ValueError("Must fit model before transform")
            
        B = self._create_basis(a_values)
        return B @ self.C.T
```

#### 3. Activation Patching with Stability Checks

```python
from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F

class ActivationPatcher:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        stability_samples: int = 10,
        patch_scale: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stability_samples = stability_samples
        self.patch_scale = patch_scale
        
    def _get_clean_activations(
        self,
        prompt: str,
        layer: int
    ) -> Tensor:
        """Get activations for clean prompt with stability check."""
        acts = []
        for _ in range(self.stability_samples):
            with torch.no_grad():
                with self.model.invoke(prompt) as invoker:
                    hidden_states = (
                        self.model.model.transformer.h[layer].output[0].save()
                    )
                acts.append(hidden_states.value)
                
        # Check stability
        acts = torch.stack(acts)
        std = torch.std(acts, dim=0)
        if torch.any(std > 1e-5):
            print(f"Warning: Unstable activations detected in layer {layer}")
            
        return acts.mean(dim=0)
        
    def patch_activation(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer: int,
        intervention_fn: Callable[[Tensor], Tensor],
        token_idx: Optional[int] = -1
    ) -> dict:
        """Perform activation patching with diagnostics."""
        # Get clean activations
        clean_acts = self._get_clean_activations(clean_prompt, layer)
        
        # Get corrupted run
        with self.model.invoke(corrupted_prompt) as invoker:
            corrupted_hidden_states = (
                self.model.model.transformer.h[layer].output[0].save()
            )
            corrupted_logits = self.model.lm_head.output.save()
            
        corr_acts = corrupted_hidden_states.value
        corr_logits = corrupted_logits.value
        
        # Apply patch
        with self.model.invoke(corrupted_prompt) as invoker:
            def intervention(output, layer_idx):
                if layer_idx == layer:
                    # Apply intervention with scaling
                    patched = intervention_fn(output)
                    return (
                        self.patch_scale * patched + 
                        (1 - self.patch_scale) * output
                    )
                return output
                
            self.model.model.transformer.h[layer].output[0][:, token_idx, :] = (
                intervention(corr_acts[:, token_idx, :], layer)
            )
            patched_logits = self.model.lm_head.output.save()
            
        patch_logits = patched_logits.value
        
        # Calculate metrics
        logit_diff = patch_logits - corr_logits
        
        # Get probabilities
        corr_probs = F.softmax(corr_logits, dim=-1)
        patch_probs = F.softmax(patch_logits, dim=-1)
        
        return {
            'logit_diff': logit_diff,
            'prob_diff': patch_probs - corr_probs,
            'corr_probs': corr_probs,
            'patch_probs': patch_probs
        }
```

### IV. Running Experiments and Analysis

#### Additional Utility Functions

```python
def compute_direct_effect(
    model,
    example: AdditionExample,
    layer: int
) -> float:
    """Compute direct effect of a layer using path patching."""
    # Implementation follows path patching methodology from paper
    pass

def identify_important_layers(layer_results: dict) -> List[int]:
    """Identify layers with significant effects on computation."""
    effects = np.array([
        results['patching_results']['total_effect']
        for results in layer_results.values()
    ])
    return list(np.where(effects > np.mean(effects) + np.std(effects))[0])

def analyze_layer_neurons(
    model,
    examples: List[AdditionExample],
    layer: int,
    activation_patcher: ActivationPatcher
) -> dict:
    """Analyze individual neurons in a layer."""
    # Implementation follows neuron analysis methodology from paper
    pass

def save_results_to_disk(
    save_path: str,
    layer_results: dict,
    component_results: dict,
    baseline_metrics: dict
):
    """Save analysis results to disk."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save results as JSON
    with open(f"{save_path}/layer_results.json", 'w') as f:
        json.dump(layer_results, f)
    with open(f"{save_path}/component_results.json", 'w') as f:
        json.dump(component_results, f)
    with open(f"{save_path}/baseline_metrics.json", 'w') as f:
        json.dump(baseline_metrics, f)
        
    # Save visualizations
    generate_and_save_visualizations(
        save_path,
        layer_results,
        component_results,
        baseline_metrics
    )
```

#### Helper Functions for Component Analysis

1. First validate the environment and dependencies:

```python
```python
def analyze_attention_heads(
    model,
    examples: List[AdditionExample],
    activation_patcher: ActivationPatcher
) -> dict:
    """Detailed analysis of attention heads."""
    results = {}
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    
    # Analyze each head's contribution
    head_effects = np.zeros((num_layers, num_heads))
    head_patterns = []
    
    for layer in tqdm(range(num_layers), desc="Analyzing layers"):
        for head in range(num_heads):
            # Compute total effect
            effects = []
            patterns = []
            
            for example in examples:
                # Get attention patterns
                with model.invoke(example.prompt) as invoker:
                    attn_pattern = (
                        model.model.transformer.h[layer]
                        .attn.output[0][:, head].save()
                    )
                patterns.append(attn_pattern.value.cpu().numpy())
                
                # Compute patching effect
                patch_results = activation_patcher.patch_activation(
                    example.prompt,
                    example.prompt.replace(
                        str(example.a),
                        str((example.a + 1) % 100)
                    ),
                    layer,
                    lambda x: x[:, head] * 0  # Zero out head
                )
                effects.append(
                    patch_results['logit_diff'][0, -1].mean().item()
                )
            
            head_effects[layer, head] = np.mean(effects)
            head_patterns.append(np.mean(patterns, axis=0))
    
    # Categorize heads
    head_categories = categorize_heads(
        head_effects,
        head_patterns,
        threshold=0.1
    )
    
    results['effects'] = head_effects
    results['patterns'] = head_patterns
    results['categories'] = head_categories
    
    return results

def analyze_mlp_layers(
    model,
    examples: List[AdditionExample],
    activation_patcher: ActivationPatcher
) -> dict:
    """Detailed analysis of MLP layers."""
    results = {}
    num_layers = model.config.n_layer
    
    # Analyze each MLP's contribution
    mlp_effects = []
    mlp_direct_effects = []
    
    for layer in tqdm(range(num_layers), desc="Analyzing MLPs"):
        layer_effects = []
        layer_direct_effects = []
        
        for example in examples:
            # Compute total effect (activation patching)
            patch_results = activation_patcher.patch_activation(
                example.prompt,
                example.prompt.replace(
                    str(example.a),
                    str((example.a + 1) % 100)
                ),
                layer,
                lambda x: torch.zeros_like(x)  # Zero out MLP
            )
            layer_effects.append(
                patch_results['logit_diff'][0, -1].mean().item()
            )
            
            # Compute direct effect (path patching)
            direct_effect = compute_direct_effect(
                model,
                example,
                layer
            )
            layer_direct_effects.append(direct_effect)
        
        mlp_effects.append(np.mean(layer_effects))
        mlp_direct_effects.append(np.mean(layer_direct_effects))
    
    results['total_effects'] = mlp_effects
    results['direct_effects'] = mlp_direct_effects
    
    return results

def analyze_neurons(
    model,
    examples: List[AdditionExample],
    activation_patcher: ActivationPatcher,
    layer_results: dict
) -> dict:
    """Detailed analysis of individual neurons."""
    results = {}
    
    # Get important layers from MLP analysis
    important_layers = identify_important_layers(layer_results)
    
    neuron_results = {}
    for layer in important_layers:
        layer_neurons = analyze_layer_neurons(
            model,
            examples,
            layer,
            activation_patcher
        )
        neuron_results[layer] = layer_neurons
    
    results['neurons'] = neuron_results
    
    return results

def validate_environment():
    """Validate environment setup and dependencies."""
    import torch
    import transformers
    import nnsight
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
        
    # Check memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory < 20e9:  # 20GB
        print("Warning: Less than 20GB GPU memory available")
        
    # Test model loading
    try:
        model_name = "EleutherAI/gpt-j-6B"
        model = nnsight.LanguageModel(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
```

2. Run the main analysis pipeline:

```python
def run_analysis_pipeline(
    model_name: str,
    num_samples: int = 10000,
    max_num: int = 99,
    save_results: bool = True,
    experiment_name: Optional[str] = None
):
    """Run complete analysis pipeline with comprehensive logging and analysis."""
    # Initialize logging
    if experiment_name is None:
        experiment_name = f"helix_analysis_{int(time.time())}"
    wandb.init(project="helix_analysis", name=experiment_name)
    
    # Initialize components
    model = nnsight.LanguageModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize analyzers
    dataset = AdditionDataset(tokenizer, max_num=max_num)
    helix_fitter = HelixFitter()
    activation_patcher = ActivationPatcher(model, tokenizer)
    
    # Generate and validate dataset
    print("Generating dataset...")
    train_examples, val_examples = dataset.generate_dataset(num_samples)
    wandb.log({
        "train_size": len(train_examples),
        "val_size": len(val_examples)
    })
    
    # Analyze model performance baseline
    print("Computing baseline performance...")
    baseline_metrics = compute_baseline_performance(model, val_examples)
    wandb.log(baseline_metrics)
    
    # Perform helix analysis for each layer
    print("Starting layer-wise helix analysis...")
    layer_results = {}
    for layer in tqdm(range(model.config.n_layer)):
        layer_results[layer] = analyze_layer(
            model,
            train_examples,
            val_examples,
            layer,
            helix_fitter,
            activation_patcher
        )
        wandb.log({f"layer_{layer}": layer_results[layer]})
        
    # Perform component analysis
    print("Starting component analysis...")
    component_results = analyze_components(
        model,
        val_examples,
        layer_results,
        activation_patcher
    )
    wandb.log({"component_analysis": component_results})
    
    # Save results
    if save_results:
        save_path = f"results/{experiment_name}"
        os.makedirs(save_path, exist_ok=True)
        save_results_to_disk(
            save_path,
            layer_results,
            component_results,
            baseline_metrics
        )
        
    return layer_results, component_results

def analyze_layer(
    model,
    train_examples: List[AdditionExample],
    val_examples: List[AdditionExample],
    layer: int,
    helix_fitter: HelixFitter,
    activation_patcher: ActivationPatcher
) -> dict:
    """Perform comprehensive analysis of a single layer."""
    results = {}
    
    # Get residual stream activations
    train_activations = get_layer_activations(model, train_examples, layer)
    val_activations = get_layer_activations(model, val_examples, layer)
    
    # Fit helix
    C, fit_metrics = helix_fitter.fit(train_activations)
    results['helix_fit_metrics'] = fit_metrics
    
    # Validate helix fit on validation set
    val_metrics = validate_helix_fit(
        helix_fitter,
        val_activations,
        val_examples
    )
    results['helix_validation_metrics'] = val_metrics
    
    # Perform activation patching experiments
    patch_results = run_patching_experiments(
        activation_patcher,
        val_examples,
        layer,
        helix_fitter
    )
    results['patching_results'] = patch_results
    
    # Analyze Fourier components
    fourier_results = analyze_fourier_components(
        train_activations,
        val_activations
    )
    results['fourier_analysis'] = fourier_results
    
    return results

def analyze_components(
    model,
    val_examples: List[AdditionExample],
    layer_results: dict,
    activation_patcher: ActivationPatcher
) -> dict:
    """Analyze model components (attention heads and MLPs)."""
    results = {}
    
    # Analyze attention heads
    print("Analyzing attention heads...")
    head_results = analyze_attention_heads(
        model,
        val_examples,
        activation_patcher
    )
    results['attention_heads'] = head_results
    
    # Analyze MLPs
    print("Analyzing MLPs...")
    mlp_results = analyze_mlp_layers(
        model,
        val_examples,
        activation_patcher
    )
    results['mlps'] = mlp_results
    
    # Analyze top neurons
    print("Analyzing neurons...")
    neuron_results = analyze_neurons(
        model,
        val_examples,
        activation_patcher,
        layer_results
    )
    results['neurons'] = neuron_results
    
    return results
    