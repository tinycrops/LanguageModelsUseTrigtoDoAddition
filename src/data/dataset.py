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
        """Initialize the dataset for addition examples.
        
        Args:
            tokenizer: HuggingFace tokenizer for the target model
            max_num: Maximum number to use in examples
            min_num: Minimum number to use in examples
            validation_split: Fraction of data to use for validation
        """
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
        """Generate train/val split of addition examples.
        
        Args:
            num_samples: Total number of examples to generate
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        examples = []
        while len(examples) < num_samples:
            example = self.generate_example()
            if example is not None:
                examples.append(example)
                
        # Split into train/val
        split_idx = int(len(examples) * (1 - self.validation_split))
        return examples[:split_idx], examples[split_idx:]

    def validate_example(self, example: AdditionExample) -> bool:
        """Validate a single addition example.
        
        Args:
            example: The example to validate
            
        Returns:
            bool indicating if the example is valid
        """
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