"""
Text quality property calculation.
Counts the number of constraint violations for text quality metrics.

Constraints:
1. Mean word length is between 3 to 10 characters
2. Symbol-to-word ratio is less than 0.1
3. 80% of words contain at least one alphabetic character

Returns: Number of violations (0-3) for each text sample.
Lower is better (0 violations = best).
"""

import torch
from tqdm import tqdm
import re


def calculate_mean_word_length(text):
    """
    Calculate the mean length of words in the text.
    
    Args:
        text: Input text string
        
    Returns:
        float: Mean word length in characters
    """
    # Split by whitespace to get words
    words = text.split()
    
    if len(words) == 0:
        return 0.0
    
    total_length = sum(len(word) for word in words)
    mean_length = total_length / len(words)
    
    return mean_length


def calculate_symbol_to_word_ratio(text):
    """
    Calculate the ratio of symbols to words in the text.
    Symbols are non-alphanumeric, non-whitespace characters.
    
    Args:
        text: Input text string
        
    Returns:
        float: Symbol-to-word ratio
    """
    # Split by whitespace to get words
    words = text.split()
    
    if len(words) == 0:
        return 0.0
    
    # Count symbols (non-alphanumeric, non-whitespace characters)
    symbol_count = 0
    for char in text:
        if not char.isalnum() and not char.isspace():
            symbol_count += 1
    
    ratio = symbol_count / len(words)
    return ratio


def calculate_alphabetic_word_ratio(text):
    """
    Calculate the fraction of words that contain at least one alphabetic character.
    
    Args:
        text: Input text string
        
    Returns:
        float: Fraction of words with at least one alphabetic character (0.0-1.0)
    """
    # Split by whitespace to get words
    words = text.split()
    
    if len(words) == 0:
        return 0.0
    
    # Count words with at least one alphabetic character
    alpha_words = 0
    for word in words:
        if any(char.isalpha() for char in word):
            alpha_words += 1
    
    ratio = alpha_words / len(words)
    return ratio


def count_quality_violations(text):
    """
    Count the number of text quality constraint violations.
    
    Args:
        text: Input text string
        
    Returns:
        int: Number of violations (0-3)
    """
    violations = 0
    
    # Constraint 1: Mean word length is between 3 to 10 characters
    mean_length = calculate_mean_word_length(text)
    if mean_length < 3.0 or mean_length > 10.0:
        violations += 1
    
    # # Constraint 2: Symbol-to-word ratio is less than 0.1
    # symbol_ratio = calculate_symbol_to_word_ratio(text)
    # if symbol_ratio >= 0.1:
    #     violations += 1
    
    # Constraint 3: 80% of words contain at least one alphabetic character
    alpha_ratio = calculate_alphabetic_word_ratio(text)
    if alpha_ratio < 0.8:
        violations += 1
    
    return violations


def calc_quality_violations_parallel(texts, batch_size, device):
    """
    Calculate quality violations for a batch of texts in parallel.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        device: Device to use (CPU or CUDA)
        
    Returns:
        torch.Tensor: Violation counts for each text [batch_size]
    """
    violations = []
    
    # Process texts with progress bar
    for text in tqdm(texts, desc="Quality", leave=False):
        violation_count = count_quality_violations(text)
        violations.append(violation_count)
    
    # Convert to tensor
    violations_tensor = torch.tensor(violations, dtype=torch.float32, device=device)
    
    return violations_tensor


def test_quality_property():
    """Test function for quality property calculator."""
    
    # Test cases
    test_texts = [
        # Good quality text (0 violations)
        "This is a normal sentence with good quality words.",
        
        # Short words (violation 1: mean length < 3)
        "I am a boy on my way to go up.",
        
        # Too many symbols (violation 2: symbol ratio >= 0.1)
        "Hello!!! World??? Why??? So??? Many??? Symbols???",
        
        # Too many non-alphabetic words (violation 3: alpha ratio < 0.8)
        "123 456 789 000 111 word1 word2 word3",
        
        # Multiple violations
        "I go. !!! ??? ### 123 456.",
        
        # Very long words (violation 1: mean length > 10)
        "extraordinarily incomprehensible unbelievable",
    ]
    
    print("\n=== Quality Property Test ===\n")
    
    for text in test_texts:
        # Calculate violations
        violations_tensor = calc_quality_violations_parallel([text], 1, torch.device('cpu'))
        violations = violations_tensor.item()
        
        # Calculate individual metrics
        mean_length = calculate_mean_word_length(text)
        symbol_ratio = calculate_symbol_to_word_ratio(text)
        alpha_ratio = calculate_alphabetic_word_ratio(text)
        
        print(f"Text: {text[:60]}...")
        print(f"  Mean word length: {mean_length:.2f} (valid: 3-10)")
        print(f"  Symbol-to-word ratio: {symbol_ratio:.3f} (valid: <0.1)")
        print(f"  Alphabetic word ratio: {alpha_ratio:.3f} (valid: >=0.8)")
        print(f"  Total violations: {int(violations)}")
        print()


if __name__ == "__main__":
    test_quality_property()
