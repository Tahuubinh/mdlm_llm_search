"""
Word n-gram repetition property calculation.
Counts the number of constraint violations for word n-gram repetition ratios.

This implements the filter described in the paper:
"Filter on word repetition ratio: We remove documents that have commonly repeated 
similar long sentences. We define the word repetition ratio as the ratio of the 
sum of the occurrences greater than or equal to 2 to the sum of all occurrences."

Constraints:
1. Word 3-gram repetition ratio <= 0.30
2. Word 4-gram repetition ratio <= 0.25
3. Word 5-gram repetition ratio <= 0.20

Returns: Number of violations (0-3) for each text sample.
Lower is better (0 violations = best).
"""

import torch
from tqdm import tqdm
import re


def normalize_text_for_ngrams(text):
    """
    Normalize text for n-gram analysis following NLP best practices.
    
    - Convert to lowercase to handle case variations
    - Remove punctuation or separate it from words
    - Split into words
    
    Args:
        text: Input text string
        
    Returns:
        list: List of normalized words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep only alphanumeric and spaces)
    # This follows Gopher/RefinedWeb approach
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words and remove empty strings
    words = [word for word in text.split() if word]
    
    return words


def calculate_word_ngram_repetition_ratio(text, n):
    """
    Calculate the word n-gram repetition ratio.
    
    Ratio = (sum of occurrences >= 2) / (sum of all occurrences)
    
    This measures how much of the text consists of repeated word sequences.
    Following Gopher/RefinedWeb/CCNet methodology.
    
    Args:
        text: Input text string OR list of normalized words
        n: N-gram size (number of words)
        
    Returns:
        float: Repetition ratio (0.0-1.0)
    """
    # Accept either string or pre-normalized word list
    if isinstance(text, str):
        words = normalize_text_for_ngrams(text)
    else:
        words = text
    
    if len(words) < n:
        return 0.0
    
    # Count word n-grams
    ngram_counts = {}
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])  # Use tuple as dictionary key
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    if not ngram_counts:
        return 0.0
    
    # Calculate sums
    sum_all_occurrences = sum(ngram_counts.values())
    sum_repeated_occurrences = sum(count for count in ngram_counts.values() if count >= 2)
    
    # Calculate ratio
    if sum_all_occurrences == 0:
        return 0.0
    
    ratio = sum_repeated_occurrences / sum_all_occurrences
    return ratio


def count_word_repetition_violations(text):
    """
    Count the number of word n-gram repetition constraint violations.
    
    Constraints (relaxed thresholds, use 0.20/0.15/0.10 for strict Gopher/RefinedWeb):
    1. Word 3-gram repetition ratio <= 0.30
    2. Word 4-gram repetition ratio <= 0.25
    3. Word 5-gram repetition ratio <= 0.20
    
    Args:
        text: Input text string
        
    Returns:
        int: Number of violations (0-3)
    """
    # Normalize text once for efficiency
    words = normalize_text_for_ngrams(text)
    
    if len(words) < 3:
        return 0  # Text too short to evaluate
    
    violations = 0
    
    # Constraint 1: 3-gram <= 0.30
    ratio_3 = calculate_word_ngram_repetition_ratio(words, 3)
    if ratio_3 > 0.30:
        violations += 1
    
    # Constraint 2: 4-gram <= 0.25 (only if text long enough)
    if len(words) >= 4:
        ratio_4 = calculate_word_ngram_repetition_ratio(words, 4)
        if ratio_4 > 0.25:
            violations += 1
    
    # Constraint 3: 5-gram <= 0.20 (only if text long enough)
    if len(words) >= 5:
        ratio_5 = calculate_word_ngram_repetition_ratio(words, 5)
        if ratio_5 > 0.20:
            violations += 1
    
    return violations


def calc_word_repetition_violations_parallel(texts, batch_size, device):
    """
    Calculate word n-gram repetition violations for a batch of texts in parallel.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        device: Device to use (CPU or CUDA)
        
    Returns:
        torch.Tensor: Violation counts for each text [batch_size]
    """
    violations = []
    
    # Process texts with progress bar
    for text in tqdm(texts, desc="Word Repetition", leave=False):
        violation_count = count_word_repetition_violations(text)
        violations.append(violation_count)
    
    # Convert to tensor
    violations_tensor = torch.tensor(violations, dtype=torch.float32, device=device)
    
    return violations_tensor


def test_word_repetition_property():
    """Test function for word repetition property calculator."""
    
    # Test cases
    test_texts = [
        # Good quality text (0 violations)
        "This is a normal sentence with no repeated phrases. Each phrase appears only once in the text.",
        
        # Test case sensitivity - should be treated as repeats
        "The Model is good. The model is good. THE MODEL IS GOOD.",
        
        # Test punctuation handling
        "Hello world. Hello world! Hello world? Hello world...",
        
        # Moderate repetition (1 violation - 3-gram)
        "I went to the store. I went to the park. I went to the library. I went to the school.",
        
        # High repetition (multiple violations)
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        
        # Very high repetition (all violations)
        "Hello world hello world hello world hello world hello world hello world hello world hello world.",
        
        # Long exact duplicated sentences
        "This is a very long sentence that appears multiple times in the document. This is a very long sentence that appears multiple times in the document. This is a very long sentence that appears multiple times in the document.",
        
        # Short text (0 violations - too short)
        "Hello world",
    ]
    
    print("\n=== Word Repetition Property Test ===\n")
    
    for i, text in enumerate(test_texts):
        # Calculate violations
        violations = count_word_repetition_violations(text)
        
        # Normalize once for testing
        words = normalize_text_for_ngrams(text)
        
        # Calculate individual ratios
        ratio_3 = calculate_word_ngram_repetition_ratio(words, 3)
        ratio_4 = calculate_word_ngram_repetition_ratio(words, 4)
        ratio_5 = calculate_word_ngram_repetition_ratio(words, 5)
        
        print(f"Text {i+1}: {text[:80]}...")
        print(f"  Normalized words: {len(words)} words")
        print(f"  3-gram ratio: {ratio_3:.4f} (threshold: 0.30) {'❌' if ratio_3 > 0.30 else '✓'}")
        print(f"  4-gram ratio: {ratio_4:.4f} (threshold: 0.25) {'❌' if ratio_4 > 0.25 else '✓'}")
        print(f"  5-gram ratio: {ratio_5:.4f} (threshold: 0.20) {'❌' if ratio_5 > 0.20 else '✓'}")
        print(f"  Total violations: {violations}")
        print()


if __name__ == "__main__":
    test_word_repetition_property()
