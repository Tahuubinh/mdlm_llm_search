"""
N-gram repetition property calculation.
Counts the number of constraint violations for character n-gram repetitions.

Constraints:
1. Top 2-gram character fraction <= 0.20
2. Top 3-gram character fraction <= 0.18
3. Top 4-gram character fraction <= 0.16
4. Duplicate 5-gram character fraction <= 0.15
5. Duplicate 6-gram character fraction <= 0.14
6. Duplicate 7-gram character fraction <= 0.13
7. Duplicate 8-gram character fraction <= 0.12
8. Duplicate 9-gram character fraction <= 0.11
9. Duplicate 10-gram character fraction <= 0.10

Returns: Number of violations (0-9) for each text sample.
Lower is better (0 violations = best).
"""

import torch
from tqdm import tqdm


def calculate_ngram_char_fraction(text, n):
    """
    Calculate the fraction of characters covered by the most frequent n-gram.
    
    Args:
        text: Input text string
        n: N-gram size
        
    Returns:
        float: Fraction of characters in text covered by the most frequent n-gram (0.0-1.0)
    """
    if len(text) < n:
        return 0.0
    
    # Count n-grams
    ngram_counts = {}
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    if not ngram_counts:
        return 0.0
    
    # Find the most frequent n-gram
    max_ngram, max_count = max(ngram_counts.items(), key=lambda x: x[1])
    
    # Find all positions where this n-gram occurs
    covered_indices = set()
    for i in range(len(text) - n + 1):
        if text[i:i+n] == max_ngram:
            for j in range(i, i+n):
                covered_indices.add(j)
    
    # Calculate fraction
    fraction = len(covered_indices) / len(text)
    return fraction


def calculate_duplicate_ngram_char_fraction(text, n):
    """
    Calculate the fraction of characters covered by duplicate n-grams.
    
    Args:
        text: Input text string
        n: N-gram size
        
    Returns:
        float: Fraction of characters covered by n-grams that appear more than once (0.0-1.0)
    """
    if len(text) < n:
        return 0.0
    
    # Count n-grams
    ngram_counts = {}
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    # Find all positions covered by duplicate n-grams
    covered_indices = set()
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        if ngram_counts[ngram] > 1:  # This n-gram appears more than once
            for j in range(i, i+n):
                covered_indices.add(j)
    
    # Calculate fraction
    if len(text) == 0:
        return 0.0
    fraction = len(covered_indices) / len(text)
    return fraction


def count_repetition_violations(text):
    """
    Count the number of n-gram repetition constraint violations.
    
    Constraints:
    1. Top 2-gram character fraction <= 0.20
    2. Top 3-gram character fraction <= 0.18
    3. Top 4-gram character fraction <= 0.16
    4. Duplicate 5-gram character fraction <= 0.15
    5. Duplicate 6-gram character fraction <= 0.14
    6. Duplicate 7-gram character fraction <= 0.13
    7. Duplicate 8-gram character fraction <= 0.12
    8. Duplicate 9-gram character fraction <= 0.11
    9. Duplicate 10-gram character fraction <= 0.10
    
    Args:
        text: Input text string
        
    Returns:
        int: Number of violations (0-9)
    """
    violations = 0
    
    # Top n-gram constraints (n=2,3,4)
    top_ngram_thresholds = {
        2: 0.20,
        3: 0.18,
        4: 0.16
    }
    
    for n, threshold in top_ngram_thresholds.items():
        fraction = calculate_ngram_char_fraction(text, n)
        if fraction > threshold:
            violations += 1
    
    # Duplicate n-gram constraints (n=5,6,7,8,9,10)
    duplicate_ngram_thresholds = {
        5: 0.15,
        6: 0.14,
        7: 0.13,
        8: 0.12,
        9: 0.11,
        10: 0.10
    }
    
    for n, threshold in duplicate_ngram_thresholds.items():
        fraction = calculate_duplicate_ngram_char_fraction(text, n)
        if fraction > threshold:
            violations += 1
    
    return violations


def calc_repetition_violations_parallel(sequence_list, batch_size, device):
    """
    Calculate repetition violation counts for a batch of sequences in parallel.
    
    Args:
        sequence_list (list): List of text sequences (already cleaned, no special tokens)
        batch_size (int): Number of sequences in the batch
        device (torch.device): Device to store results
        
    Returns:
        torch.Tensor: Number of violations (0-9) for each sequence
    """
    # Initialize violations tensor
    violations = torch.zeros(batch_size, device=device)
    
    # Process each text and count violations
    # This is CPU-bound, no need for GPU processing
    for i in tqdm(range(batch_size), desc="Repetition"):
        text = sequence_list[i]
        violation_count = count_repetition_violations(text)
        violations[i] = violation_count
    
    return violations


if __name__ == "__main__":
    # Example usage
    test_texts = [
        "This is a nice and varied text with different words and patterns.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # High 2-gram repetition
        "abcabcabcabcabcabcabcabcabc",  # High 3-gram repetition
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",  # Duplicate phrases
    ]
    
    print("Testing repetition violation counting:")
    for text in test_texts:
        violations = count_repetition_violations(text)
        print(f"\nText: {text[:60]}...")
        print(f"Violations: {violations}")
        
        # Show details for each constraint
        print("  Details:")
        for n in [2, 3, 4]:
            frac = calculate_ngram_char_fraction(text, n)
            print(f"    Top {n}-gram fraction: {frac:.3f} (threshold: {[0.20, 0.18, 0.16][n-2]:.2f})")
        for n in [5, 6, 7, 8, 9, 10]:
            frac = calculate_duplicate_ngram_char_fraction(text, n)
            threshold = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10][n-5]
            print(f"    Duplicate {n}-gram fraction: {frac:.3f} (threshold: {threshold:.2f})")
    
    print("\n\nTesting parallel violation calculation:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_violations = calc_repetition_violations_parallel(test_texts, len(test_texts), device)
    print("Batch violations:", batch_violations)
