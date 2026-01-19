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


def calculate_vocabulary_diversity(text):
    """
    Calculate vocabulary diversity as the ratio of unique words to total words.
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of unique words to total words (0.0-1.0)
    """
    # Normalize to lowercase and split
    words = text.lower().split()
    
    if len(words) == 0:
        return 1.0
    
    unique_words = len(set(words))
    diversity = unique_words / len(words)
    
    return diversity


def calculate_consecutive_repetition_ratio(text):
    """
    Calculate the ratio of consecutively repeated words.
    Detects patterns like "she she she" or "I, I, I".
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of repeated consecutive words to total words
    """
    words = text.lower().split()
    
    if len(words) <= 1:
        return 0.0
    
    repetitions = 0
    for i in range(1, len(words)):
        # Remove punctuation for comparison
        prev_word = re.sub(r'[^\w\s]', '', words[i-1])
        curr_word = re.sub(r'[^\w\s]', '', words[i])
        
        if prev_word and curr_word and prev_word == curr_word:
            repetitions += 1
    
    ratio = repetitions / len(words)
    return ratio


def has_unicode_errors(text):
    """
    Check if text contains Unicode replacement characters (�).
    
    Args:
        text: Input text string
        
    Returns:
        bool: True if Unicode errors are present
    """
    return '�' in text or '\ufffd' in text


def calculate_filler_word_ratio(text):
    """
    Calculate the ratio of filler words (um, uh, like, etc.) to total words.
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of filler words to total words
    """
    filler_words = {'um', 'uh', 'umm', 'uhh', 'er', 'err', 'hmm', 'hm'}
    
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    
    # Remove punctuation from words before checking
    filler_count = 0
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word in filler_words:
            filler_count += 1
    
    ratio = filler_count / len(words)
    return ratio


def calculate_single_char_word_ratio(text):
    """
    Calculate the ratio of single-character words (excluding 'a', 'I').
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of problematic single-char words to total words
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    allowed_single_chars = {'a', 'A', 'i', 'I'}
    single_char_count = 0
    
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        if len(clean_word) == 1 and clean_word not in allowed_single_chars:
            single_char_count += 1
    
    ratio = single_char_count / len(words)
    return ratio


def count_quality_violations(text):
    """
    Count the number of text quality constraint violations.
    
    Args:
        text: Input text string
        
    Returns:
        int: Number of violations (0-9)
    """
    violations = 0
    
    # Constraint 1: Mean word length is between 3 to 10 characters
    mean_length = calculate_mean_word_length(text)
    if mean_length < 3.0 or mean_length > 10.0:
        violations += 1
    
    # Constraint 2: Symbol-to-word ratio is less than 0.3
    # (allows normal punctuation but catches excessive symbols)
    symbol_ratio = calculate_symbol_to_word_ratio(text)
    if symbol_ratio >= 0.3:
        violations += 1
    
    # Constraint 3: 80% of words contain at least one alphabetic character
    alpha_ratio = calculate_alphabetic_word_ratio(text)
    if alpha_ratio < 0.8:
        violations += 1
    
    # Constraint 4: Vocabulary diversity should be at least 0.3
    # (i.e., at least 30% of words should be unique)
    vocab_diversity = calculate_vocabulary_diversity(text)
    if vocab_diversity < 0.3:
        violations += 1
    
    # Constraint 5: Consecutive word repetition should be less than 10%
    consecutive_rep = calculate_consecutive_repetition_ratio(text)
    if consecutive_rep > 0.1:
        violations += 1
    
    # Constraint 6: No Unicode errors
    if has_unicode_errors(text):
        violations += 1
    
    # Constraint 7: Filler word ratio should be less than 5%
    filler_ratio = calculate_filler_word_ratio(text)
    if filler_ratio > 0.05:
        violations += 1
    
    # Constraint 8: Single character word ratio (excluding 'a', 'I') should be less than 5%
    single_char_ratio = calculate_single_char_word_ratio(text)
    if single_char_ratio > 0.05:
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
    
    # Test cases including the problematic samples from user
    test_texts = [
        # Good quality text (0 violations)
        "This is a normal sentence with good quality words and proper structure.",
        
        # Problematic sample 1: Excessive repetition
        "she was so She and she like a she she her She she she to be she she to be, to a her her.",
        
        # Problematic sample 2: Short words and excessive punctuation
        "- it, and- I was- I and I, it. Because of, with, for, the, was, I- the, my and.",
        
        # Problematic sample 3: Quote and contraction repetition
        "'s more like a \" should If if're\"If it're , \" \" If If, \"If're \" \" If're If,If't",
        
        # Problematic sample 4: Unicode errors
        "\" a and a husband\" married, is a ,� a,� man. � , a,� I is �,� is,� � ,�",
        
        # Problematic sample 5: Filler words
        "\" go\" and\" the, I . I'm, I who I am\" I, I, I, I, I. um, uh, I's, is . I, that's",
        
        # Test with ~100 token samples (75-80 words)
        # Good quality long text
        "The research team conducted extensive experiments to validate their hypothesis about machine learning performance. They analyzed various datasets and implemented multiple algorithms to compare results. The findings were published in a prestigious journal and received positive feedback from experts in the field. This work contributes significantly to our understanding of artificial intelligence and its practical applications in solving complex problems that benefit society.",
        
        # Borderline repetitive (should be caught)
        "The study was conducted by researchers who studied various methods. The methods were studied extensively. They studied the data and studied the results. The study showed that studying these methods helps researchers understand research better. Further study is needed to study these findings more thoroughly and study additional research questions.",
        
        # Low diversity text (should be caught)
        "It was a day. A day like many days. The day was good. Good days are days we remember. Days come and go. We go through days. Days of work. Work days and rest days. Every day brings something. Something happens each day. Days are important. Important days matter most.",
    ]
    
    print("\n=== Quality Property Test ===\n")
    
    for i, text in enumerate(test_texts, 1):
        # Calculate violations
        violations_tensor = calc_quality_violations_parallel([text], 1, torch.device('cpu'))
        violations = violations_tensor.item()
        
        # Calculate individual metrics
        mean_length = calculate_mean_word_length(text)
        symbol_ratio = calculate_symbol_to_word_ratio(text)
        alpha_ratio = calculate_alphabetic_word_ratio(text)
        vocab_diversity = calculate_vocabulary_diversity(text)
        consecutive_rep = calculate_consecutive_repetition_ratio(text)
        has_unicode = has_unicode_errors(text)
        filler_ratio = calculate_filler_word_ratio(text)
        single_char_ratio = calculate_single_char_word_ratio(text)
        
        print(f"Sample {i}: {text[:60]}...")
        print(f"  Mean word length: {mean_length:.2f} (valid: 3-10)")
        print(f"  Symbol-to-word ratio: {symbol_ratio:.3f} (valid: <0.1)")
        print(f"  Alphabetic word ratio: {alpha_ratio:.3f} (valid: >=0.8)")
        print(f"  Vocabulary diversity: {vocab_diversity:.3f} (valid: >=0.3)")
        print(f"  Consecutive repetition: {consecutive_rep:.3f} (valid: <0.1)")
        print(f"  Has Unicode errors: {has_unicode}")
        print(f"  Filler word ratio: {filler_ratio:.3f} (valid: <0.05)")
        print(f"  Single char ratio: {single_char_ratio:.3f} (valid: <0.05)")
        print(f"  *** Total violations: {int(violations)} ***")
        print()


def test_symbol_ratio_on_problematic_samples():
    """Test symbol ratio specifically on the problematic samples."""
    
    problematic_texts = [
        # Sample 2: Excessive punctuation and dashes
        "- it, and- I was- I and I, it. Because of, with, for, the, was, I- the, my and. a a, I, a, my, the, to, to that, of the, of. and, me, the- I, the, I. the. this, I- I. a, was, to.",
        
        # Sample 3: Excessive quotes and contractions
        "'s more like a \" should If if 're\"If it're , \" \" If If, \"If're \" \" If're If,If't \"If're, if be if \" \"if, if \"If \" If You're Like \"If You're\" or \" if \"If's You, If, If \"",
        
        # Sample 4: Unicode errors and excessive commas
        "\" a and a husband\" married, is a ,� a,� man. � , a,� I is �,� is,� � ,� I, an,�, at �,� , , a member of the,�, , I, is ,�, , is ,�, , is,",
    ]
    
    print("\n=== Symbol Ratio Test on Problematic Samples ===\n")
    print(f"Current threshold: symbol_ratio < 0.3\n")
    
    for i, text in enumerate(problematic_texts, 1):
        symbol_ratio = calculate_symbol_to_word_ratio(text)
        violations = count_quality_violations(text)
        
        # Count actual symbols
        symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        word_count = len(text.split())
        
        print(f"Sample {i}: {text[:50]}...")
        print(f"  Words: {word_count}")
        print(f"  Symbols: {symbol_count}")
        print(f"  Symbol-to-word ratio: {symbol_ratio:.3f}")
        print(f"  Threshold: 0.3")
        print(f"  {'✓ CAUGHT' if symbol_ratio >= 0.3 else '✗ MISSED'} by symbol ratio constraint")
        print(f"  Total violations: {violations}")
        print()


if __name__ == "__main__":
    test_symbol_ratio_on_problematic_samples()
    print("\n" + "="*60 + "\n")
    test_quality_property()
