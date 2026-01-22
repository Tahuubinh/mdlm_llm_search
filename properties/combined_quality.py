import torch
from tqdm import tqdm
import re

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

def count_unicode_errors(text):
    """
    Count the number of Unicode replacement characters (�) in the text.
    
    Args:
        text: Input text string
        
    Returns:
        int: Number of Unicode replacement characters
    """
    return text.count('�') + text.count('\ufffd')


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


def calculate_stop_word_ratio(text):
    """
    Calculate the ratio of common English stop words.
    High stop word ratio indicates low content quality.
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of stop words to total words (0.0-1.0)
    """
    # Common English stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'that', 'this',
        'these', 'those', 'it', 'its', 'he', 'she', 'they', 'them', 'their',
        'his', 'her', 'who', 'which', 'what', 'where', 'when', 'why', 'how'
    }
    
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    
    # Count stop words (remove punctuation first)
    stop_word_count = 0
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word in stop_words:
            stop_word_count += 1
    
    ratio = stop_word_count / len(words)
    return ratio


def calculate_two_char_word_ratio(text):
    """
    Calculate the ratio of 2-character words.
    High ratio indicates low quality (e.g., "to to to the the is is").
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of 2-char words to total words (0.0-1.0)
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    two_char_count = 0
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        if len(clean_word) == 2:
            two_char_count += 1
    
    ratio = two_char_count / len(words)
    return ratio


def calculate_max_word_length(text):
    """
    Calculate the maximum word length in the text.
    Very long words (>20 chars) are often gibberish.
    
    Args:
        text: Input text string
        
    Returns:
        int: Maximum word length in characters
    """
    words = text.split()
    if len(words) == 0:
        return 0
    
    max_length = 0
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        if len(clean_word) > max_length:
            max_length = len(clean_word)
    
    return max_length


def calculate_newline_ratio(text):
    """
    Calculate the ratio of newline characters to total characters.
    Too many newlines indicates poor formatting.
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of newlines to total characters (0.0-1.0)
    """
    if len(text) == 0:
        return 0.0
    
    newline_count = text.count('\n')
    ratio = newline_count / len(text)
    
    return ratio


def has_gibberish_pattern(text):
    """
    Check if text contains gibberish patterns:
    - Words with >4 consecutive consonants
    - Words with simple repeating patterns (aa, bb, cc)
    
    Args:
        text: Input text string
        
    Returns:
        bool: True if gibberish patterns detected
    """
    words = text.split()
    consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
    
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        
        # Check for >4 consecutive consonants
        consonant_streak = 0
        for char in clean_word:
            if char in consonants:
                consonant_streak += 1
                if consonant_streak > 4:
                    return True
            else:
                consonant_streak = 0
        
        # Check for repeating 2-char patterns (e.g., "enenen", "ining")
        if len(clean_word) >= 6:
            for i in range(len(clean_word) - 5):
                pattern = clean_word[i:i+2]
                # Check if this 2-char pattern repeats 3+ times consecutively
                if clean_word[i:i+6] == pattern * 3:
                    return True
    
    return False


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

def count_combined_quality_violations(text):
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

    # Normalize text once for efficiency
    words = normalize_text_for_ngrams(text)
    
    if len(words) < 3:
        violations += 3
    else:
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

def measure_combined_quality_violations(text):
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
        2: 0.35,
        3: 0.30,
        4: 0.25
    }
    
    for n, threshold in top_ngram_thresholds.items():
        fraction = calculate_ngram_char_fraction(text, n)
        if fraction > threshold:
            # violations += (fraction - threshold)
            violations += 1
    
    # Duplicate n-gram constraints (n=5,6,7,8,9,10)
    duplicate_ngram_thresholds = {
        5: 0.25,
        6: 0.24,
        7: 0.23,
        8: 0.22,
        9: 0.21,
        10: 0.20
    }
    
    for n, threshold in duplicate_ngram_thresholds.items():
        fraction = calculate_duplicate_ngram_char_fraction(text, n)
        if fraction > threshold:
            # violations += (fraction - threshold)
            violations += 1

    # Constraint 1: Mean word length is between 3 to 10 characters
    mean_length = calculate_mean_word_length(text)
    # if mean_length < 3.0 or mean_length > 10.0:
    #     violations += 1
    if mean_length < 3.0:
        violations += (3.0 - mean_length)
        # violations += 1
    elif mean_length > 10.0:
        violations += (mean_length - 10.0)
        # violations += 1
    
    # # Constraint 2: Symbol-to-word ratio is less than 0.1
    # symbol_ratio = calculate_symbol_to_word_ratio(text)
    # if symbol_ratio >= 0.1:
    #     violations += 1
    
    # Constraint 3: 80% of words contain at least one alphabetic character
    alpha_ratio = calculate_alphabetic_word_ratio(text)
    if alpha_ratio < 0.7:
        # violations += (0.7 - alpha_ratio)
        violations += 1

    # Normalize text once for efficiency
    words = normalize_text_for_ngrams(text)
    
    if len(words) < 3:
        violations += 3
    else:
        # Constraint 1: 3-gram <= 0.30
        ratio_3 = calculate_word_ngram_repetition_ratio(words, 3)
        if ratio_3 > 0.30:
            # violations += (ratio_3 - 0.30)
            violations += 1
        
        # Constraint 2: 4-gram <= 0.25 (only if text long enough)
        if len(words) >= 4:
            ratio_4 = calculate_word_ngram_repetition_ratio(words, 4)
            if ratio_4 > 0.25:
                # violations += (ratio_4 - 0.25)
                violations += 1
        
        # Constraint 3: 5-gram <= 0.20 (only if text long enough)
        if len(words) >= 5:
            ratio_5 = calculate_word_ngram_repetition_ratio(words, 5)
            if ratio_5 > 0.20:
                # violations += (ratio_5 - 0.20)
                violations += 1
    
    return violations

def measure_combined_quality_level_violations(text):
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
        2: 0.35,
        3: 0.30,
        4: 0.25
    }

    # Top n-gram constraints (n=2,3,4)
    top_ngram_level_thresholds = {
        2: 0.20,
        3: 0.18,
        4: 0.16
    }
    
    for n, threshold in top_ngram_thresholds.items():
        fraction = calculate_ngram_char_fraction(text, n)
        if fraction > threshold:
            # violations += (fraction - threshold)
            violations += 1

    for n, threshold in top_ngram_level_thresholds.items():
        fraction = calculate_ngram_char_fraction(text, n)
        if fraction > threshold:
            # violations += (fraction - threshold)
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

    duplicate_ngram_level_thresholds = {
        5: 0.25,
        6: 0.24,
        7: 0.23,
        8: 0.22,
        9: 0.21,
        10: 0.20
    }
    
    for n, threshold in duplicate_ngram_thresholds.items():
        fraction = calculate_duplicate_ngram_char_fraction(text, n)
        if fraction > threshold:
            # violations += (fraction - threshold)
            violations += 1

    for n, threshold in duplicate_ngram_level_thresholds.items():
        fraction = calculate_duplicate_ngram_char_fraction(text, n)
        if fraction > threshold:
            # violations += (fraction - threshold)
            violations += 1

    # Constraint 1: Mean word length is between 3 to 10 characters
    mean_length = calculate_mean_word_length(text)
    # if mean_length < 3.0 or mean_length > 10.0:
    #     violations += 1
    if mean_length < 3.0:
        violations += (3.0 - mean_length)
        # violations += 1
    elif mean_length > 10.0:
        violations += (mean_length - 10.0)
        # violations += 1
    
    # # Constraint 2: Symbol-to-word ratio is less than 0.1
    # symbol_ratio = calculate_symbol_to_word_ratio(text)
    # if symbol_ratio >= 0.1:
    #     violations += 1
    
    # Constraint 3: 80% of words contain at least one alphabetic character
    alpha_ratio = calculate_alphabetic_word_ratio(text)
    if alpha_ratio < 0.7:
        # violations += (0.7 - alpha_ratio)
        violations += 1
    if alpha_ratio < 0.8:
        # violations += (0.7 - alpha_ratio)
        violations += 1

    # Normalize text once for efficiency
    words = normalize_text_for_ngrams(text)
    
    if len(words) < 3:
        violations += 3
    else:
        # Constraint 1: 3-gram <= 0.30
        ratio_3 = calculate_word_ngram_repetition_ratio(words, 3)
        if ratio_3 > 0.30:
            # violations += (ratio_3 - 0.30)
            violations += 1
        
        # Constraint 2: 4-gram <= 0.25 (only if text long enough)
        if len(words) >= 4:
            ratio_4 = calculate_word_ngram_repetition_ratio(words, 4)
            if ratio_4 > 0.25:
                # violations += (ratio_4 - 0.25)
                violations += 1
        
        # Constraint 3: 5-gram <= 0.20 (only if text long enough)
        if len(words) >= 5:
            ratio_5 = calculate_word_ngram_repetition_ratio(words, 5)
            if ratio_5 > 0.20:
                # violations += (ratio_5 - 0.20)
                violations += 1
    
    return violations

def measure_combined_heuristic_violations(text):
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
        2: 0.35,
        3: 0.30,
        4: 0.25
    }
    
    for n, threshold in top_ngram_thresholds.items():
        fraction = calculate_ngram_char_fraction(text, n)
        if fraction > threshold:
            violations += (fraction - threshold)
            # violations += 1
    
    # Duplicate n-gram constraints (n=5,6,7,8,9,10)
    duplicate_ngram_thresholds = {
        5: 0.25,
        6: 0.24,
        7: 0.23,
        8: 0.22,
        9: 0.21,
        10: 0.20
    }
    
    for n, threshold in duplicate_ngram_thresholds.items():
        fraction = calculate_duplicate_ngram_char_fraction(text, n)
        if fraction > threshold:
            violations += (fraction - threshold)
            # violations += 1

    # Constraint 1: Mean word length is between 3 to 10 characters
    mean_length = calculate_mean_word_length(text)
    # if mean_length < 3.0 or mean_length > 10.0:
    #     violations += 1
    if mean_length < 3.0:
        violations += (3.0 - mean_length)
        # violations += 1
    elif mean_length > 10.0:
        violations += (mean_length - 10.0)
        # violations += 1
    
    # Constraint 2: Symbol-to-word ratio is less than 0.3
    # (allows normal punctuation but catches excessive symbols)
    symbol_ratio = calculate_symbol_to_word_ratio(text)
    if symbol_ratio > 0.3:
        # violations += 1
        violations += (symbol_ratio - 0.3)
    
    # Constraint 3: 80% of words contain at least one alphabetic character
    alpha_ratio = calculate_alphabetic_word_ratio(text)
    if alpha_ratio < 0.7:
        violations += (0.7 - alpha_ratio)
        # violations += 1

    # Normalize text once for efficiency
    words = normalize_text_for_ngrams(text)
    
    if len(words) < 3:
        violations += 3
    else:
        # Constraint 1: 3-gram <= 0.30
        ratio_3 = calculate_word_ngram_repetition_ratio(words, 3)
        if ratio_3 > 0.30:
            violations += (ratio_3 - 0.30)
            # violations += 1
        
        # Constraint 2: 4-gram <= 0.25 (only if text long enough)
        if len(words) >= 4:
            ratio_4 = calculate_word_ngram_repetition_ratio(words, 4)
            if ratio_4 > 0.25:
                violations += (ratio_4 - 0.25)
                # violations += 1
        
        # Constraint 3: 5-gram <= 0.20 (only if text long enough)
        if len(words) >= 5:
            ratio_5 = calculate_word_ngram_repetition_ratio(words, 5)
            if ratio_5 > 0.20:
                violations += (ratio_5 - 0.20)
                # violations += 1

    # Constraint 4: Vocabulary diversity should be at least 0.3
    # (i.e., at least 30% of words should be unique)
    vocab_diversity = calculate_vocabulary_diversity(text)
    if vocab_diversity < 0.3:
        # violations += 1
        violations += (0.3 - vocab_diversity)
    
    # Constraint 5: Consecutive word repetition should be less than 10%
    consecutive_rep = calculate_consecutive_repetition_ratio(text)
    if consecutive_rep > 0.1:
        # violations += 1
        violations += (consecutive_rep - 0.1)
    
    # Constraint 6: No Unicode errors
    if count_unicode_errors(text) > 0:
        # violations += 1
        violations += count_unicode_errors(text)
    
    # Constraint 7: Filler word ratio should be less than 5%
    filler_ratio = calculate_filler_word_ratio(text)
    if filler_ratio > 0.05:
        # violations += 1
        violations += (filler_ratio - 0.05)
    
    # Constraint 8: Single character word ratio (excluding 'a', 'I') should be less than 5%
    single_char_ratio = calculate_single_char_word_ratio(text)
    # if single_char_ratio > 0.05:
    #     violations += 1
    if single_char_ratio > 0.1:
        violations += (single_char_ratio - 0.1)
    
    # Constraint 9: Stop word ratio should be less than 75%
    # (For short text, allow slightly more stop words)
    stop_word_ratio = calculate_stop_word_ratio(text)
    if stop_word_ratio > 0.75:
        violations += (stop_word_ratio - 0.75) #* 5  # Heavy penalty above 75%
    
    # Constraint 10: Two-character word ratio should be less than 35%
    # (Catches "to to the is of" spam - stricter for short text)
    two_char_ratio = calculate_two_char_word_ratio(text)
    if two_char_ratio > 0.35:
        violations += (two_char_ratio - 0.35) #* 3
    
    # Constraint 11: Maximum word length should be <= 15 characters
    # (Very long words are usually gibberish - strict for short text)
    max_word_len = calculate_max_word_length(text)
    if max_word_len > 15:
        violations += (max_word_len - 15) #* 0.2  # Penalty for each char over 15
    
    # Constraint 12: Newline ratio should be < 3%
    # (For 100 tokens ≈ 500 chars, 3% = 15 newlines = reasonable paragraph breaks)
    # (Too many newlines = poor formatting or line-by-line generation)
    newline_ratio = calculate_newline_ratio(text)
    if newline_ratio > 0.03:
        violations += (newline_ratio - 0.03) #* 20  # Heavy penalty for excessive newlines
    
    # Constraint 13: No gibberish patterns
    # (Consecutive consonants, repeating patterns)
    if has_gibberish_pattern(text):
        violations += 1.0 #5.0  # Heavy penalty for gibberish
    
    return violations


def calc_combined_quality_violations_parallel(sequence_list, batch_size, device):
    """
    Calculate combined quality violation counts for a batch of sequences in parallel.
    
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
    for i in tqdm(range(batch_size), desc="Combined Quality"):
        text = sequence_list[i]
        violation_count = count_combined_quality_violations(text)
        violations[i] = violation_count
    
    return violations

def measure_combined_quality_violations_parallel(sequence_list, batch_size, device):
    """
    Calculate combined quality violation counts for a batch of sequences in parallel.
    
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
    for i in tqdm(range(batch_size), desc="Combined Quality"):
        text = sequence_list[i]
        violation_count = measure_combined_quality_violations(text)
        violations[i] = violation_count
    
    return violations

def measure_combined_quality_level_violations_parallel(sequence_list, batch_size, device):
    """
    Calculate combined quality violation counts for a batch of sequences in parallel.
    
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
    for i in tqdm(range(batch_size), desc="Combined Quality"):
        text = sequence_list[i]
        violation_count = measure_combined_quality_level_violations(text)
        violations[i] = violation_count
    
    return violations

def measure_combined_heuristic_violations_parallel(sequence_list, batch_size, device):
    """
    Calculate combined quality violation counts for a batch of sequences in parallel.
    
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
    for i in tqdm(range(batch_size), desc="Combined Quality"):
        text = sequence_list[i]
        violation_count = measure_combined_heuristic_violations(text)
        violations[i] = violation_count
    
    return violations