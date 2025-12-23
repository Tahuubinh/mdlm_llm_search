"""
Text cleaning utilities for generated natural language
"""

def remove_special_tokens(text):
    """Remove special tokens from text"""
    special_tokens = ['<|endoftext|>', '[PAD]', '[MASK]', '<unk>', '</s>', '<s>']
    for token in special_tokens:
        text = text.replace(token, '')
    
    # Clean whitespace
    text = ' '.join(text.split())
    return text.strip()


def truncate_at_eos(text, eos_tokens=None):
    """Truncate text at first EOS token"""
    if eos_tokens is None:
        eos_tokens = ['<|endoftext|>', '[EOS]', '</s>']
    
    for eos in eos_tokens:
        if eos in text:
            text = text[:text.index(eos)]
    return text.strip()


def remove_repetitions(text, max_n_gram=10):
    """Remove repeating n-grams that appear consecutively"""
    words = text.split()
    
    for n in range(max_n_gram, 2, -1):
        i = 0
        while i < len(words) - 2*n:
            ngram = tuple(words[i:i+n])
            next_ngram = tuple(words[i+n:i+2*n])
            
            if ngram == next_ngram:
                # Found repetition, remove it and restart
                words = words[:i+n] + words[i+2*n:]
                i = 0  # Restart from beginning
            else:
                i += 1
    
    return ' '.join(words)


def truncate_at_sentence_boundary(text, max_words=None):
    """
    Truncate at last complete sentence, optionally within max_words limit
    
    Args:
        text: Input text
        max_words: Maximum number of words (if None, no limit)
    
    Returns:
        Truncated text ending at a sentence boundary
    """
    # If max_words specified, first truncate to that limit
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
    
    # Find last sentence boundary
    ending_punctuations = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    last_pos = -1
    
    for punct in ending_punctuations:
        pos = text.rfind(punct)
        if pos > last_pos:
            last_pos = pos
    
    # If no sentence boundary found, check for period at end
    if last_pos == -1:
        for punct in ['.', '!', '?']:
            if text.endswith(punct):
                return text.strip()
    
    if last_pos > 0:
        # Include the punctuation mark
        return text[:last_pos + 1].strip()
    
    # If no sentence boundary found, return as is
    return text.strip()


def remove_incomplete_last_sentence(text):
    """Remove the last sentence if it doesn't end with punctuation"""
    # Check if text ends with sentence-ending punctuation
    if text and not any(text.rstrip().endswith(p) for p in ['.', '!', '?']):
        # Find last complete sentence
        last_period = max(
            text.rfind('. '),
            text.rfind('! '),
            text.rfind('? ')
        )
        
        if last_period > 0:
            return text[:last_period + 1].strip()
    
    return text.strip()


def clean_generated_text(text, prefix_text='', max_words=100, remove_repetition=True):
    """
    Complete cleaning pipeline for generated text
    
    Args:
        text: Generated text (may include prefix)
        prefix_text: The prefix text to remove (if any)
        max_words: Maximum number of words in output (approx)
        remove_repetition: Whether to remove repetitive n-grams
    
    Returns:
        Cleaned text
    """
    # 1. Remove prefix if present
    if prefix_text and text.startswith(prefix_text):
        text = text[len(prefix_text):].strip()
    
    # 2. Remove special tokens
    text = remove_special_tokens(text)
    
    # 3. Truncate at EOS token
    text = truncate_at_eos(text)
    
    # 4. Remove repetitions (optional, can be slow)
    if remove_repetition:
        text = remove_repetitions(text, max_n_gram=10)
    
    # 5. Truncate at sentence boundary (with max_words limit)
    text = truncate_at_sentence_boundary(text, max_words=max_words)
    
    # 6. Remove incomplete last sentence if any
    text = remove_incomplete_last_sentence(text)
    
    # 7. Final cleanup
    text = text.strip()
    
    return text


if __name__ == "__main__":
    # Example usage
    test_text = "This is a test. This is a test. This is a test. And some more text that goes on and on without stopping properly because"
    
    print("Original text:")
    print(test_text)
    print()
    
    print("After removing repetitions:")
    cleaned = remove_repetitions(test_text)
    print(cleaned)
    print()
    
    print("After full cleaning pipeline:")
    final = clean_generated_text(test_text, max_words=50, remove_repetition=True)
    print(final)
