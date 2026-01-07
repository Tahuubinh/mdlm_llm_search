"""
Language-specific local search utilities.
Optimized for batch processing with property models.
"""

import torch
from collections import Counter
from properties.property_util import compare_hierarchical


def select_top_k_tokens(x_theta_probs, top_k, method='top_p', alpha=0.0):
    """
    Select top-k token indices from probability distribution.
    
    Args:
        x_theta_probs: Probability distribution (batch_size, seq_len, vocab_size)
        top_k: Number of tokens to select
        method: Selection method - 'top_p', 'locally_typical', or 'locally_typical_distance'
        alpha: Bias parameter (interpretation depends on method):
               - For 'locally_typical': additive bias weight (0.0 = pure, >0 = bias toward high prob)
               - For 'locally_typical_distance': entropy scaling factor (1.0 = pure, <1.0 = bias toward high prob)
    
    Returns:
        topk_indices: Tensor of shape (batch_size, seq_len, top_k) with selected token indices
    """
    if method == 'top_p':
        # Default: Select tokens with highest probabilities
        topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k, dim=-1)
        return topk_indices
    
    elif method == 'locally_typical':
        # Locally Typical Sampling with probability bias (additive):
        # Formula: distance = |-log(p) - H| - alpha * log(p)
        # 
        # The alpha term biases selection toward high probability tokens:
        # - alpha = 0: pure locally typical (only entropy distance matters)
        # - alpha > 0: high prob tokens get smaller distance → more likely to be selected
        # - alpha → ∞: behaves like top-p (highest prob tokens always selected)
        
        # Calculate entropy H for each position: H = -sum(p * log(p))
        epsilon = 1e-10
        log_probs = torch.log(x_theta_probs + epsilon)
        entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Calculate -log(p) for each token
        neg_log_probs = -log_probs  # (batch_size, seq_len, vocab_size)
        
        # Calculate base distance: |-log(p) - H|
        distance_to_entropy = torch.abs(neg_log_probs - entropy)  # (batch_size, seq_len, vocab_size)
        
        # Apply probability bias: subtract alpha * log(p)
        if alpha > 0:
            distance_to_entropy = distance_to_entropy - alpha * log_probs
        
        # Select top-k tokens with SMALLEST distance
        _, topk_indices = torch.topk(distance_to_entropy, k=top_k, dim=-1, largest=False)
        
        return topk_indices
    
    elif method == 'locally_typical_distance':
        # Locally Typical Sampling with entropy scaling (multiplicative):
        # Formula: distance = |-log(p) - alpha * H|
        #
        # The alpha parameter scales the target information (using alpha instead of tau for consistency):
        # - alpha = 1.0: pure locally typical (target = H, original algorithm)
        # - alpha < 1.0: scale entropy down → target moves toward 0 → prefer high prob tokens
        # - alpha → 0.0: target → 0 → equivalent to minimizing -log(p) → top-p behavior
        #
        # This is a different parameterization from 'locally_typical':
        # - 'locally_typical' uses additive bias: distance - alpha*log(p)
        # - 'locally_typical_distance' uses multiplicative scaling: |-log(p) - alpha*H|
        
        # Calculate entropy H for each position: H = -sum(p * log(p))
        epsilon = 1e-10
        log_probs = torch.log(x_theta_probs + epsilon)
        entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Calculate -log(p) for each token
        neg_log_probs = -log_probs  # (batch_size, seq_len, vocab_size)
        
        # Calculate distance with scaled entropy: |-log(p) - alpha * H|
        # When alpha = 1.0: standard locally typical
        # When alpha → 0.0: target approaches 0, which means selecting high prob tokens (low -log(p))
        scaled_entropy = alpha * entropy
        distance_to_scaled_entropy = torch.abs(neg_log_probs - scaled_entropy)  # (batch_size, seq_len, vocab_size)
        
        # Select top-k tokens with SMALLEST distance
        _, topk_indices = torch.topk(distance_to_scaled_entropy, k=top_k, dim=-1, largest=False)
        
        return topk_indices
    
    else:
        raise ValueError(f"Unknown sampling method: {method}. Choose 'top_p', 'locally_typical', or 'locally_typical_distance'")


def clean_text_samples(text_samples):
    """
    Clean text samples by removing special tokens.
    This matches the cleaning logic in inference_utils.py.
    
    Args:
        text_samples: List of text strings
    
    Returns:
        List of cleaned text strings
    """
    cleaned = []
    for sample in text_samples:
        # Remove all special tokens (matching inference_utils.py logic)
        cleaned_text = sample.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<mask>', '').replace('<unk>', '').replace('<cls>', '').replace('<sep>', '').replace('<reserved>', '').strip()
        cleaned.append(cleaned_text)
    return cleaned


def calculate_ngram_char_fraction(text, n):
    """
    Calculate the fraction of characters covered by the most common n-gram.
    
    Uses index set to avoid double-counting overlapping n-grams.
    This ensures the result is always in [0, 1].
    
    Args:
        text: Input text string
        n: N-gram size
    
    Returns:
        Fraction of character positions covered by the most common n-gram
    """
    if len(text) < n:
        return 0.0
    
    # Generate all character n-grams
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    # Count n-grams
    ngram_counts = Counter(ngrams)
    
    # Get the most common n-gram
    most_common_ngram, _ = ngram_counts.most_common(1)[0]
    
    # Find all character indices covered by this n-gram
    covered_indices = set()
    for i in range(len(text) - n + 1):
        if text[i:i+n] == most_common_ngram:
            # Add all indices that this n-gram covers
            for j in range(i, i + n):
                covered_indices.add(j)
    
    # Calculate fraction of covered positions
    fraction = len(covered_indices) / len(text)
    
    return fraction


def calculate_duplicate_ngram_char_fraction(text, n):
    """
    Calculate the fraction of characters in duplicate n-grams (appearing more than once).
    
    Uses index set to avoid double-counting overlapping n-grams.
    This ensures the result is always in [0, 1].
    
    Args:
        text: Input text string
        n: N-gram size
    
    Returns:
        Fraction of character positions covered by duplicate n-grams
    """
    if len(text) < n:
        return 0.0
    
    # Generate all character n-grams
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    # Count n-grams
    ngram_counts = Counter(ngrams)
    
    # Get n-grams that appear more than once
    duplicate_ngrams = {ngram for ngram, count in ngram_counts.items() if count > 1}
    
    if not duplicate_ngrams:
        return 0.0
    
    # Find all character indices covered by duplicate n-grams
    covered_indices = set()
    for i in range(len(text) - n + 1):
        if text[i:i+n] in duplicate_ngrams:
            # Add all indices that this n-gram covers
            for j in range(i, i + n):
                covered_indices.add(j)
    
    # Calculate fraction of covered positions
    fraction = len(covered_indices) / len(text)
    
    return fraction


def count_repetition_violations(text):
    """
    Count how many repetition criteria the text violates.
    
    Returns the number of violated criteria (0 = perfect, 9 = violates all).
    
    Criteria:
    - Top 2-gram character fraction < 0.20
    - Top 3-gram character fraction < 0.18
    - Top 4-gram character fraction < 0.16
    - Duplicate 5-gram character fraction < 0.15
    - Duplicate 6-gram character fraction < 0.14
    - Duplicate 7-gram character fraction < 0.13
    - Duplicate 8-gram character fraction < 0.12
    - Duplicate 9-gram character fraction < 0.11
    - Duplicate 10-gram character fraction < 0.10
    """
    violations = 0
    
    # Check top n-gram criteria (2-4)
    top_ngram_thresholds = {
        2: 0.20,
        3: 0.18,
        4: 0.16
    }
    
    for n, threshold in top_ngram_thresholds.items():
        fraction = calculate_ngram_char_fraction(text, n)
        if fraction >= threshold:
            violations += 1
    
    # Check duplicate n-gram criteria (5-10)
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
        if fraction >= threshold:
            violations += 1
    
    return violations


def local_search_language_batch(best_tokens, x_theta_probs, distance_to_bounds_parallel, property_calcs_parallel, tokenizer, top_k_values_for_local_search, sampling_method='top_p', locally_typical_alpha=0.0, best_sequence_rank=1, filter_repetitions=False, device='cuda'):
    """
    Batch local search for language data using top-k values from x_theta probabilities.
    For each position, only try the top-k most probable tokens.
    
    This function processes ALL sequences in the batch together for efficiency.
    Uses parallel property calculation functions for speed.
    
    IMPORTANT: Follows discrete diffusion output processing:
    1. Decode token IDs → text (with special tokens)
    2. Clean text: remove special tokens
    3. Pass cleaned text to property calculators (they handle encoding internally)
    
    Args:
        best_tokens: Current best sequences (batch_size x seq_len) tensor of token IDs
        x_theta_probs: Probability distribution over vocab (batch_size x seq_len x vocab_size)
        distance_to_bounds_parallel: List of distance functions (parallel version)
        property_calcs_parallel: List of property calculation functions (parallel version)
        tokenizer: Tokenizer for decoding sequences
        top_k_values_for_local_search: Number of top-k values to try per position
        sampling_method: Selection method - 'top_p' (highest probability) or 'locally_typical' (closest to entropy)
        best_sequence_rank: Select the sequence with the Nth smallest distance (1=best, 2=second best, etc.)
        filter_repetitions: If True, filter out neighbors that violate repetition criteria
        device: Device to use
    
    Returns:
        best_tokens: Best sequences found (batch_size x seq_len) tensor
    """
    batch_size, seq_len = best_tokens.shape
    
    # Step 1: Decode best_tokens to text
    best_texts = tokenizer.batch_decode(best_tokens.cpu().numpy())
    
    # Step 2: Clean text by removing special tokens
    best_texts_cleaned = clean_text_samples(best_texts)
    
    # Step 3: Calculate properties using cleaned text (let property calculators handle encoding)
    best_prop_values = [
        calc(best_texts_cleaned, batch_size, device)
        for calc in property_calcs_parallel
    ]
    
    # Apply distance functions to get distances
    best_distances = [
        dist_func(prop_value)
        for dist_func, prop_value in zip(distance_to_bounds_parallel, best_prop_values)
    ]
    # Stack into [batch_size, num_properties]
    best_distances_tensor = torch.stack(best_distances, dim=-1)
    
    # Get top-k token indices for each position from x_theta using specified sampling method
    # x_theta_probs shape: (batch_size, seq_len, vocab_size)
    topk_indices = select_top_k_tokens(
        x_theta_probs, 
        top_k_values_for_local_search, 
        method=sampling_method, 
        alpha=locally_typical_alpha
    )
    # topk_indices shape: (batch_size, seq_len, top_k_values_for_local_search)
    
    # Print appropriate message based on sampling method
    if sampling_method == 'locally_typical':
        print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' (additive alpha={locally_typical_alpha}) in parallel...")
    elif sampling_method == 'locally_typical_distance':
        print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' (scaling alpha={locally_typical_alpha}) in parallel...")
    else:
        print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' in parallel...")
    
    # FULLY VECTORIZED neighbor generation across ALL ranks at once!
    # Generate neighbors for ALL ranks simultaneously
    all_neighbor_tokens = []
    all_neighbor_metadata = []
    
    for k_rank in range(0, top_k_values_for_local_search):
        # Extract candidate tokens for this rank: [batch_size, seq_len]
        candidate_tokens = topk_indices[:, :, k_rank]  # Shape: [batch_size, seq_len]
        
        # Create a mask for positions where candidate != current token
        is_different = candidate_tokens != best_tokens  # Shape: [batch_size, seq_len]
        
        # Get indices of all positions that differ
        batch_indices, pos_indices = torch.where(is_different)
        
        if len(batch_indices) == 0:
            continue
        
        # Create neighbors for this rank
        neighbor_tokens_rank = best_tokens[batch_indices].clone()
        neighbor_candidate_tokens = candidate_tokens[batch_indices, pos_indices]
        neighbor_tokens_rank[torch.arange(len(batch_indices)), pos_indices] = neighbor_candidate_tokens
        
        # Store neighbors and metadata
        all_neighbor_tokens.append(neighbor_tokens_rank)
        
        # Create metadata: (batch_idx, pos, rank)
        for b_idx, p_idx in zip(batch_indices.cpu().tolist(), pos_indices.cpu().tolist()):
            all_neighbor_metadata.append((b_idx, p_idx, k_rank))
    
    if len(all_neighbor_tokens) == 0:
        print(f"  Local search: No valid neighbors found, returning original tokens")
        return best_tokens
    
    # Concatenate all neighbors into one big batch
    neighbor_tokens_batch = torch.cat(all_neighbor_tokens, dim=0)
    
    total_neighbors = neighbor_tokens_batch.shape[0]
    print(f"  Local search: Generated {total_neighbors} unique neighbors across all ranks")
    
    # Filter repetitive neighbors if requested
    if filter_repetitions:
        print(f"  Local search: Filtering repetitive neighbors...")
        # Decode neighbors to check repetition
        neighbor_texts_for_filter = tokenizer.batch_decode(neighbor_tokens_batch.cpu().numpy())
        neighbor_texts_cleaned_for_filter = clean_text_samples(neighbor_texts_for_filter)
        
        # Count violations for each neighbor
        neighbor_violations = [count_repetition_violations(text) for text in neighbor_texts_cleaned_for_filter]
        
        # Group violations by batch_idx (each sample should filter independently)
        batch_neighbor_violations = {b_idx: [] for b_idx in range(batch_size)}
        for i, (b_idx, _, _) in enumerate(all_neighbor_metadata):
            batch_neighbor_violations[b_idx].append((i, neighbor_violations[i]))
        
        # For each batch element, find its own minimum violations and keep only those neighbors
        valid_indices = []
        batch_min_violations = {}
        
        for b_idx in range(batch_size):
            if not batch_neighbor_violations[b_idx]:
                continue
            
            # Find minimum violations for THIS batch element only
            min_violations_for_batch = min(v for _, v in batch_neighbor_violations[b_idx])
            batch_min_violations[b_idx] = min_violations_for_batch
            
            # Keep neighbors with minimum violations for this batch element
            for neighbor_idx, violations in batch_neighbor_violations[b_idx]:
                if violations == min_violations_for_batch:
                    valid_indices.append(neighbor_idx)
        
        # Also check original best_tokens violations for comparison
        best_texts_for_filter = tokenizer.batch_decode(best_tokens.cpu().numpy())
        best_texts_cleaned_for_filter = clean_text_samples(best_texts_for_filter)
        best_violations = [count_repetition_violations(text) for text in best_texts_cleaned_for_filter]
        
        # Print statistics per batch
        print(f"  Local search: Violation statistics per sample:")
        for b_idx in range(batch_size):
            min_v = batch_min_violations.get(b_idx, 'N/A')
            orig_v = best_violations[b_idx] if b_idx < len(best_violations) else 'N/A'
            num_kept = sum(1 for i in valid_indices if all_neighbor_metadata[i][0] == b_idx)
            print(f"    Sample {b_idx}: Min violations in neighbors={min_v}, Original violations={orig_v}, Kept {num_kept} neighbors")
        
        if len(valid_indices) == 0:
            print(f"  Local search: No neighbors found (should not happen), returning original tokens")
            return best_tokens
        
        # Filter neighbors and metadata to keep only those with minimum violations (per batch)
        neighbor_tokens_batch = neighbor_tokens_batch[valid_indices]
        all_neighbor_metadata = [all_neighbor_metadata[i] for i in valid_indices]
        
        filtered_count = total_neighbors - len(valid_indices)
        total_neighbors = len(valid_indices)
        print(f"  Local search: Total kept {total_neighbors} neighbors, filtered out {filtered_count} neighbors")
    
    print(f"  Local search: Evaluating all {total_neighbors} neighbors in parallel...")
    
    # Step 1: Decode neighbor_tokens_batch to text
    neighbor_texts = tokenizer.batch_decode(neighbor_tokens_batch.cpu().numpy())
    
    # Step 2: Clean text by removing special tokens
    neighbor_texts_cleaned = clean_text_samples(neighbor_texts)
    
    # Step 3: Calculate properties using cleaned text (let property calculators handle encoding)
    neighbor_prop_values = [
        calc(neighbor_texts_cleaned, total_neighbors, device)
        for calc in property_calcs_parallel
    ]
    
    # Apply distance functions
    neighbor_distances = [
        dist_func(prop_value)
        for dist_func, prop_value in zip(distance_to_bounds_parallel, neighbor_prop_values)
    ]
    
    # Stack into [total_neighbors, num_properties]
    neighbor_distances_tensor = torch.stack(neighbor_distances, dim=-1)
    
    print(f"  Local search: Updating best sequences...")
    
    # Track top-k best sequences for each batch element
    # We'll store tuples of (distance_tensor, tokens_tensor) for each sequence
    # Initialize with the original best sequence ONLY if its violations are acceptable
    batch_top_sequences = []
    for batch_idx in range(batch_size):
        # Only include original best_tokens if it has violations <= min violations of its neighbors
        # This ensures that if neighbors have better (lower) violations, original will be replaced
        if filter_repetitions and batch_idx in batch_min_violations:
            min_v = batch_min_violations[batch_idx]
            orig_v = best_violations[batch_idx]
            if orig_v <= min_v:
                # Original is competitive - include it in comparison
                initial_entry = (best_distances_tensor[batch_idx].clone(), best_tokens[batch_idx].clone())
                batch_top_sequences.append([initial_entry])
                print(f"    Sample {batch_idx}: Original (violations={orig_v}) included in comparison (min_v={min_v})")
            else:
                # Original has more violations than best neighbors - exclude it, forcing replacement
                batch_top_sequences.append([])
                print(f"    Sample {batch_idx}: Original (violations={orig_v}) EXCLUDED from comparison (min_v={min_v}), will be replaced")
        else:
            # No filtering, include original
            initial_entry = (best_distances_tensor[batch_idx].clone(), best_tokens[batch_idx].clone())
            batch_top_sequences.append([initial_entry])
    
    max_keep = max(best_sequence_rank, 10)
    # Update best sequences if neighbors are better
    for i, (batch_idx, pos, k_rank) in enumerate(all_neighbor_metadata):
        neighbor_dist = neighbor_distances_tensor[i]  # [num_properties]
        neighbor_tok = neighbor_tokens_batch[i]  # [seq_len]
        
        # Check if this neighbor should be added to top sequences
        top_list = batch_top_sequences[batch_idx]
        
        # Find insertion position using hierarchical comparison
        insert_idx = len(top_list)  # Default: append at end
        neighbor_dist_list = neighbor_dist.cpu().tolist()
        
        for idx, (existing_dist, _) in enumerate(top_list):
            existing_dist_list = existing_dist.cpu().tolist()
            if compare_hierarchical(neighbor_dist_list, existing_dist_list) < 0:
                insert_idx = idx
                break
        
        # Insert neighbor at the appropriate position
        # Keep only top max(best_sequence_rank, 10) to avoid memory issues
        if insert_idx < max_keep:
            top_list.insert(insert_idx, (neighbor_dist.clone(), neighbor_tok.clone()))
            # Trim list to max_keep
            if len(top_list) > max_keep:
                top_list.pop()
    
    # Select the sequence at the specified rank for each batch element
    for batch_idx in range(batch_size):
        top_list = batch_top_sequences[batch_idx]
        
        if len(top_list) == 0:
            # This should not happen as we always have neighbors, but safety check
            print(f"    WARNING: Sample {batch_idx} has no candidates, keeping original")
            continue
        
        # Adjust for 0-indexing (rank 1 = index 0)
        rank_idx = best_sequence_rank - 1
        
        if rank_idx < len(top_list):
            # Use the sequence at the specified rank
            best_distances_tensor[batch_idx] = top_list[rank_idx][0]
            best_tokens[batch_idx] = top_list[rank_idx][1]
        else:
            # If we don't have enough sequences, use the last one available
            best_distances_tensor[batch_idx] = top_list[-1][0]
            best_tokens[batch_idx] = top_list[-1][1]
    
    if best_sequence_rank > 1:
        print(f"  Local search: Selected sequences with rank {best_sequence_rank} (distance-wise)")
    
    return best_tokens


if __name__ == "__main__":
    print("Language local search utilities for batch processing")
