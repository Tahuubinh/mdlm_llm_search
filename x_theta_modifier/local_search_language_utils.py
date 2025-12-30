"""
Language-specific local search utilities.
Optimized for batch processing with property models.
"""

import torch
from properties.property_util import compare_hierarchical


def select_top_k_tokens(x_theta_probs, top_k, method='top_p', alpha=0.0):
    """
    Select top-k token indices from probability distribution.
    
    Args:
        x_theta_probs: Probability distribution (batch_size, seq_len, vocab_size)
        top_k: Number of tokens to select
        method: Selection method - 'top_p' or 'locally_typical'
        alpha: Weight for probability bias in locally_typical (default: 0.0)
               alpha = 0.0: pure locally typical (entropy-based only)
               alpha > 0.0: bias toward high probability tokens
               alpha → ∞: approaches top-p behavior
    
    Returns:
        topk_indices: Tensor of shape (batch_size, seq_len, top_k) with selected token indices
    """
    if method == 'top_p':
        # Default: Select tokens with highest probabilities
        topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k, dim=-1)
        return topk_indices
    
    elif method == 'locally_typical':
        # Locally Typical Sampling with probability bias:
        # 1. Calculate entropy H of the distribution
        # 2. For each token, calculate distance: |(-log(p)) - H| - alpha * log(p)
        # 3. Select top-k tokens with smallest distance
        #
        # The alpha term biases selection toward high probability tokens:
        # - alpha = 0: pure locally typical (only entropy distance matters)
        # - alpha > 0: high prob tokens get smaller distance → more likely to be selected
        # - alpha → ∞: behaves like top-p (highest prob tokens always selected)
        
        # Calculate entropy H for each position: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_probs = torch.log(x_theta_probs + epsilon)
        entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Calculate -log(p) for each token
        neg_log_probs = -log_probs  # (batch_size, seq_len, vocab_size)
        
        # Calculate base distance: |(-log(p)) - H|
        distance_to_entropy = torch.abs(neg_log_probs - entropy)  # (batch_size, seq_len, vocab_size)
        
        # Apply probability bias: subtract alpha * log(p)
        # Since neg_log_probs = -log(p), we add alpha * neg_log_probs
        # This makes high probability tokens (small neg_log_probs) have smaller final distance
        if alpha > 0:
            distance_to_entropy = distance_to_entropy - alpha * log_probs
        
        # Select top-k tokens with SMALLEST distance (closest to entropy + probability bias)
        # Use topk with largest=False to get smallest values
        _, topk_indices = torch.topk(distance_to_entropy, k=top_k, dim=-1, largest=False)
        
        return topk_indices
    
    else:
        raise ValueError(f"Unknown sampling method: {method}. Choose 'top_p' or 'locally_typical'")


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


def local_search_language_batch(best_tokens, x_theta_probs, distance_to_bounds_parallel, property_calcs_parallel, tokenizer, top_k_values_for_local_search, sampling_method='top_p', locally_typical_alpha=0.0, device='cuda'):
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
    topk_indices = select_top_k_tokens(x_theta_probs, top_k_values_for_local_search, method=sampling_method, alpha=locally_typical_alpha)
    # topk_indices shape: (batch_size, seq_len, top_k_values_for_local_search)
    
    print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' sampling (alpha={locally_typical_alpha}) in parallel...")
    
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
    
    # Update best sequences if neighbors are better
    for i, (batch_idx, pos, k_rank) in enumerate(all_neighbor_metadata):
        neighbor_dist = neighbor_distances_tensor[i]  # [num_properties]
        best_dist = best_distances_tensor[batch_idx]  # [num_properties]
        
        # Convert to lists for compare_hierarchical
        neighbor_dist_list = neighbor_dist.cpu().tolist()
        best_dist_list = best_dist.cpu().tolist()
        
        # Check if neighbor is better
        if compare_hierarchical(neighbor_dist_list, best_dist_list) < 0:
            best_tokens[batch_idx] = neighbor_tokens_batch[i]
            best_distances_tensor[batch_idx] = neighbor_dist.clone()
    
    return best_tokens


if __name__ == "__main__":
    print("Language local search utilities for batch processing")
