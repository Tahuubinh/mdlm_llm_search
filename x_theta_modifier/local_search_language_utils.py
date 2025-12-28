"""
Language-specific local search utilities.
Optimized for batch processing with property models.
"""

import torch
from properties.property_util import compare_hierarchical


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


def local_search_language_batch(best_tokens, x_theta_probs, distance_to_bounds_parallel, property_calcs_parallel, tokenizer, top_k_values_for_local_search, device):
    """
    Batch local search for language data using top-k values from x_theta probabilities.
    For each position, only try the top-k most probable tokens.
    
    This function processes ALL sequences in the batch together for efficiency.
    Uses parallel property calculation functions for speed.
    
    IMPORTANT: Follows discrete diffusion output processing:
    1. Decode token IDs → text (with special tokens)
    2. Clean text: remove special tokens
    3. Encode cleaned text → new token IDs
    4. Calculate properties on cleaned token IDs
    
    Args:
        best_tokens: Current best sequences (batch_size x seq_len) tensor of token IDs
        x_theta_probs: Probability distribution over vocab (batch_size x seq_len x vocab_size)
        distance_to_bounds_parallel: List of distance functions (parallel version)
        property_calcs_parallel: List of property calculation functions (parallel version)
        tokenizer: Tokenizer for decoding sequences
        top_k_values_for_local_search: Number of top-k values to try per position
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
    
    # Get top-k token indices for each position from x_theta
    # x_theta_probs shape: (batch_size, seq_len, vocab_size)
    topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k_values_for_local_search, dim=-1)
    # topk_indices shape: (batch_size, seq_len, top_k_values_for_local_search)
    
    print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens in parallel...")
    
    # FULLY VECTORIZED neighbor generation across ALL ranks at once!
    # Generate neighbors for ALL ranks simultaneously
    all_neighbor_tokens = []
    all_neighbor_metadata = []
    
    for k_rank in range(1, top_k_values_for_local_search):
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
