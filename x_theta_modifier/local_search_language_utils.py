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


def local_search_language_batch(best_tokens, x_theta_probs, distance_to_bounds_parallel, property_calcs_parallel, tokenizer, top_k_values_for_local_search, sampling_method='top_p', locally_typical_alpha=0.0, best_sequence_rank=1, prefix_lengths=None, device='cuda'):
    """
    Batch local search for language data using top-k values from x_theta probabilities.
    For each position, only try the top-k most probable tokens.
    
    This function processes ALL sequences in the batch together for efficiency.
    Uses parallel property calculation functions for speed.
    
    IMPORTANT: Follows discrete diffusion output processing:
    1. Decode token IDs → text (with special tokens)
    2. Clean text: remove special tokens
    3. Pass cleaned text to property calculators (they handle encoding internally)
    
    CRITICAL: Prefix Protection
    - If prefix_lengths is provided, prefix tokens will NEVER be modified
    - Only non-prefix positions are candidates for local search
    
    Args:
        best_tokens: Current best sequences (batch_size x seq_len) tensor of token IDs
        x_theta_probs: Probability distribution over vocab (batch_size x seq_len x vocab_size)
        distance_to_bounds_parallel: List of distance functions (parallel version)
        property_calcs_parallel: List of property calculation functions (parallel version)
        tokenizer: Tokenizer for decoding sequences
        top_k_values_for_local_search: Number of top-k values to try per position
        sampling_method: Selection method - 'top_p' (highest probability) or 'locally_typical' (closest to entropy)
        best_sequence_rank: Select the sequence with the Nth smallest distance (1=best, 2=second best, etc.)
        prefix_lengths: Optional tensor of shape (batch_size,) indicating prefix length for each sequence
        device: Device to use
    
    Returns:
        best_tokens: Best sequences found (batch_size x seq_len) tensor
    """
    batch_size, seq_len = best_tokens.shape
    
    # Create prefix mask: True for positions that should NOT be modified
    if prefix_lengths is not None:
        # Create a mask where prefix positions are True (protected)
        prefix_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for batch_idx, prefix_len in enumerate(prefix_lengths):
            if prefix_len > 0:
                prefix_mask[batch_idx, :prefix_len] = True
        print(f"  Local search: Protecting prefix tokens (lengths: {prefix_lengths.tolist()})")
    else:
        prefix_mask = None
    
    # NOTE: We calculate best_distances here to initialize batch_top_sequences later
    # This is needed for hierarchical comparison with neighbors
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
        
        # Apply prefix mask: exclude prefix positions from being modified
        if prefix_mask is not None:
            is_different = is_different & (~prefix_mask)
        
        # Get indices of all positions that differ (and are not in prefix)
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
    
    print(f"  Local search: Evaluating neighbors with hierarchical property filtering...")
    
    # HIERARCHICAL PROPERTY EVALUATION
    # Instead of computing all properties at once, compute them one by one in priority order
    # After each property, keep only neighbors with minimum distance for that property (per batch)
    # This significantly reduces computation for lower-priority properties
    
    num_properties = len(property_calcs_parallel)
    
    # Initialize distance tensor: will be filled incrementally
    # Shape: [total_neighbors, num_properties]
    neighbor_distances_tensor = torch.zeros((total_neighbors, num_properties), device=device)
    
    # Keep track of active neighbors (indices into neighbor_tokens_batch)
    active_neighbor_indices = list(range(total_neighbors))
    active_neighbor_metadata = list(all_neighbor_metadata)
    
    # Decode texts once (will be reused)
    neighbor_texts = tokenizer.batch_decode(neighbor_tokens_batch.cpu().numpy())
    neighbor_texts_cleaned = clean_text_samples(neighbor_texts)
    
    # Process each property in order (priority: first property is most important)
    for prop_idx in range(num_properties):
        if len(active_neighbor_indices) == 0:
            print(f"  Local search: No active neighbors remaining after property {prop_idx-1}")
            break
        
        print(f"  Local search: Property {prop_idx}/{num_properties} - Evaluating {len(active_neighbor_indices)} active neighbors...")
        
        # Get texts for active neighbors only
        active_texts = [neighbor_texts_cleaned[i] for i in active_neighbor_indices]
        
        # Calculate this property for active neighbors
        prop_values = property_calcs_parallel[prop_idx](active_texts, len(active_texts), device)
        
        # Apply distance function
        distances = distance_to_bounds_parallel[prop_idx](prop_values)
        
        # Store distances in the full tensor
        for local_idx, global_idx in enumerate(active_neighbor_indices):
            neighbor_distances_tensor[global_idx, prop_idx] = distances[local_idx]
        
        # If this is NOT the last property, filter neighbors by minimum distance (per batch)
        if prop_idx < num_properties - 1:
            # Group by batch_idx
            batch_neighbor_distances = {b_idx: [] for b_idx in range(batch_size)}
            for local_idx, global_idx in enumerate(active_neighbor_indices):
                b_idx = active_neighbor_metadata[local_idx][0]
                batch_neighbor_distances[b_idx].append((global_idx, distances[local_idx].item()))
            
            # Find minimum distance per batch and keep only those neighbors
            new_active_indices = []
            new_active_metadata = []
            
            for b_idx in range(batch_size):
                if not batch_neighbor_distances[b_idx]:
                    continue
                
                # Find minimum distance for this batch
                min_dist = min(dist for _, dist in batch_neighbor_distances[b_idx])
                
                # Keep neighbors with minimum distance
                for global_idx, dist in batch_neighbor_distances[b_idx]:
                    if dist == min_dist:
                        new_active_indices.append(global_idx)
                        # Find corresponding metadata
                        for i, idx in enumerate(active_neighbor_indices):
                            if idx == global_idx:
                                new_active_metadata.append(active_neighbor_metadata[i])
                                break
            
            filtered_count = len(active_neighbor_indices) - len(new_active_indices)
            active_neighbor_indices = new_active_indices
            active_neighbor_metadata = new_active_metadata
            
            print(f"  Local search: After property {prop_idx}, kept {len(active_neighbor_indices)} neighbors (filtered {filtered_count})")
    
    print(f"  Local search: Final {len(active_neighbor_indices)} neighbors after all property filtering")
    
    print(f"  Local search: Updating best sequences...")
    
    # Track top-k best sequences for each batch element
    # We'll store tuples of (distance_tensor, tokens_tensor) for each sequence
    # Initialize with the original best sequence
    batch_top_sequences = []
    for batch_idx in range(batch_size):
        # Each batch element starts with its current best sequence
        initial_entry = (best_distances_tensor[batch_idx].clone(), best_tokens[batch_idx].clone())
        batch_top_sequences.append([initial_entry])
    
    max_keep = max(best_sequence_rank, 10)
    # Update best sequences if neighbors are better
    # NOTE: Only iterate over ACTIVE neighbors (those that survived hierarchical filtering)
    for local_idx, global_idx in enumerate(active_neighbor_indices):
        batch_idx, pos, k_rank = active_neighbor_metadata[local_idx]
        neighbor_dist = neighbor_distances_tensor[global_idx]  # [num_properties]
        neighbor_tok = neighbor_tokens_batch[global_idx]  # [seq_len]
        
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
