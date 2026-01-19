import torch
from x_theta_modifier.local_search_utils import *

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

def gumbel_sample(x_theta, num_samples):
    gumbel_noise = -torch.log(-torch.log(torch.rand(
        (num_samples, *x_theta.shape),
        device=x_theta.device
    ) + 1e-10) + 1e-10)
    all_samples = (x_theta.unsqueeze(0).log() + gumbel_noise).argmax(dim=-1)
    return all_samples

def gumbel_sample_sequential(x_theta, num_samples):
    """
    Sequential version of gumbel_sample that creates samples one at a time
    to reduce memory usage. Instead of creating all samples at once, this
    creates one sample for all batches at a time, repeating num_samples times.
    
    Args:
        x_theta: Logits tensor of shape [batch_size, seq_len, vocab_size]
        num_samples: Number of samples to generate
    
    Returns:
        Tensor of shape [num_samples, batch_size, seq_len] containing sampled token ids
    """
    batch_size, seq_len, vocab_size = x_theta.shape
    device = x_theta.device
    
    # Pre-compute log probabilities once
    log_x_theta = x_theta.log()
    
    # Pre-allocate output tensor
    all_samples = torch.empty((num_samples, batch_size, seq_len), dtype=torch.long, device=device)
    
    # Generate samples one at a time
    for i in range(num_samples):
        # Generate Gumbel noise for one sample across all batches
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(x_theta) + 1e-10) + 1e-10)
        # Sample by taking argmax
        all_samples[i] = (log_x_theta + gumbel_noise).argmax(dim=-1)
    
    return all_samples

def keep_nonmask_values(all_samples, xt, mask_index):
    non_mask = (xt != mask_index).unsqueeze(0)  # Shape: [1, batch_size, length]
    all_samples = torch.where(non_mask, xt.unsqueeze(0).expand_as(all_samples), all_samples)
    return all_samples, non_mask

def compute_properties(smiles_list, property_calcs, distance_funcs):
    property_values = [
        calc(smiles_list, len(smiles_list), smiles_list[0].device)
        for calc in property_calcs
    ]
    distances = [
        distance_func(property_value)
        for distance_func, property_value in zip(distance_funcs, property_values)
    ]
    return distances

def compute_combined_distances(token_ids, batch_size, num_x_theta_samples, property_calcs_parallel, distance_to_bounds_parallel, device, tokenizer):
    """
    Compute combined distances for token sequences.
    
    IMPORTANT: Follows discrete diffusion output processing:
    1. Decode token IDs → text (with special tokens)
    2. Clean text: remove special tokens
    3. Pass cleaned text to property calculators (they handle encoding internally)
    
    Args:
        token_ids: Token IDs tensor [batch_size * num_x_theta_samples, seq_len]
        batch_size: Batch size
        num_x_theta_samples: Number of samples per batch element
        property_calcs_parallel: List of property calculation functions
        distance_to_bounds_parallel: List of distance functions
        device: Device to use
        tokenizer: Tokenizer for decoding
    
    Returns:
        Stacked distances tensor [batch_size, num_x_theta_samples, num_properties]
    """
    property_size = batch_size * num_x_theta_samples
    
    # Step 1: Decode token IDs to text
    text_samples = tokenizer.batch_decode(token_ids.cpu().numpy())
    
    # Step 2: Clean text by removing special tokens
    cleaned_texts = clean_text_samples(text_samples)
    
    # Step 3: Calculate properties using cleaned text
    property_values = [
        calc(cleaned_texts, property_size, device)
        for calc in property_calcs_parallel
    ]
    # Apply distance_to_bounds_parallel to property_values
    distances = [
        distance_func(property_value)
        for distance_func, property_value in zip(distance_to_bounds_parallel, property_values)
    ]

    # Reshape each distance in distances to [batch_size, self.num_x_theta_samples]
    reshaped_distances = [
        distance.reshape(batch_size, num_x_theta_samples)
        for distance in distances
    ]
    return torch.stack(reshaped_distances, dim=-1)  # Shape: [batch_size, self.num_x_theta_samples, num_properties]

def find_best_tokens(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, property_calcs_parallel, distance_to_bounds_parallel):
    # Reshape samples for score calculation: (batch_size * self.num_x_theta_samples_keepbest, seq_len)
    reshaped_samples = all_samples.transpose(0, 1).reshape(-1, seq_len)
    # Decode SMILES strings from token IDs
    smiles_list = tokenizer.batch_decode(reshaped_samples.cpu().numpy())
    
    num_properties = len(property_calcs_parallel)
    property_size = batch_size * num_x_theta_samples_keepbest
    
    # Initialize best tokens and distances for each batch element
    # best_distances[b] will store [dist1, dist2, ...] for batch element b
    best_distances = []
    best_indices = []
    
    for b in range(batch_size):
        # Get sequences for this batch element
        start_idx = b * num_x_theta_samples_keepbest
        end_idx = (b + 1) * num_x_theta_samples_keepbest
        batch_sequences = smiles_list[start_idx:end_idx]
        
        # Initialize with first sample as best
        best_idx = 0
        best_dist = None
        
        # Iterate through all samples for this batch element
        for sample_idx in range(num_x_theta_samples_keepbest):
            sequence = batch_sequences[sample_idx]
            
            # Compute properties and distances one by one with early termination
            current_distances = []
            is_better = False
            is_worse = False
            
            for prop_idx in range(num_properties):
                # Compute property and distance for this index
                # For parallel calcs, we pass a single-element list
                prop_value = property_calcs_parallel[prop_idx]([sequence], 1, device)
                distance = distance_to_bounds_parallel[prop_idx](prop_value)[0]  # Get scalar value
                current_distances.append(distance.item() if torch.is_tensor(distance) else distance)
                
                # If this is the first sample, just store and continue
                if best_dist is None:
                    continue
                
                # Compare with best distance at this index
                if current_distances[prop_idx] < best_dist[prop_idx]:
                    is_better = True
                    break  # This sample is better, no need to compute remaining properties for comparison
                elif current_distances[prop_idx] > best_dist[prop_idx]:
                    is_worse = True
                    break  # This sample is worse, no need to compute remaining properties
                # If equal, continue to next property
            
            # If this is the first sample, set as best
            if best_dist is None:
                best_dist = current_distances
                best_idx = sample_idx
            # If we found a better sample, compute remaining properties and update
            elif is_better:
                # Compute remaining properties for completeness
                for prop_idx in range(len(current_distances), num_properties):
                    prop_value = property_calcs_parallel[prop_idx]([sequence], 1, device)
                    distance = distance_to_bounds_parallel[prop_idx](prop_value)[0]
                    current_distances.append(distance.item() if torch.is_tensor(distance) else distance)
                
                best_dist = current_distances
                best_idx = sample_idx
        
        best_indices.append(best_idx)
        best_distances.append(best_dist)
    
    # Convert best_indices to tensor and gather best tokens
    best_indices_tensor = torch.tensor(best_indices, device=device)
    best_tokens = all_samples[best_indices_tensor, torch.arange(batch_size, device=device)]
    return best_tokens

def find_best_tokens_sort(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, property_calcs_parallel, distance_to_bounds_parallel):
    # Reshape samples for score calculation: (batch_size * self.num_x_theta_samples_keepbest, seq_len)
    reshaped_samples = all_samples.transpose(0, 1).reshape(-1, seq_len)
    
    # Calculate combined distances (with decode → clean → encode)
    combined_distances = compute_combined_distances(reshaped_samples, batch_size, num_x_theta_samples_keepbest, property_calcs_parallel, distance_to_bounds_parallel, device, tokenizer)  # Shape: [batch_size, self.num_x_theta_samples_keepbest, num_properties]

    # combined_distances: [B, N, P] (batch_size, num_samples, num_properties)
    B, N, P = combined_distances.shape

    # Initialize indices for each sample in the batch
    sorted_indices = torch.arange(N, device=combined_distances.device).repeat(B, 1)

    # Iterate over columns from right to left for lexicographical ordering
    for col in reversed(range(P)):
        # Sort samples by current column with stable sort to preserve previous order
        _, perm = combined_distances[:, :, col].sort(dim=1, stable=True)
        
        # Update sorted indices according to the current permutation
        sorted_indices = sorted_indices.gather(1, perm)

    # Select the best sample (first in sorted order) for each batch element
    best_indices = sorted_indices[:, 0]  # Shape: [B]
    # Use advanced indexing to get the corresponding tokens
    best_tokens = all_samples[best_indices, torch.arange(B, device=device)]
    return best_tokens

def modify_x_theta_on_valid_best_tokens(best_tokens, tokenizer, x_theta, batch_size, device):
    decoded_best_tokens = tokenizer.batch_decode(best_tokens.cpu().numpy(), skip_special_tokens=True)
    validity_scores = calc_validity_parallel(decoded_best_tokens, batch_size, device)
    
    # Create a new tensor for x_theta with the same shape
    new_x_theta = torch.zeros_like(x_theta)

    # Create a mask for validity scores equal to 1
    validity_mask = validity_scores.unsqueeze(1).expand(-1, best_tokens.shape[1])  # Shape: [batch_size, seq_len]

    # Ensure the data type of new_x_theta matches validity_mask
    validity_mask = validity_mask.to(new_x_theta.dtype)

    # Use scatter_add to set weights in parallel
    new_x_theta.scatter_add_(
        2,
        best_tokens.unsqueeze(-1),
        validity_mask.unsqueeze(-1)
    )

    # If validity score is smaller than 0.5, copy the corresponding categorical distribution from x_theta
    invalid_mask = (validity_scores < 0.5).unsqueeze(1).unsqueeze(2).expand(-1, new_x_theta.shape[1], new_x_theta.shape[2])
    new_x_theta[invalid_mask] = x_theta[invalid_mask]
    return new_x_theta, validity_mask

def modify_x_theta_no_condition(best_tokens, tokenizer, x_theta, batch_size, device):
    # Create a new tensor for x_theta with the same shape
    new_x_theta = torch.zeros_like(x_theta)

    # Set the weight of the corresponding token to 1 for each categorical distribution
    new_x_theta.scatter_(2, best_tokens.unsqueeze(-1), 1.0)

    return new_x_theta, None
