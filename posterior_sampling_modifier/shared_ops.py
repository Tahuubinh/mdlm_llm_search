import torch

def gumbel_sample(x_theta, num_samples):
    gumbel_noise = -torch.log(-torch.log(torch.rand(
        (num_samples, *x_theta.shape),
        device=x_theta.device
    ) + 1e-10) + 1e-10)
    all_samples = (x_theta.unsqueeze(0).log() + gumbel_noise).argmax(dim=-1)
    return all_samples

def nucleus_sampling_filter(categorical_probs, nucleus_threshold):
    # Sort probabilities in descending order and calculate cumulative sum
    sorted_probs, sorted_indices = torch.sort(categorical_probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens where cumulative probability exceeds the nucleus threshold
    nucleus_mask = cumulative_probs > nucleus_threshold
    # Ensure at least one token is always included
    nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
    nucleus_mask[..., 0] = False

    # Zero out probabilities outside the nucleus
    sorted_probs[nucleus_mask] = 0.0

    # Scatter the sorted probabilities back to their original positions
    probs_nucleus = torch.zeros_like(categorical_probs)
    probs_nucleus.scatter_(-1, sorted_indices, sorted_probs)

    # Renormalize so that each distribution sums to 1
    categorical_probs = probs_nucleus / probs_nucleus.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return categorical_probs

def filter_top_k_with_mask(categorical_probs, xt, mask_index, top_k_categorical):
    mask_positions = (xt == mask_index)  # (B, L)

    # Save original for reference
    orig_probs = categorical_probs.clone()

    # For masked positions, zero out the mask index temporarily
    orig_probs[:, :, mask_index] = -float('inf')

    # Mask out values less than or equal to 0
    orig_probs[orig_probs <= 0] = -float('inf')

    # Find top-k values and their indices excluding the mask index
    topk_vals, topk_idx = torch.topk(orig_probs, top_k_categorical, dim=-1)

    # Get probabilities of mask token at those positions
    batch_idx, token_idx = mask_positions.nonzero(as_tuple=True)  # (num_masked,)
    mask_probs = categorical_probs[batch_idx, token_idx, mask_index]  # (num_masked,)

    # Initialize output tensor with zeros
    probs_topk = torch.zeros_like(categorical_probs)

    # Scatter the top-k values back to their original positions
    probs_topk.scatter_(-1, topk_idx, topk_vals)

    # Renormalize so that the sum of top-k non-mask tokens equals 1 - mask_probs
    probs_topk[batch_idx, token_idx] *= ((1.0 - mask_probs).unsqueeze(-1) / probs_topk[batch_idx, token_idx].sum(dim=-1, keepdim=True).clamp_min(1e-12))

    # Restore the mask_index probability
    probs_topk[batch_idx, token_idx, mask_index] = mask_probs

    # Update categorical_probs with the adjusted probabilities
    return probs_topk