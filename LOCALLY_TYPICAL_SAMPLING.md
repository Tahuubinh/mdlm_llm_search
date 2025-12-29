# Locally Typical Sampling for Local Search

## Overview

Locally Typical Sampling is an alternative token selection method for local search that helps avoid degenerate outputs (mode collapse) like repetitive text.

## Problem with Standard Top-P Sampling

Standard top-p (highest probability) sampling can lead to:
- **Mode collapse**: Repetitive patterns like "wewewewewe..."
- **Too obvious choices**: High probability tokens that lack diversity
- **Low perplexity but poor quality**: Repetitive text has low perplexity but is meaningless

## How Locally Typical Sampling Works

Instead of always selecting the highest probability tokens, Locally Typical Sampling:

1. **Calculate Entropy H**: Compute the entropy of the probability distribution at each position
   ```
   H = -Σ p(x) * log p(x)
   ```

2. **Measure Distance to Entropy**: For each token, calculate:
   ```
   distance = | -log p(token) - H |
   ```

3. **Select Closest Tokens**: Choose top-k tokens with **smallest distance** to entropy

### Intuition

- Tokens with **high probability** (low -log p) may be too obvious → mode collapse
- Tokens with **low probability** (high -log p) may be too random → incoherent
- Tokens **closest to entropy** balance predictability and diversity → better quality

## Usage

### Command Line

```bash
python inference_search.py \
  --local_search_sampling_method locally_typical \
  --top_k_values_for_local_search 5
```

### Options

- `--local_search_sampling_method`: 
  - `top_p` (default): Select highest probability tokens
  - `locally_typical`: Select tokens closest to entropy

### Example Comparison

**Top-P (Default)**:
```bash
python inference_search.py \
  --x_theta_type local_search_language \
  --local_search_sampling_method top_p \
  --top_k_values_for_local_search 5
```

**Locally Typical**:
```bash
python inference_search.py \
  --x_theta_type local_search_language \
  --local_search_sampling_method locally_typical \
  --top_k_values_for_local_search 5
```

## Expected Results

### Top-P Issues
- May produce repetitive patterns: "we-we-we-we..."
- Low perplexity (2.5) but poor quality
- Mode collapse in local search

### Locally Typical Benefits
- More diverse token selection
- Avoids extreme repetition
- Better balance between fluency and diversity
- Higher quality text with reasonable perplexity

## Configuration File

In `configs/x_theta_modifier/config.yaml`:

```yaml
method: local_search_language
top_k_values_for_local_search: 5
local_search_sampling_method: locally_typical  # or 'top_p'
```

## Technical Details

### Algorithm

```python
# For each position in sequence:
# 1. Compute entropy
H = -sum(p * log(p))

# 2. Compute distance for all tokens
distance[i] = |(-log p[i]) - H|

# 3. Select top-k with smallest distance
selected_tokens = argsort(distance)[:k]
```

### Complexity
- Same as top-p: O(V log k) where V is vocab size, k is top_k
- Negligible overhead compared to property calculations

## Troubleshooting

### High Repetition
- Switch from `top_p` to `locally_typical`
- Increase `top_k_values_for_local_search` for more diversity

### Too Random Output
- Switch from `locally_typical` to `top_p`
- Decrease `top_k_values_for_local_search`

### No Effect
- Locally typical sampling only affects **neighbor generation** in local search
- It doesn't affect initial BoN sampling or diffusion sampling

## References

- Meister et al. (2022): "Locally Typical Sampling"
- Helps balance between likelihood and diversity in text generation
