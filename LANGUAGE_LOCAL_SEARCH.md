# Language Local Search for MDLM

## Overview

This implementation provides an efficient local search method specifically designed for language data. Instead of exhaustively trying all vocabulary tokens at each position (which would be ~50K × 100 positions = 5M evaluations), it uses the probability distribution from `x_theta` to guide the search.

## Algorithm

### Key Idea
For language data, the model's `x_theta` output contains valuable information about which tokens are most likely at each position. We use this to intelligently select candidate tokens for local search.

### Process

1. **Start with best sequence** from BoN sampling
2. **For each rank k** in [1, 2, ..., `top_k_values_for_local_search`]:
   - For each position p in sequence:
     - Get the k-th most probable token from `x_theta[p]`
     - Create neighbor by replacing position p with this token
     - Evaluate properties (toxicity, perplexity, etc.)
     - Keep if better than current best
3. **Return** the best sequence found

### Complexity
- **Old approach (molecules)**: Try all valid tokens at each position
  - For language: ~50,000 vocab × 100 positions = **5,000,000 evaluations**
- **New approach**: Only try top-k tokens from x_theta
  - With k=10: 10 × 100 = **1,000 evaluations** (5000× speedup!)
  - With k=5: 5 × 100 = **500 evaluations** (10000× speedup!)

## Usage

### Command Line Arguments

```bash
python inference_search.py \
    --data openwebtext-split \
    --x_theta_type bon_localsearch \
    --x_theta_num_local_searches 1 \
    --top_k_values_for_local_search 10 \
    --property_type toxicity perplexity \
    --lower_bound 0 50 \
    --upper_bound 0.75 150
```

### Key Parameters

- `--x_theta_type bon_localsearch`: Use BoN with local search
- `--x_theta_num_local_searches 1`: Number of local search iterations (usually 1 is enough)
- `--top_k_values_for_local_search K`: Number of top tokens to try per position
  - **Smaller K** = faster but less thorough search
  - **Larger K** = slower but more thorough search
  - Recommended: 5-20 for language data

### Example: Toxicity + Perplexity

Generate text with low toxicity and low perplexity:

```bash
python inference_search.py \
    --data openwebtext-split \
    --prefix_dir data/toxicity/1000_samples \
    --num_samples 10 \
    --batch_size 10 \
    --version toxicity_perplexity_localsearch \
    --x_theta_type bon_localsearch \
    --num_x_theta_samples 16 \
    --x_theta_num_local_searches 1 \
    --top_k_values_for_local_search 10 \
    --lower_bound 0 50 \
    --upper_bound 0.75 150 \
    --property_type toxicity perplexity \
    --seed 0
```

This will:
1. Sample 16 candidates from diffusion model at each step
2. Select best based on toxicity + perplexity
3. Apply local search: try top-10 most probable tokens at each of 100 positions
4. Total: 16 initial + (10 × 100) = 16 + 1000 = 1016 evaluations per step

## Implementation Details

### Files Modified

1. **`x_theta_modifier/local_search_utils.py`**
   - Added `local_search_language()` function
   - Modified `local_search_on_best_tokens()` to support both modes

2. **`x_theta_modifier/bon_local_search.py`**
   - Added `top_k_values_for_local_search` parameter
   - Auto-detect language data types
   - Pass `x_theta` probabilities to local search

3. **`inference_search.py`**
   - Added `--top_k_values_for_local_search` argument

4. **`inference_utils.py`**
   - Pass new parameter through config pipeline

5. **`diffusion_search.py`**
   - Pass parameter to modifier instantiation

### Comparison: Old vs New Local Search

| Aspect | Molecule/Peptide Search | Language Search |
|--------|------------------------|-----------------|
| **Candidates per position** | Fixed set (atoms, amino acids) | Top-k from x_theta |
| **Total candidates** | ~20-50 per position | k per position (configurable) |
| **Uses model info** | No | Yes (x_theta probabilities) |
| **Complexity** | O(vocab_size × seq_len) | O(k × seq_len) |
| **Example** | 50 × 100 = 5,000 | 10 × 100 = 1,000 |

## Performance Tuning

### Memory Considerations

The new local search still loads property models (toxicity, perplexity) temporarily:
- Models start on CPU
- Move to GPU during evaluation
- Move back to CPU after
- Use `torch.cuda.empty_cache()` to free memory

### Speed vs Quality Trade-off

```python
# Fast but less thorough
--top_k_values_for_local_search 3   # 300 evaluations

# Balanced
--top_k_values_for_local_search 10  # 1,000 evaluations

# Thorough but slower
--top_k_values_for_local_search 20  # 2,000 evaluations

# Very thorough (not recommended)
--top_k_values_for_local_search 50  # 5,000 evaluations
```

## Data Type Support

The code automatically detects data type and uses appropriate search:

- **Language data** (uses new local search):
  - `openwebtext-split`
  - `openwebtext`
  - `lm1b`
  - `wikitext103`
  - `wikitext2`

- **Structured data** (uses old local search):
  - `qm9` (molecules)
  - `grampa` (peptides)
  - `trna` (RNA sequences)

## Future Improvements

Possible enhancements:
1. **Adaptive k**: Start with large k, reduce as search converges
2. **Position-specific k**: Use different k values for different positions
3. **Beam search**: Keep top-N candidates instead of just best
4. **Parallel evaluation**: Evaluate all neighbors in batch for speed
