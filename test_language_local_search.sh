#!/bin/bash
# Test script for language local search with toxicity and perplexity properties

# Activate conda environment
source /u/nzj6jt/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

# Test with both properties using the new language local search
# Using top_k_values_for_local_search=5 means:
# - For each of the 100 positions in sequence
# - Try the top-5 most probable tokens from x_theta
# - Total: 5 * 100 = 500 neighbor sequences evaluated per diffusion step
# This is much faster than trying all 50k vocab tokens!

python inference_search.py \
    --data openwebtext-split \
    --prefix_dir data/toxicity/1000_samples \
    --num_samples 3 \
    --batch_size 3 \
    --version test_language_local_search \
    --x_theta_type bon_localsearch \
    --num_x_theta_samples 16 \
    --x_theta_num_local_searches 1 \
    --top_k_values_for_local_search 5 \
    --lower_bound 0 50 \
    --upper_bound 0.75 150 \
    --property_type toxicity perplexity \
    --seed 0

echo ""
echo "Test completed!"
echo "With top_k=5 and seq_len=100, we evaluated ~500 neighbors per step"
echo "vs. ~5M neighbors if we tried the whole vocabulary!"
