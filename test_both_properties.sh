#!/bin/bash
# Test script for using both toxicity and perplexity properties together

# Activate conda environment
source /u/nzj6jt/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

# Test with both properties
python inference_search.py \
    --data openwebtext-split \
    --prefix_dir data/toxicity/1000_samples \
    --num_samples 5 \
    --batch_size 5 \
    --version test_toxicity_perplexity \
    --x_theta_type bon \
    --lower_bound 0 50 \
    --upper_bound 0.75 150 \
    --property_type toxicity perplexity \
    --seed 0

echo "Test completed! Check outputs for results."
