#!/bin/bash
# Test script demonstrating the optimization when num_x_theta_samples=1
# With num_x_theta_samples=1, BoN selection is skipped entirely - no property computation!
# Only local search will use properties.

source /u/nzj6jt/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

echo "=========================================="
echo "Test 1: num_x_theta_samples=1 (Optimized)"
echo "=========================================="
echo "This will skip BoN property computation!"
echo "Only local search will compute properties"
echo ""

python inference_search.py \
    --data openwebtext-split \
    --prefix_dir data/toxicity/1000_samples \
    --num_samples 3 \
    --batch_size 3 \
    --version test_num_samples_1 \
    --x_theta_type bon_localsearch \
    --num_x_theta_samples 1 \
    --x_theta_num_local_searches 1 \
    --top_k_values_for_local_search 5 \
    --lower_bound 0 50 \
    --upper_bound 0.75 150 \
    --property_type toxicity perplexity \
    --seed 0

echo ""
echo "=========================================="
echo "Test 2: num_x_theta_samples=16 (Standard)"
echo "=========================================="
echo "This will use BoN with property computation"
echo ""

python inference_search.py \
    --data openwebtext-split \
    --prefix_dir data/toxicity/1000_samples \
    --num_samples 3 \
    --batch_size 3 \
    --version test_num_samples_16 \
    --x_theta_type bon_localsearch \
    --num_x_theta_samples 16 \
    --x_theta_num_local_searches 1 \
    --top_k_values_for_local_search 5 \
    --lower_bound 0 50 \
    --upper_bound 0.75 150 \
    --property_type toxicity perplexity \
    --seed 0

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "Test 1 (num_samples=1): Should see 'Skipping BoN selection' messages"
echo "Test 2 (num_samples=16): Normal BoN with property evaluation"
echo ""
echo "The optimization saves significant time when you only want local search"
echo "without the overhead of evaluating multiple samples."
