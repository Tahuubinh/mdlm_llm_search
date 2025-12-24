#!/bin/bash
# Test the new batch-optimized language local search

source /u/nzj6jt/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

echo "=========================================="
echo "Testing NEW batch-optimized language local search"
echo "=========================================="
echo "This version processes all sequences in batch"
echo "Much faster than the old sequential version!"
echo ""

python inference_search.py \
    --data openwebtext-split \
    --prefix_dir data/toxicity/1000_samples \
    --num_samples 1 \
    --batch_size 1 \
    --version test_batch_language_search \
    --x_theta_type local_search_language \
    --num_x_theta_samples 1 \
    --x_theta_num_local_searches 1 \
    --top_k_values_for_local_search 2 \
    --lower_bound -100 -100 \
    --upper_bound 0.75 0 \
    --property_type toxicity perplexity \
    --seed 0

echo ""
echo "=========================================="
echo "Key differences from old implementation:"
echo "=========================================="
echo "✅ Batch processing: All sequences processed together"
echo "✅ Single model loading: Properties computed in batches"
echo "✅ Much faster: No per-sequence model loading overhead"
echo "✅ Better logging: Shows progress per rank/position"
