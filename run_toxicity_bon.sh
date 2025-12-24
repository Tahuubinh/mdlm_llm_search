#!/bin/bash
# Complete example for toxicity-guided generation with BoN
# This script demonstrates how to use Best-of-N (BoN) sampling
# with toxicity property to generate less toxic continuations

echo "=========================================="
echo "Toxicity-Guided Generation with BoN"
echo "=========================================="
echo ""
echo "This example generates text continuations from toxic prefixes"
echo "and uses BoN sampling to select the least toxic completions."
echo ""

# Configuration
DATA="openwebtext-split"
NUM_SAMPLES=10
BATCH_SIZE=10
VERSION="bon_toxicity_demo"
SEED=42

# Sampling parameters
POSTERIOR_METHOD="standard"
NUM_POSTERIOR_SAMPLES=32
X_THETA_TYPE="bon"
NUM_X_THETA_SAMPLES=16  # BoN: select best out of N samples

# Property constraints (toxicity)
# Lower bound: 0.0 (minimum toxicity)
# Upper bound: 0.3 (accept only if toxicity < 0.3, i.e., relatively non-toxic)
PROPERTY_TYPE="toxicity"
LOWER_BOUND=0.0
UPPER_BOUND=0.3

# Prefix directory
PREFIX_DIR="data/toxicity/1000_samples"

echo "Configuration:"
echo "  Dataset: $DATA"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  X-theta type: $X_THETA_TYPE"
echo "  Number of BoN samples: $NUM_X_THETA_SAMPLES"
echo "  Property: $PROPERTY_TYPE"
echo "  Toxicity bounds: [$LOWER_BOUND, $UPPER_BOUND]"
echo "  Prefix directory: $PREFIX_DIR"
echo ""

echo "Starting generation..."
python inference_search.py \
    --data $DATA \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --version $VERSION \
    --seed $SEED \
    --posterior_sampling_method $POSTERIOR_METHOD \
    --num_posterior_samples $NUM_POSTERIOR_SAMPLES \
    --x_theta_type $X_THETA_TYPE \
    --num_x_theta_samples $NUM_X_THETA_SAMPLES \
    --prefix_dir $PREFIX_DIR \
    --argmax_mode none \
    --property_type $PROPERTY_TYPE \
    --lower_bound $LOWER_BOUND \
    --upper_bound $UPPER_BOUND

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Generation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  sample_results/$DATA/topk_all_nucleus_all/$POSTERIOR_METHOD/$X_THETA_TYPE/${PROPERTY_TYPE}_lb${LOWER_BOUND}_ub${UPPER_BOUND}/$VERSION/seed_$SEED/"
    echo ""
    echo "You can evaluate the generated texts using:"
    echo "  python properties/toxicity.py"
else
    echo ""
    echo "Generation failed! Check the error messages above."
    exit 1
fi
