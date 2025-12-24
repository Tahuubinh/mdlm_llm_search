#!/bin/bash
# Quick test to verify toxicity property is working

echo "Testing toxicity property integration..."
echo ""

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✓ Cleared Python cache"

# Test imports
echo ""
echo "Testing imports..."
python -c "from properties.toxicity_property import get_toxicity_model; print('✓ toxicity_property import OK')" || exit 1
python -c "from x_theta_modifier.bon import BoNXThetaModifier; print('✓ BoN modifier import OK')" || exit 1
python -c "from posterior_sampling_modifier.base import Sampler; print('✓ Posterior sampler import OK')" || exit 1
python -c "import inference_search; print('✓ inference_search import OK')" || exit 1

echo ""
echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  bash test_bon_toxicity.sh"
echo "  or"
echo "  bash run_toxicity_bon.sh"
