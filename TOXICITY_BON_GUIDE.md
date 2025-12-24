# Toxicity-Guided Text Generation with BoN

This guide explains how to use Best-of-N (BoN) sampling with toxicity property to generate less toxic text continuations.

## Overview

The toxicity property is calculated using a GPT-Neo based classifier that scores text on a scale from 0.0 (non-toxic) to 1.0 (toxic). The BoN sampler generates N candidate completions at each diffusion step and selects the one with the lowest toxicity score.

## Architecture

### Key Components

1. **Toxicity Model** (`properties/toxicity_property.py`)
   - Singleton pattern to load model once
   - Efficient batch processing
   - GPT-Neo 1.3B classifier fine-tuned on toxicity data

2. **BoN X-Theta Modifier** (`x_theta_modifier/bon.py`)
   - Generates N candidate samples using Gumbel sampling
   - Evaluates each candidate using toxicity model
   - Selects best candidate based on property constraints

3. **Property Integration**
   - Added to `x_theta_modifier/base.py`
   - Added to `posterior_sampling_modifier/base.py`
   - Seamless integration with existing property framework

## Usage

### Basic Example

```bash
bash run_toxicity_bon.sh
```

### Advanced Configuration

```bash
python inference_search.py \
    --data openwebtext-split \
    --num_samples 100 \
    --batch_size 10 \
    --version my_experiment \
    --seed 42 \
    --x_theta_type bon \
    --num_x_theta_samples 32 \
    --property_type toxicity \
    --lower_bound 0.0 \
    --upper_bound 0.3 \
    --prefix_dir data/toxicity/1000_samples
```

### Parameters

- `--x_theta_type bon`: Use Best-of-N sampling
- `--num_x_theta_samples N`: Generate N candidates and select best (default: 32)
- `--property_type toxicity`: Use toxicity as the property to optimize
- `--lower_bound 0.0`: Minimum acceptable toxicity (usually 0.0)
- `--upper_bound 0.3`: Maximum acceptable toxicity (lower = less toxic)
- `--prefix_dir`: Directory containing toxic prefixes to continue from

## How It Works

### 1. Model Loading (Once)
The toxicity model is loaded once using a singleton pattern:
```python
from properties.toxicity_property import get_toxicity_model
model, tokenizer, device = get_toxicity_model()
```

### 2. BoN Sampling Process
At each diffusion step:
1. Generate N candidate samples using Gumbel sampling
2. Decode each candidate to text
3. Calculate toxicity score for each
4. Select the candidate with lowest toxicity (within bounds)
5. Use this as the next diffusion state

### 3. Property Constraint
- If toxicity < `upper_bound`: Accept the sample
- Otherwise: Try next best candidate
- Combines with other properties if specified

## Evaluation

After generation, evaluate the toxicity of generated texts:

```python
python properties/toxicity.py
```

Modify the script to point to your output directory:
```python
output_sequence_dir = "sample_results/openwebtext-split/.../molecules"
```

## Performance Tips

1. **Batch Size**: Use larger batch sizes (e.g., 100) for faster generation
2. **BoN Samples**: More samples (N=32-64) gives better quality but slower
3. **GPU Memory**: Toxicity model requires ~5GB VRAM
4. **Caching**: Model is loaded once and cached for all calculations

## Example Output Structure

```
sample_results/openwebtext-split/
└── topk_all_nucleus_all/
    └── standard/
        └── bon/
            └── toxicity_lb0.0_ub0.3/
                └── my_experiment/
                    └── seed_42/
                        ├── molecules/      # Post-prefix generations
                        │   ├── 0.txt
                        │   ├── 1.txt
                        │   └── ...
                        ├── raw_molecules/  # Full generations (with prefix)
                        ├── generation_time.txt
                        └── params.txt
```

## Toxicity Bounds Guidelines

- **Very Low Toxicity**: `upper_bound=0.2`
- **Low Toxicity**: `upper_bound=0.3`
- **Moderate**: `upper_bound=0.5`
- **Baseline** (no filtering): `upper_bound=1.0`

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Toxicity model not found at outputs/toxicity/classifier/toxicity_gpt2_neo.pt
```
**Solution**: Ensure the toxicity classifier model is present in the specified path.

### Out of Memory
```
CUDA out of memory
```
**Solution**: Reduce `num_x_theta_samples` or `batch_size`.

### Slow Generation
**Solution**: 
- Reduce `num_x_theta_samples` (e.g., 8-16 instead of 32)
- Increase `batch_size` for better GPU utilization
- Use fewer diffusion steps (e.g., 500 instead of 1000)

## Citation

If you use this toxicity-guided generation in your research, please cite the original MDLM paper and the toxicity dataset used for training the classifier.
