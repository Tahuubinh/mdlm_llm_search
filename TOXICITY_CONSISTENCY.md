# Toxicity Calculation Consistency Check

## Summary
✅ **FIXED**: Evaluation now matches training/sampling preprocessing

## Three Places Toxicity is Calculated

### 1. **properties/toxicity_property.py** (Training/Sampling)
- **Purpose**: Calculate toxicity during sampling/local search
- **Text preprocessing**: 
  ```python
  clean_text_samples(texts)  # Removes <bos>, <eos>, <pad>, etc.
  ```
- **Tokenization**:
  ```python
  tokenizer(texts, padding=True, truncation=True, max_length=100)
  ```
- **Model**: GPTNeoForBinaryClassification from `outputs/toxicity/classifier/toxicity_gpt2_neo.pt`
- **Device**: CPU by default, moves to GPU during inference, then back to CPU

### 2. **evaluation/toxicity.py** (Evaluation) - NOW FIXED
- **Purpose**: Evaluate toxicity of saved text files
- **Text preprocessing** (BEFORE FIX):
  ```python
  f.read().strip()  # ❌ NO special token removal!
  ```
- **Text preprocessing** (AFTER FIX):
  ```python
  clean_text_sample(f.read().strip())  # ✅ Removes special tokens
  ```
- **Tokenization**:
  ```python
  tokenizer(texts, padding=True, truncation=True, max_length=100)
  ```
- **Model**: Same GPTNeoForBinaryClassification
- **Device**: GPU if available

### 3. **x_theta_modifier/local_search_language_utils.py** (Input Preparation)
- **Purpose**: Prepare text for property calculation
- **Decoding**:
  ```python
  best_texts = tokenizer.batch_decode(best_tokens.cpu().numpy())
  ```
- **Cleaning**:
  ```python
  best_texts_cleaned = clean_text_samples(best_texts)
  ```
- **Then passes to**: `calc_toxicity_parallel(best_texts_cleaned, ...)`

## Key Consistency Points

### ✅ CONSISTENT NOW:
1. **Text cleaning**: All use same special token removal
2. **max_length**: All use 100
3. **Model**: Same model file
4. **Tokenizer**: Same GPT2Tokenizer from 'EleutherAI/gpt-neo-1.3B'
5. **Preprocessing order**: decode → clean → tokenize → model

### ⚠️ Minor Differences (Acceptable):
1. **Device management**: 
   - Training: CPU → GPU temporarily → CPU (memory efficient)
   - Evaluation: GPU throughout (faster for batch evaluation)
2. **Batch size**:
   - Training: chunk_size_gpu = 256
   - Evaluation: User-specified batch_size (typically 8)

## Verification

To verify consistency, test the same text in both systems:

```python
from properties.toxicity_property import calc_toxicity_parallel
import torch

text = "This is a test sentence."
device = torch.device('cuda')

# Training/sampling calculation
score_training = calc_toxicity_parallel([text], 1, device)
print(f"Training: {score_training.item():.4f}")

# Evaluation calculation (after reading from file)
# Should give same result
```

## Changes Made

1. **evaluation/toxicity.py**:
   - Added `clean_text_sample()` function
   - Modified file reading to clean special tokens
   - Added comment about max_length consistency

## Recommendation

Consider extracting text cleaning logic to a shared utility module:
```python
# properties/text_utils.py
def clean_special_tokens(text):
    """Shared cleaning logic for all text preprocessing"""
    return text.replace('<bos>', '').replace('<eos>', '')...
```

Then import from all three locations to ensure perfect consistency.
