"""
Perplexity property calculation using GPT-2 model.
This module provides a singleton model loader to ensure the perplexity model
is loaded only once and reused across all calculations.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
from tqdm import tqdm

# Singleton pattern for model loading
_PERPLEXITY_MODEL = None
_PERPLEXITY_TOKENIZER = None
_PERPLEXITY_DEVICE = None


def get_perplexity_model(model_name='gpt2-large'):
    """
    Get the perplexity model (singleton pattern).
    Loads the model only once and reuses it for all subsequent calls.
    
    Args:
        model_name (str): Name of the GPT-2 model to use
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    global _PERPLEXITY_MODEL, _PERPLEXITY_TOKENIZER, _PERPLEXITY_DEVICE
    
    if _PERPLEXITY_MODEL is None:
        # Load model directly on GPU and keep it there permanently
        _PERPLEXITY_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading perplexity model ({model_name}) on device: {_PERPLEXITY_DEVICE} (will stay in VRAM)")
        
        # Load tokenizer
        _PERPLEXITY_TOKENIZER = GPT2Tokenizer.from_pretrained(model_name)
        _PERPLEXITY_TOKENIZER.pad_token = _PERPLEXITY_TOKENIZER.eos_token
        
        # Load model on GPU
        _PERPLEXITY_MODEL = GPT2LMHeadModel.from_pretrained(model_name)
        _PERPLEXITY_MODEL.to(_PERPLEXITY_DEVICE)
        _PERPLEXITY_MODEL.eval()
        
        print(f"Perplexity model loaded successfully on {_PERPLEXITY_DEVICE}!")
        
        # OLD CODE (moved model between CPU/GPU to save VRAM):
        # _PERPLEXITY_DEVICE = torch.device('cpu')
        # print(f"Loading perplexity model ({model_name}) on device: {_PERPLEXITY_DEVICE} (will move to GPU only during inference)")
        # _PERPLEXITY_MODEL.to('cpu')
        # _PERPLEXITY_MODEL.eval()
    
    return _PERPLEXITY_MODEL, _PERPLEXITY_TOKENIZER, _PERPLEXITY_DEVICE


def calculate_perplexity(text, model_name='gpt2-large', max_length=512):
    """
    Calculate perplexity score for a single text.
    
    Args:
        text (str): Input text to evaluate
        model_name (str): Name of the GPT-2 model to use
        max_length (int): Maximum token length for model input
        
    Returns:
        float: Perplexity score, or float('inf') if calculation fails
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return float('inf')
    
    try:
        model, tokenizer, model_device = get_perplexity_model(model_name)
        
        # Model is already on GPU permanently, no need to move
        # OLD CODE (moved model to GPU temporarily):
        # model.to('cuda')
        
        # Tokenize
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="longest",
        )
        
        input_ids = encodings["input_ids"].to('cuda')
        attention_mask = encodings["attention_mask"].to('cuda')
        
        # Calculate perplexity
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            nll = outputs.loss.item()  # Average cross-entropy over tokens
        
        ppl = math.exp(nll)
        
        # Free GPU memory immediately
        del input_ids, attention_mask, outputs
        
        # Model stays on GPU permanently, no need to move back to CPU
        # OLD CODE (moved model back to CPU to save VRAM):
        # model.to('cpu')
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        
        return ppl
    
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')


def calc_perplexity_parallel(sequence_list, batch_size, device, model_name='gpt2-large', max_length=512):
    """
    Calculate perplexity scores for a batch of sequences in parallel.
    Lower perplexity is better (more fluent text).
    
    Args:
        sequence_list (list): List of text sequences (already cleaned, no special tokens)
        batch_size (int): Number of sequences in the batch
        device (torch.device): Device to store results
        model_name (str): Name of the GPT-2 model to use
        max_length (int): Maximum token length for model input
        
    Returns:
        torch.Tensor: Perplexity scores for each sequence
    """
    model, tokenizer, model_device = get_perplexity_model(model_name)
    
    # Initialize perplexity tensor with infinity (worst score)
    perplexities = torch.full((batch_size,), float('inf'), device=device)
    
    # Filter out empty sequences
    valid_sequences = []
    valid_indices = []
    for i, seq in enumerate(sequence_list):
        if seq and isinstance(seq, str) and len(seq.strip()) > 0:
            valid_sequences.append(seq)
            valid_indices.append(i)
    
    if not valid_sequences:
        return perplexities
    
    # Model is already on GPU permanently, no need to move
    # OLD CODE (moved model to GPU temporarily):
    # model.to('cuda')
    
    # Process in chunks to avoid OOM (reduced from 128 to 64)
    chunk_size_gpu = 64
    all_perplexities = []
    
    for chunk_start in tqdm(range(0, len(valid_sequences), chunk_size_gpu), desc="Perplexity"):
        chunk_end = min(chunk_start + chunk_size_gpu, len(valid_sequences))
        chunk_sequences = valid_sequences[chunk_start:chunk_end]
        
        # Tokenize chunk
        encodings = tokenizer(
            chunk_sequences,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        input_ids = encodings["input_ids"].to('cuda')
        attention_mask = encodings["attention_mask"].to('cuda')
        
        # Calculate perplexity scores for chunk
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Get per-sample loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            
            # Calculate loss per sample
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())
            
            # Apply attention mask and calculate mean loss per sample
            loss = (loss * shift_attention_mask).sum(dim=1) / shift_attention_mask.sum(dim=1)
            
            # Convert to perplexity
            chunk_perplexities = torch.exp(loss).cpu()
            all_perplexities.append(chunk_perplexities)
        
        torch.cuda.empty_cache()
    
    # Concatenate all perplexities
    batch_perplexities = torch.cat(all_perplexities) if len(all_perplexities) > 1 else all_perplexities[0]
    
    # Model stays on GPU permanently, no need to move back to CPU
    # OLD CODE (moved model back to CPU to save VRAM):
    # model.to('cpu')
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    
    # Assign scores to corresponding positions
    for idx, ppl in zip(valid_indices, batch_perplexities):
        perplexities[idx] = ppl.to(device)
    
    return perplexities


if __name__ == "__main__":
    # Example usage
    test_texts = [
        "This is a well-written and grammatically correct sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "asdf qwer zxcv tyui",  # Nonsensical text should have high perplexity
        "",  # Empty text
    ]
    
    print("Testing single perplexity calculation:")
    for text in test_texts:
        score = calculate_perplexity(text)
        print(f"Text: {text[:50] if text else '(empty)'}...")
        print(f"Perplexity score: {score:.4f}\n")
    
    print("\nTesting parallel perplexity calculation:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_scores = calc_perplexity_parallel(test_texts, len(test_texts), device)
    print("Batch scores:", batch_scores)
