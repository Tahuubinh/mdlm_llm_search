"""
Toxicity property calculation using GPTNeo classifier.
This module provides a singleton model loader to ensure the toxicity model
is loaded only once and reused across all calculations.
"""

import torch
# from transformers import GPT2Tokenizer
import os
from tqdm import tqdm

# Singleton pattern for model loading
_TOXICITY_MODEL = None
_TOXICITY_TOKENIZER = None
_TOXICITY_DEVICE = None

def get_toxicity_model():
    """
    Get the toxicity model (singleton pattern).
    Loads the model only once and reuses it for all subsequent calls.
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    global _TOXICITY_MODEL, _TOXICITY_TOKENIZER, _TOXICITY_DEVICE
    
    if _TOXICITY_MODEL is None:
        # Import model classes directly to avoid circular dependencies
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        from torch import nn
        import torch.nn.functional as F
        
        # Define model class inline
        class GPTNeoForBinaryClassification(nn.Module):
            def __init__(self, base_model_name='EleutherAI/gpt-neo-1.3B', num_labels=1):
                super(GPTNeoForBinaryClassification, self).__init__()
                self.gpt_neo = GPTNeoForCausalLM.from_pretrained(base_model_name)
                
                # Freeze most of the model parameters
                for param in self.gpt_neo.parameters():
                    param.requires_grad = False
                # Only fine-tune the last 2 layers
                for param in self.gpt_neo.transformer.h[-2:].parameters():
                    param.requires_grad = True
                    
                self.classifier = nn.Sequential(
                    nn.Linear(self.gpt_neo.config.hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_labels)
                )
                self.sigmoid = nn.Sigmoid()

            def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None):
                outputs = self.gpt_neo(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
                pooled_output = masked_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                logits = self.classifier(pooled_output)
                probs = self.sigmoid(logits)

                loss = None
                if labels is not None:
                    pos_weight = torch.tensor([2.0]).to(labels.device)
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fct(logits.view(-1), labels.float().view(-1))

                return {
                    'loss': loss,
                    'logits': probs
                }
        
        # Load model directly on GPU and keep it there permanently
        _TOXICITY_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading toxicity model on device: {_TOXICITY_DEVICE} (will stay in VRAM)")
        
        # Load model
        model_path = 'outputs/toxicity/classifier/toxicity_gpt2_neo.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Toxicity model not found at {model_path}")
        
        _TOXICITY_MODEL = GPTNeoForBinaryClassification()
        _TOXICITY_MODEL.load_state_dict(torch.load(model_path, map_location=_TOXICITY_DEVICE))
        _TOXICITY_MODEL.to(_TOXICITY_DEVICE)
        # CRITICAL: Force eval mode recursively on ALL submodules (including gpt_neo)
        _TOXICITY_MODEL.eval()
        _TOXICITY_MODEL.training = False
        for module in _TOXICITY_MODEL.modules():
            module.training = False
        
        # Setup tokenizer
        _TOXICITY_TOKENIZER = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        _TOXICITY_TOKENIZER.pad_token = _TOXICITY_TOKENIZER.eos_token
        
        print(f"Toxicity model loaded successfully on {_TOXICITY_DEVICE}!")
        
        # OLD CODE (moved model between CPU/GPU to save VRAM):
        # _TOXICITY_DEVICE = torch.device('cpu')
        # print(f"Loading toxicity model on device: {_TOXICITY_DEVICE} (will move to GPU only during inference)")
        # _TOXICITY_MODEL.to('cpu')
        # _TOXICITY_MODEL.eval()
    
    return _TOXICITY_MODEL, _TOXICITY_TOKENIZER, _TOXICITY_DEVICE


def calculate_toxicity(text, max_length=100):
    """
    Calculate toxicity score for a single text.
    
    Args:
        text (str): Input text to evaluate
        max_length (int): Maximum token length for model input
        
    Returns:
        float: Toxicity score (0.0 to 1.0), or None if calculation fails
    """
    if not text or not isinstance(text, str):
        return None
    
    try:
        model, tokenizer, model_device = get_toxicity_model()
        
        # Model is already on GPU permanently, no need to move
        # OLD CODE (moved model to GPU temporarily):
        # model.to('cuda')
        # model.eval()
        # model.training = False
        # for module in model.modules():
        #     module.training = False
        
        # Force deterministic behavior
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        input_ids = inputs["input_ids"].to('cuda')
        attention_mask = inputs["attention_mask"].to('cuda')
        
        # Compute toxicity
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        toxicity_score = outputs['logits'].cpu().item()
        
        # Free GPU memory immediately
        del input_ids, attention_mask, outputs
        
        # Model stays on GPU permanently, no need to move back to CPU
        # OLD CODE (moved model back to CPU to save VRAM):
        # model.to('cpu')
        # model.eval()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        
        return toxicity_score
    
    except Exception as e:
        print(f"Error calculating toxicity: {e}")
        return None


def calc_toxicity_parallel(sequence_list, batch_size, device, max_length=100):
    """
    Calculate toxicity scores for a batch of sequences in parallel.
    
    Args:
        sequence_list (list): List of text sequences (already cleaned, no special tokens)
        batch_size (int): Number of sequences in the batch
        device (torch.device): Device to store results
        max_length (int): Maximum token length for model input
        
    Returns:
        torch.Tensor: Toxicity scores for each sequence
    """
    model, tokenizer, model_device = get_toxicity_model()
    
    # Initialize rewards tensor
    rewards = torch.zeros(batch_size, device=device)
    
    # Model is already on GPU permanently, no need to move
    # OLD CODE (moved model to GPU temporarily):
    # model.to('cuda')
    # model.eval()
    # model.training = False
    # for module in model.modules():
    #     module.training = False
    
    # Force deterministic behavior for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic
    
    # Process in chunks to avoid OOM (reduced from 128 to 64)
    chunk_size_gpu = 64
    all_toxicity_scores = []
    
    for chunk_start in tqdm(range(0, batch_size, chunk_size_gpu), desc="Toxicity"):
        chunk_end = min(chunk_start + chunk_size_gpu, batch_size)
        chunk_sequences = sequence_list[chunk_start:chunk_end]
        
        # Tokenize chunk sequences
        # CRITICAL: Must pad to EXACTLY max_length to match file evaluation
        # padding='max_length' ensures consistent padding regardless of batch composition
        inputs = tokenizer(
            chunk_sequences,
            return_tensors='pt',
            padding='max_length',  # Force pad to exactly max_length
            truncation=True,
            max_length=max_length
        ).to('cuda')
        
        # DEBUG: Print token IDs for sequence at global index 1 if in this chunk
        global_idx_1 = 1
        if chunk_start <= global_idx_1 < chunk_end:
            local_idx = global_idx_1 - chunk_start
            print(f"    DEBUG properties/toxicity Batch 1 input_ids: {inputs['input_ids'][local_idx].cpu().tolist()[:20]}...(total {inputs['input_ids'][local_idx].shape[0]} tokens)")
            print(f"    DEBUG properties/toxicity Batch 1 attention_mask: {inputs['attention_mask'][local_idx].cpu().tolist()[:20]}...(total {inputs['attention_mask'][local_idx].shape[0]} tokens)")
        
        # Compute toxicity scores for chunk
        with torch.no_grad():
            outputs = model(**inputs)
        
        # DEBUG: Print logit value for Batch 1
        if chunk_start <= global_idx_1 < chunk_end:
            local_idx = global_idx_1 - chunk_start
            logit_value = outputs['logits'][local_idx].item()
            print(f"    DEBUG properties/toxicity Batch 1 RAW toxicity score (after sigmoid): {logit_value:.6f}")
        
        chunk_scores = outputs['logits'].squeeze(-1).cpu()
        all_toxicity_scores.append(chunk_scores)
    
    # Concatenate all scores
    toxicity_scores = torch.cat(all_toxicity_scores) if len(all_toxicity_scores) > 1 else all_toxicity_scores[0]
    
    # Model stays on GPU permanently, no need to move back to CPU
    # OLD CODE (moved model back to CPU to save VRAM):
    # model.to('cpu')
    # model.eval()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    
    rewards[:batch_size] = toxicity_scores.to(device)
    return rewards


if __name__ == "__main__":
    # Example usage
    test_texts = [
        "This is a nice and friendly message.",
        "You are stupid and I hate you!",
        "Let's have a pleasant conversation.",
    ]
    
    print("Testing single toxicity calculation:")
    for text in test_texts:
        score = calculate_toxicity(text)
        print(f"Text: {text[:50]}...")
        print(f"Toxicity score: {score:.4f}\n")
    
    print("\nTesting parallel toxicity calculation:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_scores = calc_toxicity_parallel(test_texts, len(test_texts), device)
    print("Batch scores:", batch_scores)
