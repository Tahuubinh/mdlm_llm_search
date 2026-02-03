"""
Reward property calculation using OpenAssistant reward model.
This module provides a singleton model loader to ensure the reward model
is loaded only once and reused across all calculations.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Singleton pattern for model loading
_REWARD_MODEL = None
_REWARD_TOKENIZER = None
_REWARD_DEVICE = None


def get_reward_model(model_name='OpenAssistant/reward-model-deberta-v3-large-v2'):
    """
    Get the reward model (singleton pattern).
    Loads the model only once and reuses it for all subsequent calls.
    
    Args:
        model_name (str): Name of the reward model to use
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    global _REWARD_MODEL, _REWARD_TOKENIZER, _REWARD_DEVICE
    
    if _REWARD_MODEL is None:
        # Load model directly on GPU and keep it there permanently
        _REWARD_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading reward model ({model_name}) on device: {_REWARD_DEVICE} (will stay in VRAM)")
        
        # Disable TorchScript JIT compilation globally to avoid fabs error in DeBERTa
        torch.jit._state.disable()
        
        # Load tokenizer
        _REWARD_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        
        # Load model on GPU
        _REWARD_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        _REWARD_MODEL.to(_REWARD_DEVICE)
        _REWARD_MODEL.eval()
        
        print(f"Reward model loaded successfully on {_REWARD_DEVICE}!")
        print("Note: TorchScript JIT compilation disabled to avoid DeBERTa compilation errors")
    
    return _REWARD_MODEL, _REWARD_TOKENIZER, _REWARD_DEVICE


def calculate_reward(text, model_name='OpenAssistant/reward-model-deberta-v3-large-v2', max_length=512):
    """
    Calculate reward score for a single text.
    Higher reward = better quality text.
    
    Args:
        text (str): Input text to evaluate
        model_name (str): Name of the reward model to use
        max_length (int): Maximum token length for model input
        
    Returns:
        float: Reward score (higher is better), or -float('inf') if calculation fails
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return -float('inf')
    
    try:
        model, tokenizer, model_device = get_reward_model(model_name)
        
        # Use raw text directly (no template needed for DeBERTa-based reward model)
        # Model gives scores typically in range -10 to +10
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        
        input_ids = encodings["input_ids"].to(model_device)
        attention_mask = encodings["attention_mask"].to(model_device)
        
        # Calculate reward score
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Reward model outputs a single score (logit)
            reward_score = outputs.logits.squeeze(-1).item()
        
        # Free GPU memory immediately
        del input_ids, attention_mask, outputs
        
        return reward_score
    
    except Exception as e:
        print(f"Error calculating reward: {e}")
        return -float('inf')


def calc_reward_parallel(sequence_list, batch_size, device, model_name='OpenAssistant/reward-model-deberta-v3-large-v2', max_length=512):
    """
    Calculate reward scores for a batch of sequences in parallel.
    Higher reward is better (higher quality text).
    
    Args:
        sequence_list (list): List of text sequences (already cleaned, no special tokens)
        batch_size (int): Number of sequences in the batch
        device (torch.device): Device to store results
        model_name (str): Name of the reward model to use
        max_length (int): Maximum token length for model input
        
    Returns:
        torch.Tensor: Reward scores for each sequence (higher is better)
    """
    model, tokenizer, model_device = get_reward_model(model_name)
    
    # Initialize reward tensor with negative infinity (worst score)
    reward_scores = torch.full((batch_size,), -float('inf'), device=device, dtype=torch.float32)
    
    if batch_size == 0 or not sequence_list:
        return reward_scores
    
    # Filter out empty sequences
    valid_sequences = []
    valid_indices = []
    for i, seq in enumerate(sequence_list):
        if seq and isinstance(seq, str) and len(seq.strip()) > 0:
            valid_sequences.append(seq)
            valid_indices.append(i)
    
    if not valid_sequences:
        return reward_scores
    
    # Process in chunks to avoid OOM
    chunk_size_gpu = 128  # Chunk size for DeBERTa-v3-Large
    all_reward_scores = []
    
    for chunk_start in tqdm(range(0, len(valid_sequences), chunk_size_gpu), desc="Reward"):
        chunk_end = min(chunk_start + chunk_size_gpu, len(valid_sequences))
        chunk_sequences = valid_sequences[chunk_start:chunk_end]
        
        # Use raw text directly (no template needed for DeBERTa-based reward model)
        # Tokenize chunk
        encodings = tokenizer(
            chunk_sequences,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        
        input_ids = encodings["input_ids"].to(model_device)
        attention_mask = encodings["attention_mask"].to(model_device)
        
        # Calculate reward scores for chunk
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Reward model outputs scores (logits) with shape [batch_size, 1] or [batch_size]
            chunk_rewards = outputs.logits.squeeze(-1).cpu().float()
            all_reward_scores.append(chunk_rewards)
        
        # Free GPU memory
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
    
    # Concatenate all rewards
    if all_reward_scores:
        batch_rewards = torch.cat(all_reward_scores) if len(all_reward_scores) > 1 else all_reward_scores[0]
        
        # Assign valid rewards back to original indices
        for i, idx in enumerate(valid_indices):
            reward_scores[idx] = batch_rewards[i]
    
    # Move final results to target device
    reward_scores = reward_scores.to(device)
    
    return reward_scores


def calc_reward_scores_with_mask(
    sampled_token_ids, 
    prefix_token_ids,
    tokenizer,
    batch_size,
    device,
    model_name='OpenAssistant/reward-model-deberta-v3-large-v2',
    max_length=512
):
    """
    Calculate reward scores for sequences with prefix.
    Evaluates only the post-prefix portion (removes prefix before evaluation).
    
    Args:
        sampled_token_ids (torch.Tensor): Full token IDs including prefix [batch_size, seq_len]
        prefix_token_ids (torch.Tensor): Prefix token IDs [prefix_len]
        tokenizer: Tokenizer for decoding
        batch_size (int): Batch size
        device (torch.device): Target device for results
        model_name (str): Reward model name
        max_length (int): Maximum sequence length for reward model
        
    Returns:
        torch.Tensor: Reward scores [batch_size] (higher is better)
    """
    prefix_length = len(prefix_token_ids)
    
    # Extract post-prefix tokens
    post_prefix_ids = sampled_token_ids[:, prefix_length:]
    
    # Decode to text (skip special tokens like <|endoftext|>)
    texts = [
        tokenizer.decode(seq, skip_special_tokens=True)
        for seq in post_prefix_ids
    ]
    
    # Calculate reward scores
    reward_scores = calc_reward_parallel(
        texts, 
        batch_size, 
        device, 
        model_name=model_name,
        max_length=max_length
    )
    
    return reward_scores


class RewardProperty:
    """
    Property class for reward-based text quality evaluation.
    Higher reward = better quality.
    """
    
    def __init__(self, model_name='OpenAssistant/reward-model-deberta-v3-large-v2', max_length=512):
        """
        Initialize the reward property.
        
        Args:
            model_name (str): Name of the reward model to use
            max_length (int): Maximum sequence length for reward model
        """
        self.model_name = model_name
        self.max_length = max_length
        self.name = "reward"
        
        # Load model immediately
        get_reward_model(model_name)
    
    def __call__(self, sampled_token_ids, prefix_token_ids, tokenizer, batch_size, device):
        """
        Calculate reward scores for a batch of sequences.
        
        Args:
            sampled_token_ids (torch.Tensor): Token IDs [batch_size, seq_len]
            prefix_token_ids (torch.Tensor): Prefix token IDs
            tokenizer: Tokenizer
            batch_size (int): Batch size
            device (torch.device): Device
            
        Returns:
            torch.Tensor: Reward scores (higher is better)
        """
        return calc_reward_scores_with_mask(
            sampled_token_ids,
            prefix_token_ids,
            tokenizer,
            batch_size,
            device,
            model_name=self.model_name,
            max_length=self.max_length
        )
    
    def calculate_single(self, text):
        """
        Calculate reward for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Reward score (higher is better)
        """
        return calculate_reward(text, model_name=self.model_name, max_length=self.max_length)
