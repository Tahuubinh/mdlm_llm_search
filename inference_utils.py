import torch
from omegaconf import OmegaConf
import diffusion_search
import dataloader
import os
from transformers import GPT2Tokenizer

def setup_config_no_guidance(data, sequence_length, diffusion_steps, checkpoint_path, tokenizer_name_or_path, argmax_mode='none', posterior_sampling_config=None, x_theta_config=None, seed=42):
    """Setup config for local checkpoint without guidance
    
    Args:
        checkpoint_path: Path to the local checkpoint file
        tokenizer_name_or_path: Path to the tokenizer model
        x_theta_config: Optional dictionary with x_theta modification parameters
        argmax_mode: When to use argmax sampling ('none', 'last_step', 'all_steps')
        posterior_sampling_config: Configuration for posterior sampling method
        seed: Random seed for reproducibility
    """
    # Create a complete config for QM9 UDLM model without guidance
    config = OmegaConf.create({
        # Model and backbone settings
        'mode': 'train',
        'backbone': 'dit',  # Changed from 'hf_dit' to 'dit'
        'classifier_backbone': None,
        'diffusion': 'absorbing_state',
        'parameterization': 'subs',
        'time_conditioning': True,
        'subs_masking': False,
        'zero_recon_loss': True,
        'T': 0,
        'is_vision': False,
        'seed': seed,
        
        # Model config - updated for local checkpoint
        'model': {
            'name': 'small',
            'type': 'ddit',
            'hidden_size': 768,
            'cond_dim': 128,
            'length': sequence_length,  
            'n_blocks': 12,
            'n_heads': 12,
            'scale_by_sigma': True,
            'dropout': 0.1,
            'tie_word_embeddings': False
        },
        
        # Data config
        'data': {
            'train': data,
            'valid': data,
            'tokenizer_name_or_path': tokenizer_name_or_path,
            'cache_dir': '/tmp/huggingface_cache',
            'wrap': False,
            'streaming': False,
            'override_cache': False,
            'add_special_tokens': True,
            'label_col': None,
            'label_col_pctile': None,
            'num_classes': None
        },
        
        # Noise schedule
        'noise': {
            'type': 'loglinear'
        },
        
        # Guidance (None for no guidance)
        'guidance': None,
        
        
        # Sampling config
        'sampling': {
            'predictor': 'ddpm_cache',  # Required: analytic, ddpm, ddpm_cache
            'use_cache': False,  # Match eval script for UDLM
            'steps': diffusion_steps,
            'noise_removal': True,  # Required by diffusion_search.py
            'batch_size': 16,    # Not important here; will be overridden
            'num_sample_batches': 64,  # Not important here; will be overridden
            'num_sample_log': 2,  # Required by config
            'semi_ar': False,  # Required by config
            'stride_length': 1,  # Required by config
            'num_strides': 1,  # Required by config
            'use_float64': False,
            'argmax_mode': argmax_mode,
            # Add posterior sampling configuration
            'posterior_sampling_method': posterior_sampling_config.get('method'),
            'num_posterior_samples': posterior_sampling_config.get('num_posterior_samples'),
            'num_mcts_samples': posterior_sampling_config.get('num_mcts_samples'),
            'num_local_searches': posterior_sampling_config.get('num_local_searches'),
            'top_k_categorical': posterior_sampling_config.get('top_k_categorical'),
            'property_type': posterior_sampling_config.get('property_type'),
            'nucleus': posterior_sampling_config.get('nucleus'),
            'lower_bound': posterior_sampling_config.get('lower_bound'),
            'upper_bound': posterior_sampling_config.get('upper_bound'),
            'none_distances': posterior_sampling_config.get('none_distances'),
            'x_theta_modifier_method': x_theta_config.get('method'),
            'num_x_theta_samples': x_theta_config.get('num_x_theta_samples'),
            'x_theta_num_local_searches': x_theta_config.get('num_local_searches'),
            'max_candidate_tokens': x_theta_config.get('max_candidate_tokens'),
            'top_k_values_for_local_search': x_theta_config.get('top_k_values_for_local_search'),
            'local_search_sampling_method': x_theta_config.get('local_search_sampling_method', 'top_p'),
            'locally_typical_alpha': x_theta_config.get('locally_typical_alpha', 0.0),
            'best_sequence_rank': x_theta_config.get('best_sequence_rank')
        },
        
        # Training config (needed even for inference)
        'training': {
            'ema': 0.9999,
            'antithetic_sampling': True,
            'importance_sampling': False,
            'sampling_eps': 1e-3,
            'change_of_variables': False,
            'compute_loss_on_pad_tokens': False,
            'use_simple_ce_loss': False,
            'guidance': None  # Add guidance field to training config
        },
        
        # Loader config
        'loader': {
            'global_batch_size': 512, # Not important here; will be overridden
            'eval_global_batch_size': 512, # Not important here; will be overridden
            'batch_size': 16,    # Not important here; will be overridden dynamically
            'eval_batch_size': 16,  # Not important here; will be overridden dynamically
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True
        },
        
        # Eval config - updated with checkpoint path
        'eval': {
            'checkpoint_path': checkpoint_path,
            'disable_ema': False,
            'generate_samples': True,
            'generated_samples_path': '',
            'max_samples': 50000,
            'compute_generative_perplexity': False,  # Required
            'perplexity_batch_size': 8,  # Required
            'compute_perplexity_on_sanity': False,  # Required
            'gen_ppl_eval_model_name_or_path': 'gpt2-large',  # Required
        },
        
        # Optimizer config (required by diffusion.py)
        'optim': {
            'weight_decay': 0,
            'lr': 3e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8
        },
        
        # LR Scheduler config (required by diffusion.py)
        'lr_scheduler': {
            '_target_': 'constant_warmup',
            'warmup_steps': 1000,
            'constant_lr': 3e-4
        },
        
        # Trainer config (minimal for inference)
        'trainer': {
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1 if torch.cuda.is_available() else 'auto',
            'precision': 'bf16' if torch.cuda.is_available() else '32'
        }
    })
    return config

def create_x_theta_config(args):
    """Create x_theta configuration from parsed arguments."""
    return {
            'method': args.x_theta_type,
            'num_x_theta_samples': args.num_x_theta_samples,
            'num_local_searches': args.x_theta_num_local_searches,
            'top_k_categorical': args.top_k_categorical,
            'max_num_local_search_iters': args.x_theta_max_num_local_search_iters,
            'property_type': args.property_type,
            'lower_bound': args.lower_bound,
            'upper_bound': args.upper_bound,
            'max_candidate_tokens': args.max_candidate_tokens,
            'top_k_values_for_local_search': args.top_k_values_for_local_search,
            'local_search_sampling_method': args.local_search_sampling_method,
            'locally_typical_alpha': args.locally_typical_alpha,
            'best_sequence_rank': args.best_sequence_rank
        }

def create_posterior_sampling_config(args):
    """Create posterior sampling configuration from parsed arguments."""
    return {
            'method': args.posterior_sampling_method,
            'num_posterior_samples': args.num_posterior_samples,
            'num_mcts_samples': args.num_mcts_samples,
            'num_local_searches': args.num_local_searches,
            'top_k_categorical': args.top_k_categorical,
            'nucleus': args.nucleus,
            'max_num_local_search_iters': args.max_num_local_search_iters,
            'property_type': args.property_type,
            'lower_bound': args.lower_bound,
            'upper_bound': args.upper_bound,
            'none_distances': args.none_distances
        }  # Standard method doesn't need special config

def load_toxicity_prefixes(prefix_dir, tokenizer, max_prefixes=None, start_index=0):
    """Load prefix texts from directory and tokenize them.
    
    Args:
        prefix_dir: Directory containing .txt files with prefixes
        tokenizer: Tokenizer to use for encoding
        max_prefixes: Maximum number of prefixes to load (None for all)
        start_index: Starting index for file selection (default: 0)
        
    Returns:
        List of tuples: [(prefix_text, prefix_token_ids), ...]
    """
    prefix_files = sorted([f for f in os.listdir(prefix_dir) if f.endswith('.txt')],
                         key=lambda x: int(x.split('.')[0]))
    
    # Apply start_index and max_prefixes slicing
    if start_index > 0:
        prefix_files = prefix_files[start_index:]
    
    if max_prefixes:
        prefix_files = prefix_files[:max_prefixes]
    
    print(f"Loading {len(prefix_files)} prefixes starting from index {start_index}")
    
    prefixes = []
    for filename in prefix_files:
        filepath = os.path.join(prefix_dir, filename)
        with open(filepath, 'r') as f:
            prefix_text = f.read().strip()
        
        # Tokenize without adding special tokens (they'll be added during generation)
        prefix_token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prefixes.append((prefix_text, prefix_token_ids))
    
    return prefixes

def generate_molecules_no_guidance(data, sequence_length, diffusion_steps, num_samples=100, batch_size=16, checkpoint_path=None, tokenizer_name_or_path=None, argmax_mode='none', posterior_sampling_config=None, x_theta_config=None, prefix_dir=None, start_sample_index=0, seed=42):
    """
    Generate molecules using the pre-trained UDLM model from local checkpoint without guidance
    
    Args:
        num_samples: Number of molecules to generate
        batch_size: Batch size for generation
        checkpoint_path: Path to the local checkpoint file
        argmax_mode: When to use argmax sampling ('none', 'last_step', 'all_steps')
        posterior_sampling_config: Configuration for posterior sampling method
        prefix_dir: Directory containing prefix files (for toxicity generation)
        start_sample_index: Starting index for prefix file selection (default: 0)
        seed: Random seed for reproducibility
    """
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Using batch size: {batch_size}")
    
    # Load toxicity prefixes if applicable
    prefixes = None
    if data == 'openwebtext-split' and prefix_dir:
        print(f"Loading toxicity prefixes from: {prefix_dir}")
        print(f"Starting from sample index: {start_sample_index}, generating {num_samples} samples")
        # Load tokenizer first to tokenize prefixes
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        prefixes = load_toxicity_prefixes(prefix_dir, gpt2_tokenizer, max_prefixes=num_samples, start_index=start_sample_index)
        print(f"Loaded {len(prefixes)} prefixes")
        # Update num_samples to match number of prefixes
        num_samples = len(prefixes)
    
    # Setup config without guidance but with optional x_theta modifications
    config = setup_config_no_guidance(data, sequence_length, diffusion_steps, checkpoint_path, tokenizer_name_or_path, argmax_mode, posterior_sampling_config, x_theta_config, seed)
    config.sampling.batch_size = min(batch_size, num_samples)  # Use provided batch_size
    config.sampling.num_sample_batches = (num_samples + config.sampling.batch_size - 1) // config.sampling.batch_size
    
    # Also update loader config to match
    config.loader.batch_size = config.sampling.batch_size
    config.loader.eval_batch_size = config.sampling.batch_size
    
    # Print posterior sampling info
    if posterior_sampling_config:
        method = posterior_sampling_config.get('method')
        print(f"Using posterior sampling method: {method}")
    else:
        print("Using standard posterior sampling")
    
    # Load tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    
    # Load model from checkpoint
    model = diffusion_search.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config, 
        logger=False,
        weights_only=False
    )
    
    # Move model to GPU if available
    model = model.to(device)
    model.eval()
    
    all_samples = []
    all_token_ids = []  # Store token IDs for consistent prefix removal
    prefix_idx = 0  # Track which prefix we're using
    
    for i in range(config.sampling.num_sample_batches):
        print(f"Generating batch {i+1}/{config.sampling.num_sample_batches}")
        
        # Prepare prefix conditioning for this batch if using toxicity data
        batch_prefixes = None
        if prefixes:
            batch_end = min(prefix_idx + config.sampling.batch_size, len(prefixes))
            batch_prefixes = prefixes[prefix_idx:batch_end]
            prefix_idx = batch_end
            
            # If this is the last batch and it's smaller, adjust batch size temporarily
            if len(batch_prefixes) < config.sampling.batch_size:
                actual_batch_size = len(batch_prefixes)
                print(f"  Adjusting batch size to {actual_batch_size} for last batch")
            else:
                actual_batch_size = config.sampling.batch_size
        
        # Generate one batch without guidance
        with torch.no_grad():
            if batch_prefixes:
                # Temporarily adjust batch size if needed
                original_batch_size = config.sampling.batch_size
                config.sampling.batch_size = len(batch_prefixes)
                # Pass prefixes to the model for conditioning
                samples = model.sample(prefixes=batch_prefixes)
                # Restore original batch size
                config.sampling.batch_size = original_batch_size
            else:
                samples = model.sample()
            
        # Store token IDs for consistent prefix removal later
        all_token_ids.append(samples.cpu())
        
        # Decode samples
        decoded_samples = tokenizer.batch_decode(samples)
        all_samples.extend(decoded_samples)
        
        if len(all_samples) >= num_samples:
            break
    
    # Clean up samples
    cleaned_samples = []
    for sample in all_samples[:num_samples]:
        if data == 'abc':
            # For toxicity: first find and truncate at <eos>, then remove other special tokens
            # Find the first <eos> token
            eos_pos = sample.find('<eos>')
            if eos_pos != -1:
                # Keep everything before <eos>
                sample = sample[:eos_pos]
            
            # Remove other special tokens
            cleaned = sample.replace('<bos>', '').replace('<pad>', '').replace('<mask>', '').replace('<unk>', '').replace('<cls>', '').replace('<sep>', '').replace('<reserved>', '').strip()
        else:
            # For other data types: just remove all special tokens
            cleaned = sample.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<mask>', '').replace('<unk>', '').replace('<cls>', '').replace('<sep>', '').replace('<reserved>', '').strip()
        
        if cleaned:
            cleaned_samples.append(cleaned)
    
    # Concatenate all token IDs
    all_token_ids_tensor = torch.cat(all_token_ids, dim=0)[:num_samples]
    
    # Return cleaned samples, raw samples, and token IDs
    return cleaned_samples, all_samples[:num_samples], all_token_ids_tensor