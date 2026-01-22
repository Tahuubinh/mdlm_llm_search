#!/usr/bin/env python3
"""
Simple script to inference molecules using UDLM model without guidance
Now enhanced with optional x_theta modification support
"""

import torch
import os
import sys
import argparse
import numpy as np
import lightning as L
from inference_utils import create_x_theta_config, create_posterior_sampling_config, generate_molecules_no_guidance
import time


def parse_arguments():
    """Parse command-line arguments for molecule generation."""
    parser = argparse.ArgumentParser(description="Generate molecules with optional x_theta modifications")
    parser.add_argument('--data', type=str, required=True,
                           choices=['openwebtext-split'],
                           help='Type of dataset to use')
    # Basic generation parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of molecules to generate (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for generation')
    parser.add_argument('--version', type=str, default='1',
                       help='Version of the model or configuration (default: None)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Posterior sampling method parameters
    parser.add_argument('--posterior_sampling_method', type=str, default='standard',
                       choices=['standard', 'mcts'],)
    parser.add_argument('--num_posterior_samples', type=int, default=32,
                       help='Number of samples for posterior sampling method')
    parser.add_argument('--num_mcts_samples', type=int, default=128,
                       help='Number of samples for Monte Carlo Tree Search sampling method')
    parser.add_argument("--max_num_local_search_iters", type=int, default=1000000000, help="Maximum number of local search iterations")
    parser.add_argument("--num_local_searches", type=int, default=30, help="Number of local searches to perform")
    parser.add_argument('--top_k_categorical', type=int, default=None,
                       help='Top K categorical sampling for local search methods')
    parser.add_argument("--property_type", type=str, nargs='+', default=["qed"], help="List of property types")
    parser.add_argument("--lower_bound", type=float, nargs='+', default=None, help="List of lower bounds for the property values")
    parser.add_argument("--upper_bound", type=float, nargs='+', default=None, help="List of upper bounds for the property values")
    parser.add_argument("--none_distances", type=float, nargs='+', default=None, help="List of none distances for the property values")
    parser.add_argument('--argmax_mode', type=str, default='none',
                       choices=['none', 'last_step', 'all_steps'],
                       help='When to use argmax sampling: none (never, use categorical), last_step (only at final step), all_steps (at every diffusion step) (default: none)')
    parser.add_argument('--nucleus', type=float, default=None,
                       help='Nucleus sampling probability (default: None)')

    # X-theta modification parameters
    parser.add_argument('--x_theta_type', type=str, default='standard',
                       choices=['standard', 'bon', 'bon_localsearch', 'bon_localsearch_laststep', 'local_search_language'],
                       help='Type of x_theta modification to apply')
    parser.add_argument('--num_x_theta_samples', type=int, default=1,
                       help='Number of samples for x_theta sampling method')
    parser.add_argument("--x_theta_max_num_local_search_iters", type=int, default=10000000, help="Maximum number of local search iterations for x_theta")
    parser.add_argument("--x_theta_num_local_searches", type=int, default=30, help="Number of local searches to perform for x_theta")
    parser.add_argument("--max_candidate_tokens", type=int, default=1000, help="Number of candidate tokens for local search")
    parser.add_argument("--top_k_values_for_local_search", type=int, default=10, 
                       help="For language data: number of top-k tokens to try per position in local search")
    parser.add_argument("--local_search_sampling_method", type=str, default="top_p", 
                       choices=["top_p", "locally_typical", "locally_typical_distance"],
                       help="Sampling method for local search: 'top_p', 'locally_typical' (additive bias), or 'locally_typical_distance' (entropy scaling)")
    parser.add_argument("--locally_typical_alpha", type=float, default=0.0,
                       help="Bias parameter for locally typical methods. For 'locally_typical': additive bias (0.0=pure, >0=bias high prob). For 'locally_typical_distance': entropy scaling (1.0=pure, <1.0=bias high prob). Default: 0.0 for additive, 1.0 for tau")
    parser.add_argument("--best_sequence_rank", type=int, default=1,
                       help="Select the sequence with the Nth smallest distance (1=best, 2=second best, 3=third best, etc.). Default: 1")
    
    # Toxicity-specific parameters
    parser.add_argument('--prefix_dir', type=str, default='data/toxicity/1000_samples',
                       help='Directory containing prefix text files for toxicity generation')
    parser.add_argument('--start_sample_index', type=int, default=0,
                       help='Starting index for prefix file selection (e.g., 10 to start from 10.txt)')
    parser.add_argument('--diffusion_steps', type=int, default=250,
                       help='Number of diffusion steps for molecule generation (default: 250)')

    return parser.parse_args()

def main():
    """Main function to demonstrate molecule generation with optional x_theta modifications"""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility (following qm9_eval.py approach)
    L.seed_everything(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    data = args.data
    posterior_sampling_method = args.posterior_sampling_method
    property_type = args.property_type  
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    top_k_categorical = args.top_k_categorical
    x_theta_type = args.x_theta_type
    nucleus = args.nucleus
    max_candidate_tokens = args.max_candidate_tokens
    if data == 'qm9':
        checkpoint_path = 'outputs/QM9/mdlm/best.ckpt'
        tokenizer_name_or_path = 'yairschiff/qm9-tokenizer'
        sequence_length = 32
        diffusion_steps = 32
    elif data == 'grampa':
        checkpoint_path = 'outputs/grampa/mdlm_no-guidance_padTrue/checkpoints/best.ckpt'
        tokenizer_name_or_path = 'grampa'
        sequence_length = 50
        diffusion_steps = 32
    elif data == 'trna':
        checkpoint_path = 'outputs/trna/mdlm_no-guidance_full/checkpoints/best.ckpt'
        tokenizer_name_or_path = 'trna'
        sequence_length = 110
        diffusion_steps = 32
    elif data == 'openwebtext-split':
        checkpoint_path = 'outputs/toxicity/mdlm/best.ckpt'
        tokenizer_name_or_path = 'gpt2'
        sequence_length = 100
        diffusion_steps = args.diffusion_steps  # Can adjust this

    
    # Generate molecules
    print("Generating molecules without guidance (standard sampling)...")
    
    # Create posterior sampling configuration
    posterior_sampling_config = create_posterior_sampling_config(args)
    x_theta_config = create_x_theta_config(args)
    start_time = time.time()
    molecules, raw_molecules, token_ids = generate_molecules_no_guidance(
        data=data,
        sequence_length=sequence_length,
        diffusion_steps=diffusion_steps,
        num_samples=args.num_samples, 
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        argmax_mode=args.argmax_mode,
        posterior_sampling_config=posterior_sampling_config,
        x_theta_config=x_theta_config,
        prefix_dir=args.prefix_dir if data == 'openwebtext-split' else None,
        start_sample_index=args.start_sample_index,
        seed=args.seed
    )
    end_time = time.time()
    
    if not top_k_categorical:
        top_k_categorical = 'all'
    if not nucleus:
        nucleus = 'all'
    # Determine output filename
    output_folder = f"sample_results/{data}/topk_{top_k_categorical}_nucleus_{nucleus}"
    # if args.version:
    #     output_folder = f"{output_folder}_{args.version}"

    folder_name_suffix = f"{posterior_sampling_method}/{x_theta_type}/{property_type}_lb{lower_bound}_ub{upper_bound}"
    if "localsearch" in posterior_sampling_method:
        folder_name_suffix = f"{posterior_sampling_method}{args.num_local_searches}_{property_type}_lb{lower_bound}_ub{upper_bound}"
    elif "localsearch" in x_theta_type:
        folder_name_suffix = f"{posterior_sampling_method}/{x_theta_type}_{args.x_theta_num_local_searches}_{max_candidate_tokens}/{property_type}_lb{lower_bound}_ub{upper_bound}"

    mol_folder = f"{output_folder}/{folder_name_suffix}/rank_{args.best_sequence_rank}/{args.local_search_sampling_method}/{args.locally_typical_alpha}/{args.version}/seed_{args.seed}"

    # Save molecules based on data type
    if data == 'openwebtext-split':
        # For toxicity, load prefixes to determine what to remove
        from transformers import GPT2Tokenizer
        from inference_utils import load_toxicity_prefixes
        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        prefixes = load_toxicity_prefixes(args.prefix_dir, gpt2_tokenizer, max_prefixes=args.num_samples, start_index=args.start_sample_index)
        
        # Save each output as separate file (post-prefix only)
        # File index matches prefix index (e.g., if start_sample_index=10, saves as 10.txt, 11.txt, ...)
        # CRITICAL: Use token IDs directly from generation, not re-encoded text!
        # This ensures consistency with property evaluation during generation.
        molecules_dir = f"{mol_folder}/molecules"
        os.makedirs(molecules_dir, exist_ok=True)
        for i, (mol, mol_token_ids) in enumerate(zip(molecules, token_ids)):
            # Calculate the actual file index based on start_sample_index
            file_idx = args.start_sample_index + i
            
            # Get the prefix length for this sample (in tokens)
            if i < len(prefixes):
                prefix_text, prefix_token_ids = prefixes[i]
                prefix_len = len(prefix_token_ids)
                
                # Use token IDs directly from generation (no re-encoding!)
                # Slice tokens to remove prefix
                if prefix_len > 0 and prefix_len < len(mol_token_ids):
                    post_prefix_tokens = mol_token_ids[prefix_len:].tolist()
                    # Decode only the post-prefix part
                    post_prefix_text = gpt2_tokenizer.decode(post_prefix_tokens).strip()
                else:
                    # Fallback: if prefix >= full length, use empty string
                    # Or if no prefix, use full text
                    post_prefix_text = mol if prefix_len == 0 else ""
            else:
                post_prefix_text = mol
            
            mol_file_path = f"{molecules_dir}/{file_idx}.txt"
            with open(mol_file_path, 'w') as f:
                f.write(post_prefix_text)
        print(f"\nGenerated molecules (post-prefix only) saved to: {molecules_dir}/ ({len(molecules)} files, indices {args.start_sample_index}-{args.start_sample_index + len(molecules) - 1})")
        
        # Save raw molecules (post-prefix only)
        # CRITICAL: Use token IDs directly, same as cleaned molecules
        raw_molecules_dir = f"{mol_folder}/raw_molecules"
        os.makedirs(raw_molecules_dir, exist_ok=True)
        for i, (mol, mol_token_ids) in enumerate(zip(raw_molecules, token_ids)):
            # Calculate the actual file index based on start_sample_index
            file_idx = args.start_sample_index + i
            
            # Get the prefix length for this sample (in tokens)
            if i < len(prefixes):
                prefix_text, prefix_token_ids = prefixes[i]
                prefix_len = len(prefix_token_ids)
                
                # Use token IDs directly from generation (no re-encoding!)
                # Slice tokens to remove prefix
                if prefix_len > 0 and prefix_len < len(mol_token_ids):
                    post_prefix_tokens = mol_token_ids[prefix_len:].tolist()
                    # Decode only the post-prefix part
                    post_prefix_text = gpt2_tokenizer.decode(post_prefix_tokens).strip()
                else:
                    # Fallback: if prefix >= full length, use empty string
                    # Or if no prefix, use full text
                    post_prefix_text = mol if prefix_len == 0 else ""
            else:
                post_prefix_text = mol
            
            raw_mol_file_path = f"{raw_molecules_dir}/{file_idx}.txt"
            with open(raw_mol_file_path, 'w') as f:
                f.write(post_prefix_text)
        print(f"\nGenerated raw molecules (post-prefix only) saved to: {raw_molecules_dir}/ ({len(raw_molecules)} files, indices {args.start_sample_index}-{args.start_sample_index + len(raw_molecules) - 1})")
        
        # Save full molecules (with prefix)
        full_molecules_dir = f"{mol_folder}/full_molecules"
        os.makedirs(full_molecules_dir, exist_ok=True)
        for i, mol in enumerate(molecules):
            file_idx = args.start_sample_index + i
            full_mol_file_path = f"{full_molecules_dir}/{file_idx}.txt"
            with open(full_mol_file_path, 'w') as f:
                f.write(mol)
        print(f"\nGenerated full molecules (with prefix) saved to: {full_molecules_dir}/ ({len(molecules)} files, indices {args.start_sample_index}-{args.start_sample_index + len(molecules) - 1})")
        
        # Save full raw molecules (with prefix)
        full_raw_molecules_dir = f"{mol_folder}/full_raw_molecules"
        os.makedirs(full_raw_molecules_dir, exist_ok=True)
        for i, mol in enumerate(raw_molecules):
            file_idx = args.start_sample_index + i
            full_raw_mol_file_path = f"{full_raw_molecules_dir}/{file_idx}.txt"
            with open(full_raw_mol_file_path, 'w') as f:
                f.write(mol)
        print(f"\nGenerated full raw molecules (with prefix) saved to: {full_raw_molecules_dir}/ ({len(raw_molecules)} files, indices {args.start_sample_index}-{args.start_sample_index + len(raw_molecules) - 1})")
    else:
        # For other data types, save all to single file
        mol_file = f"{mol_folder}/molecules.txt"
        os.makedirs(os.path.dirname(mol_file), exist_ok=True)
        with open(mol_file, 'w') as mol_file:
            for i, mol in enumerate(molecules):
                mol_file.write(f"{mol}\n")
        print(f"\nGenerated molecules saved to: {mol_file.name}")

        raw_mol_file = f"{mol_folder}/raw_molecules.txt"
        os.makedirs(os.path.dirname(raw_mol_file), exist_ok=True)
        with open(raw_mol_file, 'w') as raw_mol_file:
            for i, mol in enumerate(raw_molecules):
                raw_mol_file.write(f"{mol}\n")
        print(f"\nGenerated raw molecules saved to: {raw_mol_file.name}")

    time_file = f"{mol_folder}/generation_time/start_index_{args.start_sample_index}_num_samples_{args.num_samples}.txt"
    os.makedirs(os.path.dirname(time_file), exist_ok=True)
    with open(time_file, 'w') as time_file:
       time_file.write(f"Generation time: {end_time - start_time} seconds\n")

    print(f"\nGeneration time: {end_time - start_time} seconds")
    
    # Save all arguments to a file
    params_file = f"{mol_folder}/params.txt"
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    with open(params_file, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"\nParameters saved to: {params_file}")

if __name__ == "__main__":
    main()