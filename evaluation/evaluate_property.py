import torch
import os
import sys
import argparse
from itertools import combinations
import json

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from properties.peptides import *
from properties.molecules import *
from properties.trna import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate properties")
    parser.add_argument('--data', type=str, required=True,
                           choices=['qm9', 'grampa', 'trna'],
                           help='Type of dataset to use')
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input file")
    parser.add_argument("--property_type", type=str, nargs='+', default=None, help="List of property types")
    parser.add_argument("--lower_bound", type=float, nargs='+', default=None, help="List of lower bounds for the property values")
    parser.add_argument("--upper_bound", type=float, nargs='+', default=None, help="List of upper bounds for the property values")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    data = args.data
    file_path = args.file_path
    property_type = args.property_type
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    save_path = os.path.splitext(file_path)[0] + f'_prop_calc/{property_type}/{lower_bound}/{upper_bound}/properties.json'
    # Create the directory for save_path if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Read sequences from file and filter unique ones
    with open(file_path, 'r') as f:
        sequences = set(line.strip() for line in f if line.strip())

    # Read filtered sequences from the file
    if data == 'qm9':
        filtered_sequences_path = os.path.join(project_root, 'data', 'QM9', 'qm9_canonical_smiles.txt')
    elif data == 'grampa':
        filtered_sequences_path = os.path.join(project_root, 'data', 'peptides', 'filtered_peptides_46.txt')
    elif data == 'trna':
        filtered_sequences_path = os.path.join(project_root, 'data', 'trna', 'final_sequences.txt')
    with open(filtered_sequences_path, 'r') as f:
        filtered_sequences = set(line.strip() for line in f if line.strip())

    # Filter out sequences that are not novel
    novel_sequences = sequences - filtered_sequences
    print(f"Number of novel sequences: {len(novel_sequences)}")

    # Initialize counters for each property
    property_satisfy_counts = {prop: 0 for prop in property_type}

    # Store results for subsets of properties
    subset_satisfy_counts = {}

    # Calculate properties for each sequence
    sequence_results = []  # Store results for each sequence
    for sequence in novel_sequences:
        sequence_result = {}
        for i, prop in enumerate(property_type):
            calc_func = {
                "logp": calc_logp,
                "qed": calc_qed,
                "sa": calc_sa,
                "valid": calc_validity,
                "peptide_length": calculate_peptide_length,
                "hydrophobic_ratio": calculate_hydrophobic_ratio,
                "net_charge": calculate_net_charge,
                "trna_length": calc_trna_length,
                "acceptor_stem": count_acceptor_stem,
                "t_loop": get_t_loop_score,
                "fold": check_fold_structure_constraints,
            }.get(prop)

            if calc_func is None:
                raise ValueError(f"Unsupported property type: {prop}")

            # Calculate property value
            value = calc_func(sequence)

            # Skip if value is None
            if value is None:
                sequence_result[prop] = False
                continue

            # Check if the value satisfies the constraints
            satisfies = lower_bound[i] <= value <= upper_bound[i]
            if satisfies:
                property_satisfy_counts[prop] += 1

            sequence_result[prop] = satisfies
        sequence_results.append(sequence_result)

    # Check subsets of properties
    for k in range(1, len(property_type) + 1):
        for subset in combinations(property_type, k):
            subset_key = tuple(subset)
            subset_satisfy_counts[subset_key] = 0

            for sequence_result in sequence_results:
                if all(sequence_result[prop] for prop in subset):
                    subset_satisfy_counts[subset_key] += 1

    # Print results for individual properties
    for prop, count in property_satisfy_counts.items():
        print(f"Property '{prop}': {count} sequences satisfy the constraints")

    # Print results for subsets of properties
    for subset, count in subset_satisfy_counts.items():
        subset_str = ', '.join(subset)
        print(f"Subset '{subset_str}': {count} sequences satisfy all constraints")

    # Save results to JSON file
    results = {
        "property_satisfy_counts": property_satisfy_counts,
        "subset_satisfy_counts": {str(list(subset)): count for subset, count in subset_satisfy_counts.items()},
        "novel_sequences_count": len(novel_sequences)
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {save_path}")
