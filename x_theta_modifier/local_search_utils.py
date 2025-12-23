import random
import time  # Import time module for timing functionality
import math  # Import math module for infinity
# Add RDKit SA_Score to path for sascorer import
from properties.peptides import *
from properties.molecules import *
from properties.trna import *
from properties.property_util import *

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("yairschiff/qm9-tokenizer", trust_remote_code=True)

special_tokens = ['<bos>', '<eos>', '<mask>', '<pad>', '<unk>']

def local_search_qm9(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens):
    # Initialize best molecule
    tokens = fix_basic_syntax_tokens(tokens[:])
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"

    for i, token in enumerate(tokens):
        if token in atom_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, atom_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'atom substitution')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, atom_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'atom insertion')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def local_search_grampa(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens):
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"

    for i, token in enumerate(tokens):
        if token in aa_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, aa_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'amino acid substitution')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, aa_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'amino acid insertion')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found


    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def local_search_trna(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens):
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"
    if is_found:
        return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
    

    for i, token in enumerate(tokens):
        if token in nucleotide_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, nucleotide_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'nucleotide substitution')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, nucleotide_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'nucleotide insertion')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found


    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, operator_type):
    is_max_num_local_search_iters = False
    is_found = False
    num_candidate_tokens = 0
    num_local_search_iters += 1
    random.shuffle(candidates)

    for replacement in candidates:
        if token == replacement: # Skip replacing a token with itself
            continue
        num_candidate_tokens += 1
        # Replace the token with the candidate
        new_tokens = tokens[:]
        new_tokens[i] = replacement
        new_sequence = ''.join(token for token in new_tokens if token not in special_tokens)
        new_prop_values = [calc(new_sequence) for calc in property_calcs]
        new_distances = [dist(value) for dist, value in zip(distance_to_bounds, new_prop_values)]

        # Check if the new molecule is better
        if compare_hierarchical(new_distances, best_distances) < 0:
            operator = operator_type
            best_tokens = new_tokens
            best_prop_values = new_prop_values
            best_distances = new_distances
            if all(distance == 0 for distance in best_distances):
                is_found = True
                return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters
        # Break if maximum iterations are reached
        if num_local_search_iters >= max_num_local_search_iters:
            is_max_num_local_search_iters = True
            return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters
        if num_candidate_tokens >= max_candidate_tokens:
            break # Only try `max_candidate_tokens` replacements per position

    return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters

def local_search_on_best_tokens(best_tokens, tokenizer, num_local_searches, distance_to_bounds, property_calcs, local_search, device, max_candidate_tokens):
    # Iterate through each best_token in best_tokens
    modified_tokens = []
    for best_token_index, best_token in enumerate(best_tokens):
        # Move best_token to CPU for RDKit and tokenizer operations
        best_token_cpu = best_token.cpu()
        # shape = best_token_cpu.shape
        # best_token_cpu = torch.arange(1, shape[0]+1, dtype=best_token_cpu.dtype, device=best_token_cpu.device)

        # Convert token IDs to tokens directly to preserve length
        sequence_string = tokenizer.decode(best_token_cpu, skip_special_tokens=False)
        modified_token = tokenizer.tokenize(sequence_string)
        # modified_token = tokenizer.convert_ids_to_tokens(best_token_cpu.tolist())

        max_num_local_search_iters = 1e9
        index_local_search = 0
        num_local_search_iters = 0
        is_found = False

        while index_local_search < num_local_searches and num_local_search_iters < max_num_local_search_iters and not is_found:
            new_best_tokens, best_prop_values, num_local_search_iters, operator, is_found = local_search(
                tokens=modified_token,
                distance_to_bounds=distance_to_bounds,
                property_calcs=property_calcs,
                num_local_search_iters=num_local_search_iters,
                max_num_local_search_iters=max_num_local_search_iters,
                max_candidate_tokens=max_candidate_tokens
            )
            index_local_search += 1
            if new_best_tokens == modified_token:
                break
            else:
                modified_token = new_best_tokens[:]

        modified_token_ids = tokenizer.convert_tokens_to_ids(modified_token)
        # --- Append to the list of modified tokens ---
        modified_tokens.append(modified_token_ids)

    # Stack into a tensor with proper device
    best_tokens = torch.tensor(modified_tokens, dtype=torch.long, device=device)
    return best_tokens

if __name__ == "__main__":
    smiles = "C1=CC=CC=)C1O"
    import ipdb; ipdb.set_trace()