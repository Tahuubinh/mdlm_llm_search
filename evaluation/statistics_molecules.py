from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit import RDLogger
import json
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import RDConfig
import sys
import os
import argparse
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # type: ignore

# Function to calculate logP
def calc_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)

# Function to calculate QED (Quantitative Estimate of Drug-likeness)
def calc_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol)

def calc_sa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)

def read_smiles(file_path):
    """Read SMILES strings from a file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def canonicalize_smiles(smiles_list):
    """Convert a list of SMILES strings to canonical SMILES."""
    canonical_smiles = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles.add(Chem.MolToSmiles(mol))
    return canonical_smiles

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Calculate statistics for molecules.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing molecule data.")
    parser.add_argument('--strategy_name', type=str, default="local_search_hierarchical_branch_['valid', 'sa', 'qed']_lb[0.5, -1000.0, 1.0]_ub[1000.0, 3.0, 1000.0]", help="Name of the strategy used for molecule generation.")
    parser.add_argument('--molecules', action='store_true', help="Boolean flag to indicate molecule processing.")
    parser.add_argument('--sa_upperbound', type=float, default=3.0, help="Upper bound for synthetic accessibility (SA) score.")

    args = parser.parse_args()
    folder_path = args.folder_path
    strategy_name = args.strategy_name
    is_molecules = args.molecules
    sa_upperbound = args.sa_upperbound

    strategy_name ="local_search_hierarchical_branch_['valid', 'sa', 'qed']_lb[0.5, -1000.0, 1.0]_ub[1000.0, 3.0, 1000.0]"
    local_search_file_name =f"modified_smiles_{strategy_name}"

    if is_molecules:
        strategy_name = "molecules"
        local_search_file_name =f"molecules"

    local_search_file = f"{folder_path}/{local_search_file_name}.txt"
    local_search_info_file = f"{folder_path}/final_info_{strategy_name}.json"
    
    qm9_file = 'data/QM9/qm9_canonical_smiles.txt'

    # Calculate average time_taken_seconds
    # local_search_info_file = "sample_results/generated_molecules_no_guidance_multi_gumbel_reward_qed_score/final_info_local_search_hierarchical_['valid', 'sa', 'qed']_lb[0.5, -1000.0, 1.0]_ub[1000.0, 3.0, 1000.0].json"
    # Statistic to a JSON file
    statistic_file = f"{folder_path}/{local_search_file_name}_statistic.json"
    try:
        with open(local_search_info_file, 'r') as file:
            local_search_info = json.load(file)

        time_taken_values = [entry['time_taken_seconds'] for entry in local_search_info if 'time_taken_seconds' in entry]
        if time_taken_values:
            sum_time_taken = sum(time_taken_values)
            avg_time_taken = sum(time_taken_values) / len(time_taken_values)
            print(f"Total time taken (seconds) for local search: {sum_time_taken}")
            print(f"Average time taken (seconds): {avg_time_taken}")    
    except FileNotFoundError:
        print(f"Info file {local_search_info_file} not found. Skipping time taken statistics.")
        sum_time_taken = None
        avg_time_taken = None

    # Read SMILES strings from files
    local_search_smiles = read_smiles(local_search_file)
    qm9_smiles = read_smiles(qm9_file)

    # Calculate number of valid molecules in local search results
    valid_smiles = [smiles for smiles in local_search_smiles if Chem.MolFromSmiles(smiles) is not None]
    num_valid = len(valid_smiles)
    print(f"Number of valid molecules in local search results: {num_valid}")
    
    # Convert to canonical SMILES and create sets
    local_search_set = canonicalize_smiles(valid_smiles)
    qm9_set = set(qm9_smiles)
    
    # Filter molecules not in QM9
    filtered_smiles = local_search_set - qm9_set
    
    # Count number of valid and novel molecules
    num_valid_novel = len(filtered_smiles)

    print(f"Number of valid and novel molecules: {num_valid_novel}")

    # Calculate average SA for valid and novel molecules
    sa_values = []
    for smiles in filtered_smiles:
        sa = calc_sa(smiles)
        if sa is not None:
            sa_values.append(sa)
    # sa_upperbound = 3
    sa_binary_list = [1 if sa <= sa_upperbound else 0 for sa in sa_values]
    if sa_values:
        avg_sa = sum(sa_values) / len(sa_values)
        print(f"Average SA for valid and novel molecules: {avg_sa}")

    num_sa_below_upperbound = sum(sa_binary_list)
    print(f'Total number of SA values smaller than or equal to {sa_upperbound}: {num_sa_below_upperbound}')
    print(f'Proportion of SA values smaller than or equal to {sa_upperbound}: {num_sa_below_upperbound / len(sa_values)}')

    # Print min and max SA values for valid and novel molecules
    if sa_values:
        min_sa = min(sa_values)
        max_sa = max(sa_values)
        print(f"Minimum SA for valid and novel molecules: {min_sa}")
        print(f"Maximum SA for valid and novel molecules: {max_sa}")

    if len(sa_binary_list) > 64:
        sorted_sa_binary_list = sorted(sa_binary_list, reverse=True)[:64]
        avg_sa_top64 = sum(sorted_sa_binary_list) / len(sorted_sa_binary_list)
        print(f"Average SA for top 64 valid and novel molecules: {avg_sa_top64}")

    # Calculate average SA for top 96 valid and novel molecules
    if len(sa_binary_list) > 96:
        sorted_sa_binary_list = sorted(sa_binary_list, reverse=True)[:96]
        avg_sa_top96 = sum(sorted_sa_binary_list) / len(sorted_sa_binary_list)
        print(f"Average SA for top 96 valid and novel molecules: {avg_sa_top96}")

    # Calculate average SA for top 241 valid and novel molecules
    if len(sa_binary_list) > 241:
        sorted_sa_binary_list = sorted(sa_binary_list, reverse=True)[:241]
        avg_sa_top241 = sum(sorted_sa_binary_list) / len(sorted_sa_binary_list)
        print(f"Average SA for top 241 valid and novel molecules: {avg_sa_top241}")
    else:
        avg_sa_top241 = -1

    # Filter molecules with SA <= sa_upperbound for QED calculation
    filtered_smiles_with_sa = []
    for smiles in filtered_smiles:
        sa = calc_sa(smiles)
        if sa is not None and sa <= sa_upperbound:
            filtered_smiles_with_sa.append(smiles)
    
    print(f"Number of valid and novel molecules with SA <= {sa_upperbound}: {len(filtered_smiles_with_sa)}")

    # Calculate average QED for valid and novel molecules with SA <= sa_upperbound
    qed_values = []
    for smiles in filtered_smiles_with_sa:
        qed = calc_qed(smiles)
        if qed is not None:
            qed_values.append(qed)
    if qed_values:
        avg_qed = sum(qed_values) / len(qed_values)
        print(f"Average QED for valid and novel molecules with SA <= {sa_upperbound}: {avg_qed}")

    # Print min and max QED values for valid and novel molecules
    if qed_values:
        min_qed = min(qed_values)
        max_qed = max(qed_values)
        print(f"Minimum QED for valid and novel molecules: {min_qed}")
        print(f"Maximum QED for valid and novel molecules: {max_qed}")

    # print(len(qed_values))
    # Calculate average QED for top 64 valid and novel molecules
    if len(qed_values) > 64:
        sorted_qed_values = sorted(qed_values, reverse=True)[:64]
        avg_qed_top64 = sum(sorted_qed_values) / len(sorted_qed_values)
        print(f"Average QED for top 64 valid and novel molecules: {avg_qed_top64}")

    # Calculate average QED for top 96 valid and novel molecules
    if len(qed_values) > 96:
        sorted_qed_values = sorted(qed_values, reverse=True)[:96]
        avg_qed_top96 = sum(sorted_qed_values) / len(sorted_qed_values)
        print(f"Average QED for top 96 valid and novel molecules: {avg_qed_top96}")

    # Calculate average QED for top 241 valid and novel molecules
    if len(qed_values) > 241:
        sorted_qed_values = sorted(qed_values, reverse=True)[:241]
        avg_qed_top241 = sum(sorted_qed_values) / len(sorted_qed_values)
        print(f"Average QED for top 241 valid and novel molecules: {avg_qed_top241}")
    else:
        avg_qed_top241 = -1


    statistics = {
        "sum_time_taken_seconds": sum_time_taken,
        "avg_time_taken_seconds": avg_time_taken,
        "num_valid": num_valid,
        "num_valid_novel": num_valid_novel,
        "avg_qed": avg_qed,
        "min_qed": min_qed,
        "max_qed": max_qed,
        "avg_qed_top64": avg_qed_top64,
        "avg_qed_top96": avg_qed_top96,
        "avg_qed_top241": avg_qed_top241,
        "avg_sa": avg_sa ,
        "num_sa_below_upperbound": num_sa_below_upperbound,
        "proportion_sa_below_upperbound": num_sa_below_upperbound / len(sa_values) if sa_values else None,
        "min_sa": min_sa,
        "max_sa": max_sa,
        "avg_sa_top64": avg_sa_top64,
        "avg_sa_top96": avg_sa_top96,
        "avg_sa_top241": avg_sa_top241
    }
    with open(statistic_file, 'w') as json_file:
        json.dump(statistics, json_file, indent=4)
    print(f"Statistics saved to {statistic_file}")


if __name__ == "__main__":
    main()