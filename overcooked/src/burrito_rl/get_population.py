import os
import shutil
import argparse
import csv
import json

def extract_policies(base_path, checkpoint_numbers, population_size, output_dir):
    """
    Extract policies from multiple checkpoints and rename them sequentially in a new folder.
    Also creates a mapping file to track the relationship between original and new policy names.
    
    Args:
        base_path (str): Base path to the policy parameters folder
        checkpoint_numbers (list): List of checkpoint numbers to extract policies from
        population_size (int): Number of policies per checkpoint (n)
        output_dir (str): Output directory for the consolidated policies
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for new policy naming
    new_policy_counter = 1
    
    # Create mapping data structures
    mapping_list = []  # For CSV
    mapping_dict = {}  # For JSON
    
    for checkpoint_num in checkpoint_numbers:
        checkpoint_folder = f"checkpoint_{checkpoint_num:06d}"
        checkpoint_path = os.path.join(base_path, checkpoint_folder, "policies")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Path {checkpoint_path} does not exist. Skipping...")
            continue
        
        print(f"Processing checkpoint: {checkpoint_num}")
        
        # For each policy in the checkpoint
        for pol_idx in range(1, population_size + 1):
            src_policy_folder = os.path.join(checkpoint_path, f"pol{pol_idx}")
            
            if not os.path.exists(src_policy_folder):
                print(f"Warning: Policy folder {src_policy_folder} does not exist. Skipping...")
                continue
            
            # Destination policy folder with the new name
            dst_policy_folder = os.path.join(output_dir, f"pol{new_policy_counter}")
            
            # Copy the policy folder to the output directory with the new name
            print(f"  Copying {src_policy_folder} -> {dst_policy_folder}")
            shutil.copytree(src_policy_folder, dst_policy_folder, dirs_exist_ok=True)
            
            # Record the mapping
            mapping_entry = {
                "checkpoint_number": checkpoint_num,
                "original_policy": f"pol{pol_idx}",
                "new_policy": f"pol{new_policy_counter}"
            }
            mapping_list.append(mapping_entry)
            
            mapping_key = f"pol{new_policy_counter}"
            mapping_dict[mapping_key] = {
                "checkpoint_number": checkpoint_num,
                "original_policy": f"pol{pol_idx}"
            }
            
            # Increment the counter
            new_policy_counter += 1
    
    # Write mapping to CSV file
    csv_path = os.path.join(output_dir, "policy_mapping.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['new_policy', 'checkpoint_number', 'original_policy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in mapping_list:
            writer.writerow(entry)
    
    # Write mapping to JSON file
    json_path = os.path.join(output_dir, "policy_mapping.json")
    with open(json_path, 'w') as jsonfile:
        json.dump(mapping_dict, jsonfile, indent=2)
    
    # Write mapping to text file (more human-readable)
    txt_path = os.path.join(output_dir, "policy_mapping.txt")
    with open(txt_path, 'w') as txtfile:
        txtfile.write("Policy Mapping Information\n")
        txtfile.write("==========================\n\n")
        txtfile.write(f"Total policies extracted: {new_policy_counter - 1}\n\n")
        txtfile.write("Format: New Policy Name -> Checkpoint Number, Original Policy Name\n\n")
        
        for i in range(1, new_policy_counter):
            policy_key = f"pol{i}"
            checkpoint = mapping_dict[policy_key]["checkpoint_number"]
            orig_policy = mapping_dict[policy_key]["original_policy"]
            txtfile.write(f"{policy_key} -> Checkpoint {checkpoint}, {orig_policy}\n")
    
    print(f"Extraction complete. {new_policy_counter - 1} policies extracted to {output_dir}")
    print(f"Mapping files created: policy_mapping.csv, policy_mapping.json, and policy_mapping.txt")

def main():
    parser = argparse.ArgumentParser(description="Extract and rename policy folders from multiple checkpoints")
    parser.add_argument("--base-path", required=True, help="Base path to the policy parameters folder")
    parser.add_argument("--checkpoints", required=True, type=str, help="Comma-separated list of checkpoint numbers (e.g., '60,120,210')")
    parser.add_argument("--population-size", required=True, type=int, help="Number of policies per checkpoint (n)")
    parser.add_argument("--output-dir", required=True, help="Output directory for the consolidated policies")
    
    args = parser.parse_args()
    
    # Parse the checkpoint numbers
    checkpoint_numbers = [int(num.strip()) for num in args.checkpoints.split(",")]
    
    extract_policies(
        args.base_path,
        checkpoint_numbers,
        args.population_size,
        args.output_dir
    )

if __name__ == "__main__":
    main()