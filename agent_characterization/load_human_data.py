import os
import json
import argparse

from gen_data import HumanTrajectoryDataset, save_dataset

def load_all_human_jsons(json_dir):
    """Load multiple human JSON trajectories from a directory"""
    all_trajs = []
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(json_dir, fname)
            with open(fpath, 'r') as f:
                traj = json.load(f)
                all_trajs.append(traj)
    print(f"Loaded {len(all_trajs)} human trajectory files from {json_dir}")
    return all_trajs

def main():
    parser = argparse.ArgumentParser(description="Load human trajectories and save dataset")
    parser.add_argument('--human-json-dir', type=str, required=True, help='Directory with human JSON files')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to save the dataset')
    parser.add_argument('--layout', type=str, default="hallway", help='Environment layout')
    parser.add_argument('--encoder', type=str, default="lossless", help='Observation encoder type')
    parser.add_argument('--action-dim', type=int, default=27, help='Number of discrete actions')
    args = parser.parse_args()

    trajectories = load_all_human_jsons(args.human_json_dir)

    dataset = HumanTrajectoryDataset(trajectories, action_dim=args.action_dim,
                                     encoder=args.encoder, layout=args.layout)
    
    print(f"Processed human dataset with {len(dataset)} samples")
    if len(dataset) > 0:
        print(f"First sample obs shape: {dataset[0]['obs_sequence'][0].shape}")

    save_dataset(dataset, args.dataset_path)

if __name__ == "__main__":
    main()
