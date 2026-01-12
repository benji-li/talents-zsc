from gen_data import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to saved dataset')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)

    print(f"Loaded dataset with {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample agent: {sample['agent']}")
        print(f"Obs sequence shape: {len(sample['obs_sequence'])} frames, each of shape {sample['obs_sequence'][0].shape}")
        print(f"Action sequence length: {len(sample['action_sequence'])}")
        print(f"Partner action sequence length: {len(sample['partner_actions'])}")
    else:
        print("No samples in dataset!")

if __name__ == "__main__":
    main()