from agent_characterization.analysis.analyze_vae import preprocess_data, analyze_role_representations,analyze_action_patterns, eval_vae_accuracy, load_vae #, compute_human_agent_js_divergence, compute_human_agent_js_divergence_simple
from agent_characterization.gen_data import load_dataset, ChunkedTrajectoryDataset
from agent_characterization.clustering.kmeans import analyze_kmeans_clusters
import os
import matplotlib.pyplot as plt
import argparse

def eval_and_cluster(vae, dataset, save_path, batch_size,clusters=[8,20]):
    info = preprocess_data(vae, dataset, save_path, batch_size)
    accuracy, cluster_labels, gaussians = analyze_kmeans_clusters(info, save_path, clusters[0], clusters[1])
    #compute_human_agent_js_divergence(vae, dataset, save_path)
    num_clusters = len(gaussians)
    analyze_role_representations(info,save_path,num_clusters)
    analyze_action_patterns(info,save_path)
    #eval_vae_accuracy(vae, dataset, save_path, num_clusters, batch_size)
    # close figures
    plt.close('all')
    return accuracy, cluster_labels, gaussians

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE model")
    parser.add_argument("--model-path", type=str, default="hallway_training_ll/20250508_231458_ep_100", help="Path to the saved model file")
    parser.add_argument("--dataset-path", type=str, default="../data/burrito_hallway_bp_11pol.pkl", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--window-length", type=int, default=50, help="Length of each encoder traj input")
    parser.add_argument("--horizon", type=int, default=50, help="How many actions in the future to predict")
    parser.add_argument("--latent-dim", type=int, default=8, help="Dimension of the latent space")
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=20)

    args = parser.parse_args()
 
    dataset = load_dataset(args.dataset_path)
    # Get observation dimension and action dimension
    obs_shape = dataset.obs_shape
    action_dim = dataset.action_dim
    
    print(f"obs_shape: {obs_shape}, Action dim: {action_dim}")

    dataset = ChunkedTrajectoryDataset(
        dataset=dataset,
        window_size=args.window_length,
        horizon=args.horizon,
    )
    #dataset1 = load_dataset("./../data/human_open.pkl")

    #dataset.append_samples(dataset1)

    model = os.path.join(args.model_path, "encoder.pt")
    vae = load_vae(
        model_path=model,
        obs_shape=obs_shape,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        window_length=args.window_length,
        horizon=args.horizon
    )
    vae.eval()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(args.model_path, f"{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
 
    eval_and_cluster(vae, dataset, analysis_dir, args.batch_size, [args.min_clusters, args.max_clusters])
