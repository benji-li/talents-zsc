import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

from agent_characterization.role_encoder import ImprovedRoleVAE
from agent_characterization.gen_data import load_dataset, ChunkedTrajectoryDataset
from agent_characterization.clustering.kde import kde_clustering, analyze_kde_clusters, evaluate_kde_clusters, visualize_kde_clusters
from agent_characterization.clustering.kmeans import kmeans_clustering, analyze_kmeans_clusters, evaluate_kmeans_clusters, visualize_kmeans_clusters

def preprocess_data(vae, dataset, output_dir, batch_size=32, trim=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Collect data
    latent_vectors = []
    agent_labels = []
    action_sequences = []
    future_action_sequences = []
    sample_indices = []  # Track indices for evaluation
    
    print("Extracting latent representations...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get current batch data
            obs_sequence = batch['obs_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            future_actions = batch['future_actions'].to(device)
            future_obs = batch['future_obs'].to(device)
            
            # Extract agent labels
            batch_agents = batch['agent']
            
            # Encode observations to get latent vectors
            #z, mixture_weights, mu, log_var, comp_indx = vae.encode(obs_sequence, action_sequence)

            z, mu, log_var = vae.encode(obs_sequence, action_sequence)
            
            # Run forward pass to get predictions
            #pred_actions, _,_,_, _ = vae(obs_sequence, action_sequence, future_obs)

            pred_actions, _, _ = vae(obs_sequence, action_sequence, future_obs)
            
            # Store data
            latent_vectors.append(z.cpu().numpy())
            agent_labels.extend(batch_agents)
            action_sequences.append(action_sequence.cpu().numpy())
            future_action_sequences.append(future_actions.cpu().numpy())
            
            # Store indices
            batch_indices = list(range(
                batch_idx * batch_size, 
                min((batch_idx + 1) * batch_size, len(dataset))
            ))
            sample_indices.extend(batch_indices)
            
            # Process only a subset for prediction accuracy evaluation
            if trim and len(latent_vectors) * batch_size >= 2000:
                break
    
    # Combine all batches
    latent_vectors = np.vstack(latent_vectors)
    action_sequences = np.vstack(action_sequences)
    future_action_sequences = np.vstack(future_action_sequences)

    # for comparing with humans
    # for i in range(len(agent_labels)):
    #     if "agent" in agent_labels[i]:
    #         agent_labels[i] = "agent"
    #     #else:
    #     #    agent_labels[i] = "human"
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame({
        'agent': agent_labels,
        'sample_idx': sample_indices
    })
    
    # Add the latent vectors as columns
    for i in range(latent_vectors.shape[1]):
        df[f'latent_{i}'] = latent_vectors[:, i]
    
    # Save the raw data for further analysis
    print(f"Saving raw data to {output_dir}...")
    np.save(os.path.join(output_dir, "latent_vectors.npy"), latent_vectors)

    df.to_csv(os.path.join(output_dir, "latent_data.csv"), index=False)
    
    # Get unique agent types
    unique_agents = sorted(set(agent_labels))
    print(f"Found {len(unique_agents)} unique agent types: {unique_agents}")

    return (df, latent_vectors, unique_agents, agent_labels, future_action_sequences)

def analyze_role_representations(info, output_dir, n_clusters=4):
    """
    Analyze the learned role representations with enhanced visualizations and evaluation
    
    Args:
        vae: Trained VAE model
        dataset: ChunkedTrajectoryDataset containing the data
        output_dir: Directory to save analysis results
        n_clusters: Number of clusters for K-means clustering
        batch_size: Batch size for processing
    """
   
    df, latent_vectors, unique_agents, agent_labels, _ = info

    # Analyze latent space
    print("Analyzing latent space...")
    
    # 1. PCA Visualization by Agent Type
    print("Generating PCA visualization...")
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)
    
    plt.figure(figsize=(12, 10))
    for agent_type in unique_agents:
        mask = [a == agent_type for a in agent_labels]
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            label=agent_type,
            alpha=0.7
        )
    
    plt.title('Latent Space PCA by Agent Type')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_pca_by_agent.png'))


    # 2. t-SNE Visualization (better for capturing clusters)
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d_tsne = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(12, 10))
    for agent_type in unique_agents:
        mask = [a == agent_type for a in agent_labels]
        plt.scatter(
            latent_2d_tsne[mask, 0], 
            latent_2d_tsne[mask, 1], 
            label=agent_type,
            alpha=0.7
        )
    
    plt.title('Latent Space t-SNE by Agent Type')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_tsne_by_agent.png'))


    # 3. K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(latent_vectors)
    clusters = kmeans.labels_
    
    # Add cluster labels to dataframe
    df['cluster'] = clusters
    
    # Visualize clusters
    plt.figure(figsize=(12, 10))

    from matplotlib import cm
    colormap = cm.cividis # Try: 'plasma', 'inferno', 'cividis', 'tab10', 'Set1', etc.
    colors = colormap(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        mask = clusters == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            label=f'Cluster {cluster_id}',
            color=colors[cluster_id],
            alpha=0.7
        )
    
    plt.title('K-means Clusters in Latent Space (PCA)')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kmeans_clusters_pca_perm.png'))
    
    # 4. Agent distribution across clusters
    print("Analyzing agent distribution across clusters...")
    agent_cluster_counts = pd.crosstab(
        pd.Series([agent_labels[i] for i in range(len(clusters))], name='Agent'),
        pd.Series(clusters, name='Cluster')
    )
    
    # Plot the distribution
    plt.figure(figsize=(12, 8))
    sns.heatmap(agent_cluster_counts, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Agent Distribution Across Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_cluster_distribution.png'))
    
    # Save the distribution to CSV
    agent_cluster_counts.to_csv(os.path.join(output_dir, 'agent_cluster_counts.csv'))
    
    return df, kmeans.cluster_centers_


def analyze_action_patterns(info, output_dir):
    _, _, unique_agents, agent_labels, future_action_sequences = info

    print("Analyzing action patterns by agent type...")
    action_patterns_by_agent = {}
    
    for agent_type in unique_agents:
        agent_mask = [a == agent_type for a in agent_labels]
        agent_actions = future_action_sequences[agent_mask]
        
        # Count frequency of each action
        action_counts = defaultdict(int)
        for seq in agent_actions:
            for action in seq:
                action_counts[int(action)] += 1
        
        action_patterns_by_agent[agent_type] = action_counts
    
    # Create a bar chart of action distributions
    action_names = ["North", "South", "East", "West", "Stay", "Interact"]
    
    # Prepare data for plotting
    action_data = {agent: [0] * len(action_names) for agent in unique_agents}
    for agent, counts in action_patterns_by_agent.items():
        for action, count in counts.items():
            if action < len(action_names):
                action_data[agent][action] = count
    
    # Normalize counts
    for agent in action_data:
        total = sum(action_data[agent])
        if total > 0:
            action_data[agent] = [count / total for count in action_data[agent]]
    
    # Plot action distribution
    plt.figure(figsize=(14, 8))
    x = np.arange(len(action_names))
    width = 0.8 / len(unique_agents)
    
    for i, agent in enumerate(unique_agents):
        plt.bar(
            x + i * width - 0.4 + width/2, 
            action_data[agent], 
            width, 
            label=agent
        )
    
    plt.xlabel('Action')
    plt.ylabel('Normalized Frequency')
    plt.title('Action Distribution by Agent Type')
    plt.xticks(x, action_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution_by_agent.png'))

    return

def eval_vae_accuracy(vae, dataset, output_dir, n_clusters=4, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.eval()
    
    print("Evaluating agent type classification based on latent space proximity...")
    
    # First pass: collect all latent vectors and agent types
    all_latent_vectors = []

    all_agent_labels = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            obs_sequence = batch['obs_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            batch_agents = batch['agent']
            
            # Encode observations to get latent vectors
            #_, _, mu, _, _ = vae.encode(obs_sequence, action_sequence)

            z,mu, logvar = vae.encode(obs_sequence, action_sequence)

            latent_vectors = mu.cpu().numpy()

            # Store latent vectors and agent labels
            all_latent_vectors.append(latent_vectors)

            all_agent_labels.extend(batch_agents)
    
    # Combine all batches
    all_latent_vectors = np.vstack(all_latent_vectors)
    
    # Get unique agent types
    unique_agents = sorted(set(all_agent_labels))
    
    # Calculate mean latent vector for each agent type
    agent_means = {}
    for agent_type in unique_agents:
        agent_mask = [a == agent_type for a in all_agent_labels]
        agent_latents = all_latent_vectors[agent_mask]
        agent_means[agent_type] = np.mean(agent_latents, axis=0)

    
    print(f"Computed mean latent vectors for {len(unique_agents)} agent types")
    
    # Classify each sample based on nearest centroid
    y_true = all_agent_labels
    y_pred = []
    
    for i, latent_vector in enumerate(all_latent_vectors):
        # Calculate distance to each agent mean
        distances = {}
        for agent_type, mean_vector in agent_means.items():
            dist = np.linalg.norm(latent_vector - mean_vector)
            distances[agent_type] = dist
        
        # Find the agent type with minimum distance
        predicted_agent = min(distances, key=distances.get)
        #y_pred.append(predicted_agent)
        y_pred.append(predicted_agent)
    

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_agents)
    
    # Calculate classification accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall classification accuracy: {accuracy:.2%}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=unique_agents,
        yticklabels=unique_agents
    )
    plt.xlabel('Predicted Agent Type')
    plt.ylabel('True Agent Type')
    plt.title('Agent Type Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_type_confusion_matrix.png'))
    
    # Calculate per-class metrics
    precision_by_agent = {}
    recall_by_agent = {}
    
    for i, agent in enumerate(unique_agents):
        # Precision = TP / (TP + FP)
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        precision_by_agent[agent] = precision
        
        # Recall = TP / (TP + FN)
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        recall_by_agent[agent] = recall
    
    # Plot precision and recall by agent type
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(unique_agents))
    width = 0.35
    
    plt.bar(x - width/2, [precision_by_agent[a] for a in unique_agents], width, label='Precision')
    plt.bar(x + width/2, [recall_by_agent[a] for a in unique_agents], width, label='Recall')
    
    plt.xlabel('Agent Type')
    plt.ylabel('Score')
    plt.title('Precision and Recall by Agent Type')
    plt.xticks(x, unique_agents)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_type_precision_recall.png'))
    
    # Save results to summary file
    with open(os.path.join(output_dir, 'agent_classification_summary.txt'), 'w') as f:
        f.write("AGENT TYPE CLASSIFICATION SUMMARY\n")
        f.write("================================\n\n")
        f.write(f"Overall accuracy: {accuracy:.2%}\n\n")
        
        f.write("Per-agent type metrics:\n")
        for agent in unique_agents:
            f.write(f"  {agent}:\n")
            f.write(f"    Precision: {precision_by_agent[agent]:.2%}\n")
            f.write(f"    Recall: {recall_by_agent[agent]:.2%}\n")
    
    print(f"Evaluation complete! Results saved to {output_dir}")
    
    # Return the DataFrame with classification results
    df = pd.DataFrame({
        'true_agent': y_true,
        'predicted_agent': y_pred,
    })
    
    return df

def load_and_evaluate_vae(model_path,
                          dataset,
                          analysis_dir,
                          batch_size,
                          obs_shape,
                          action_dim,
                          latent_dim,
                          window_length,
                          horizon,
                          n_clusters
                          ):
    """
    Load a trained VAE model and evaluate it
    
    Args:
        model_path: Path to the saved model state dict
        dataset: Dataset to use for evaluation
        analysis_dir: Directory to save analysis results
        batch_size: Batch size for processing
    """
    vae = load_vae(model_path=model_path,
                   obs_shape=obs_shape,
                   action_dim=action_dim,
                   latent_dim=latent_dim,
                   window_length=window_length,
                   horizon=horizon)

    eval_results, kmean_centers, xmean_centers = evaluate_vae(vae, dataset,analysis_dir, batch_size, n_clusters)
   
    print("Analysis complete!")
    return eval_results, kmean_centers, xmean_centers

def load_vae(model_path, obs_shape, action_dim, latent_dim, window_length, horizon):
    vae = ImprovedRoleVAE(
        state_shape=obs_shape,
        action_dim=action_dim+1,
        latent_dim=latent_dim,
        traj_length=window_length,
        horizon=horizon
    )
    
    # Load the saved state dictionary
    print(f"Loading VAE from {model_path}...")
    vae.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    return vae


def evaluate_vae(vae, dataset, analysis_dir, batch_size, n_clusters):
    """
    Load a trained VAE model and evaluate it
    """
    vae.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    
    # Preprocess data using the loaded model
    print(f"Preprocessing data and saving to {analysis_dir}...")
    info = preprocess_data(vae, dataset, analysis_dir, batch_size=batch_size)
    
    # Run other analysis functions as needed
    _, kmean_centers = analyze_role_representations(info, analysis_dir, n_clusters)
    analyze_action_patterns(info, analysis_dir)
    
    # Add KDE clustering analysis
    kde_accuracy, kde_labels = analyze_kde_clusters(info, analysis_dir)
    
    # Run analyze_kde_clusters again with explicit return of cluster centers
    # This is needed since the original function might not return centers
    _, latent_vectors, _, _, _ = info
    kde_cluster_labels, kde_cluster_centers = kde_clustering(info, analysis_dir)
    
    # Add X-means clustering analysis
    xmeans_accuracy, xmeans_labels, xmeans_centers = analyze_xmeans_clusters(info, analysis_dir)


    kmac, kmcl, gaussians = analyze_kmeans_clusters(info, analysis_dir)
    
    # Add HDBSCAN clustering analysis
    from hdbscan_clustering import analyze_hdbscan_clusters

    # With this:
    hdbscan_accuracy, hdbscan_labels, hdbscan_exemplars = analyze_hdbscan_clusters(
    info, 
    analysis_dir, 
    min_cluster_size=10,         # Smaller clusters for more granularity
    min_samples=1,              # More sensitive to detect small clusters
    cluster_selection_epsilon=0.1  # Allow some cluster expansion
    )
    
    # Compare all clustering methods
    df, latent_vectors, unique_agents, agent_labels, _ = info
    df['kde_cluster'] = kde_labels
    df['xmeans_cluster'] = xmeans_labels
    df['hdbscan_cluster'] = hdbscan_labels
 
    # Create improved comparison visualization with ground truth and all clustering methods
    plt.figure(figsize=(20, 10))
    
    # Get PCA transformed data once
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)
    
    # Use matplotlib's categorical colormaps that are more visually distinct
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    
    # Use Tab10 colormap for ground truth (distinct colors, better visibility)
    tab10 = cm.get_cmap('tab10')
    # Use Dark2 colormap for KDE (distinct darker colors, good contrast)
    dark2 = cm.get_cmap('Dark2')
    # Use Set1 colormap for X-means (distinct colors, good visibility)
    set1 = cm.get_cmap('Set1')
    # Use Paired colormap for HDBSCAN (works well for noise points)
    paired = cm.get_cmap('Paired')
    
    # Ground Truth subplot
    plt.subplot(2, 2, 1)
    
    # Calculate centroid for each agent type
    agent_centroids = {}
    for agent_type in unique_agents:
        mask = [a == agent_type for a in agent_labels]
        agent_points = latent_2d_pca[mask]
        agent_centroids[agent_type] = np.mean(agent_points, axis=0)
    
    # Plot each agent type separately to create proper legend entries
    for i, agent_type in enumerate(unique_agents):
        mask = [a == agent_type for a in agent_labels]
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            color=tab10(i % 10),
            alpha=0.7,
            label=agent_type
        )
        
    # Plot centroids for each agent type as red X
    for i, agent_type in enumerate(unique_agents):
        centroid = agent_centroids[agent_type]
        plt.scatter(
            centroid[0], 
            centroid[1], 
            color='red', 
            marker='X', 
            s=150, 
            edgecolor='black',
            zorder=5  # ensure centroids are drawn on top
        )
    
    # Add a single red X to the legend
    plt.scatter([], [], color='red', marker='X', s=100, edgecolor='black', label='Centroid')
    
    plt.title('Ground Truth Agent Types')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(title="Agent Type", loc='best', fontsize='small')
    
    # KDE subplot
    plt.subplot(2, 2, 2)
    kde_unique_labels = sorted(set(kde_labels))
    
    for i, cluster_id in enumerate(kde_unique_labels):
        mask = kde_labels == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            color=dark2(i % 8),  # Dark2 has 8 colors
            alpha=0.7,
            label=f'Cluster {cluster_id}'
        )
    
    # Transform KDE cluster centers to PCA space and plot as red X
    if kde_cluster_centers is not None:
        kde_centers_pca = pca.transform(kde_cluster_centers)
        plt.scatter(
            kde_centers_pca[:, 0], 
            kde_centers_pca[:, 1], 
            color='red', 
            marker='X', 
            s=150, 
            edgecolor='black',
            zorder=5
        )
        # Add a single red X to the legend
        plt.scatter([], [], color='red', marker='X', s=100, edgecolor='black', label='Cluster Center')
    
    plt.title(f'KDE Clustering (Accuracy: {kde_accuracy:.1%})')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(title="Cluster ID", loc='best', fontsize='small')
    
    # X-Means subplot
    plt.subplot(2, 2, 3)
    xmeans_unique_labels = sorted(set(xmeans_labels))
    
    for i, cluster_id in enumerate(xmeans_unique_labels):
        mask = xmeans_labels == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            color=set1(i % 9),  # Set1 has 9 colors
            alpha=0.7,
            label=f'Cluster {cluster_id}'
        )
    
    # Transform X-means cluster centers to PCA space and plot as red X
    if xmeans_centers is not None:
        xmeans_centers_pca = pca.transform(xmeans_centers)
        plt.scatter(
            xmeans_centers_pca[:, 0], 
            xmeans_centers_pca[:, 1], 
            color='red', 
            marker='X', 
            s=150, 
            edgecolor='black',
            zorder=5
        )
        # Add a single red X to the legend
        plt.scatter([], [], color='red', marker='X', s=100, edgecolor='black', label='Cluster Center')
    
    plt.title(f'X-Means Clustering (Accuracy: {xmeans_accuracy:.1%})')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(title="Cluster ID", loc='best', fontsize='small')
    
    # HDBSCAN subplot
    plt.subplot(2, 2, 4)
    hdbscan_unique_labels = sorted(set(hdbscan_labels))
    
    # Plot noise points first if they exist
    if -1 in hdbscan_unique_labels:
        hdbscan_unique_labels.remove(-1)
        noise_mask = hdbscan_labels == -1
        plt.scatter(
            latent_2d_pca[noise_mask, 0], 
            latent_2d_pca[noise_mask, 1], 
            color='lightgray',
            marker='.',
            alpha=0.5,
            label='Noise'
        )
    
    # Plot regular clusters
    for i, cluster_id in enumerate(hdbscan_unique_labels):
        mask = hdbscan_labels == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            color=paired(i % 12),  # Paired has 12 colors
            alpha=0.7,
            label=f'Cluster {cluster_id}'
        )
    
    # Transform HDBSCAN exemplars to PCA space and plot as red X
    if hdbscan_exemplars:
        exemplar_points = np.vstack([hdbscan_exemplars[cluster_id] for cluster_id in hdbscan_unique_labels 
                                    if cluster_id in hdbscan_exemplars])
        if len(exemplar_points) > 0:
            exemplar_pca = pca.transform(exemplar_points)
            plt.scatter(
                exemplar_pca[:, 0], 
                exemplar_pca[:, 1], 
                color='red', 
                marker='X', 
                s=150, 
                edgecolor='black',
                zorder=5
            )
            # Add a single red X to the legend
            plt.scatter([], [], color='red', marker='X', s=100, edgecolor='black', label='Cluster Center')
    
    plt.title(f'HDBSCAN Clustering (Accuracy: {hdbscan_accuracy:.1%})')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(title="Cluster ID", loc='best', fontsize='small')
 
    plt.suptitle('Clustering Method Comparison with Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'clustering_method_comparison.png'))
    
    # Add this comparative analysis to summary file
    with open(os.path.join(analysis_dir, 'clustering_comparison.txt'), 'w') as f:
        f.write("CLUSTERING METHOD COMPARISON\n")
        f.write("===========================\n\n")
        
        # Get K-means accuracy (this requires calculating it)
        kmeans_labels = df['cluster'].values
        kmeans_accuracy = calculate_clustering_accuracy(kmeans_labels, agent_labels, unique_agents)
        
        f.write(f"K-means clustering accuracy: {kmeans_accuracy:.2%}\n")
        f.write(f"KDE/MeanShift clustering accuracy: {kde_accuracy:.2%}\n")
        f.write(f"X-means clustering accuracy: {xmeans_accuracy:.2%}\n")
        f.write(f"HDBSCAN clustering accuracy: {hdbscan_accuracy:.2%}\n\n")
        
        f.write(f"Number of agent types (ground truth): {len(unique_agents)}\n")
        f.write(f"Number of K-means clusters: {df['cluster'].nunique()}\n")
        f.write(f"Number of KDE/MeanShift clusters: {df['kde_cluster'].nunique()}\n")
        f.write(f"Number of X-means clusters: {df['xmeans_cluster'].nunique()}\n")
        
        # For HDBSCAN, count clusters excluding noise
        n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        n_noise = list(hdbscan_labels).count(-1)
        f.write(f"Number of HDBSCAN clusters: {n_hdbscan_clusters}")
        if -1 in hdbscan_labels:
            f.write(f" (plus {n_noise} noise points, {n_noise/len(hdbscan_labels):.1%} of data)")
        f.write("\n")
    
    # Run the standard evaluation function
    eval_results = eval_vae_accuracy(vae, dataset, analysis_dir, batch_size=batch_size)
    
    print("Analysis complete!")
    return eval_results, kmean_centers, xmeans_centers

# Note: You might need to modify analyze_xmeans_clusters to return cluster centers
# Ensure it returns (accuracy, labels, centers) instead of just (accuracy, labels)
def analyze_xmeans_clusters(info, output_dir, min_clusters=2, max_clusters=15):
    """
    Perform X-means clustering analysis and evaluate against ground truth
    
    Args:
        info: Tuple containing (df, latent_vectors, unique_agents, agent_labels, future_action_sequences)
        output_dir: Directory to save analysis results
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
    
    Returns:
        accuracy: Accuracy of X-means clustering compared to ground truth agent types
        cluster_labels: The cluster assignments
        cluster_centers: The centers of each cluster
    """
    df, latent_vectors, unique_agents, agent_labels, _ = info
    
    # Create output directory for X-means
    xmeans_dir = os.path.join(output_dir, 'xmeans')
    os.makedirs(xmeans_dir, exist_ok=True)
    
    # Run X-means clustering
    print("Performing X-means clustering analysis...")
    cluster_labels, cluster_centers, optimal_k = kmeans_clustering(
        info, xmeans_dir, min_clusters, max_clusters
    )
    
    # Visualize X-means clusters
    print("Visualizing X-means clusters...")
    visualize_kmeans_clusters(latent_vectors, cluster_labels, cluster_centers, optimal_k, xmeans_dir)
    
    # Evaluate X-means clusters against ground truth
    print("Evaluating X-means clusters against ground truth agent types...")
    accuracy = evaluate_kmeans_clusters(cluster_labels, agent_labels, unique_agents, xmeans_dir)
    
    print(f"X-means clustering analysis complete. Found {optimal_k} clusters with accuracy {accuracy:.2%}")
    
    # Add xmeans cluster labels to the dataframe
    df['xmeans_cluster'] = cluster_labels
    df.to_csv(os.path.join(xmeans_dir, 'latent_data_with_xmeans.csv'), index=False)
    
    return accuracy, cluster_labels, cluster_centers

def calculate_clustering_accuracy(cluster_labels, agent_labels, unique_agents):
    """Helper function to calculate accuracy for any clustering method"""
    # Map clusters to agent types based on majority vote
    cluster_to_agent = {}
    unique_clusters = sorted(set(cluster_labels))
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_agents = [agent_labels[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
        
        # Find the most common agent in this cluster
        agent_counts = {}
        for agent in cluster_agents:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        majority_agent = max(agent_counts, key=agent_counts.get)
        cluster_to_agent[cluster_id] = majority_agent
    
    # Create predicted labels based on cluster mapping
    predicted_agents = [cluster_to_agent[cluster] for cluster in cluster_labels]
    
    # Calculate accuracy
    correct_predictions = sum(1 for true, pred in zip(agent_labels, predicted_agents) if true == pred)
    accuracy = correct_predictions / len(agent_labels)
    
    return accuracy

    return js_divergence
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE model")
    parser.add_argument("--model", type=str, default="encoder.pt", help="Path to the saved model file")
    parser.add_argument("--dataset-path", type=str, default="./data/4role_trajectories.pkl", help="Path to the dataset")
    parser.add_argument("--analysis_dir", type=str, default="analysis_results", help="Directory to save analysis results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--window-length", type=int, default=100, help="Length of each encoder traj input")
    parser.add_argument("--horizon", type=int, default=10, help="How many actions in the future to predict")
    parser.add_argument("--latent-dim", type=int, default=8, help="Dimension of the latent space")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    # Get observation dimension and action dimension
    obs_shape = dataset.obs_shape
    action_dim = dataset.action_dim # From ACTION_MAP in overcooked_rllib.py
    
    print(f"obs_shape: {obs_shape}, Action dim: {action_dim}")

   
    # Load your dataset
    # This will depend on how your dataset class is defined
    dataset = ChunkedTrajectoryDataset(
        dataset=dataset,
        window_size=args.window_length,
        horizon=args.horizon,
    )
    
    # Load the model and run evaluation
    eval_results, kmeans_centers, xmeans_centers = load_and_evaluate_vae(
        args.model,
        dataset,
        args.analysis_dir,
        args.batch_size,
        obs_shape,
        action_dim,
        args.latent_dim,
        args.window_length,
        args.horizon,
        n_clusters=4
    )
