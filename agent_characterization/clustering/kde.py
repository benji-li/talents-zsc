import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd

def kde_clustering(info, output_dir, bandwidth='auto'):
    from sklearn.cluster import MeanShift, estimate_bandwidth
    df, latent_vectors, _, _, _ = info

    if bandwidth == 'auto':
        bandwidth = estimate_bandwidth(
            latent_vectors,
            quantile=0.1,
            n_samples=int(len(latent_vectors)/5)
        )
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    cluster_labels = meanshift.fit_predict(latent_vectors)
    cluster_centers = meanshift.cluster_centers_
    n_clusters = len(cluster_centers)
    print(f"Found {n_clusters} clusters using KDE MeanShift")
    return cluster_labels, cluster_centers

def visualize_kde_clusters(latent_2d_pca, cluster_labels, cluster_centers, n_clusters, output_dir):
    """
    Visualize KDE/MeanShift clusters in PCA space
    
    Args:
        latent_2d_pca: PCA-reduced latent vectors (2D)
        cluster_labels: Cluster labels from KDE/MeanShift
        cluster_centers: Cluster centers from KDE/MeanShift
        n_clusters: Number of unique clusters
        output_dir: Directory to save the visualization
    """
    plt.figure(figsize=(12, 10))
    
    from matplotlib import cm
    colormap = cm.viridis  # Different colormap from K-means for easy distinction
    colors = colormap(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            label=f'Cluster {cluster_id}',
            color=colors[cluster_id],
            alpha=0.7
        )
    
    # Plot cluster centers
    centers_2d_pca = pca.transform(cluster_centers)
    plt.scatter(
        centers_2d_pca[:, 0],
        centers_2d_pca[:, 1],
        marker='X',
        color='red',
        s=200,
        label='Cluster Centers'
    )
    
    plt.title('KDE/MeanShift Clusters in Latent Space (PCA)')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kde_clusters_pca.png'))

def evaluate_kde_clusters(cluster_labels, agent_labels, unique_agents, output_dir):
    """
    Evaluate how well KDE clusters align with ground truth agent types
    
    Args:
        cluster_labels: Cluster labels from KDE/MeanShift
        agent_labels: True agent labels
        unique_agents: List of unique agent types
        output_dir: Directory to save evaluation results
    
    Returns:
        accuracy: Overall accuracy of clustering compared to agent types
    """
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
    
    # Create confusion matrix
    cm = confusion_matrix(agent_labels, predicted_agents, labels=unique_agents)
    
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
    plt.xlabel('Predicted Agent Type (via KDE Clusters)')
    plt.ylabel('True Agent Type')
    plt.title(f'KDE Clustering Accuracy: {accuracy:.2%}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kde_clusters_confusion_matrix.png'))
    
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
    plt.title('KDE Clustering: Precision and Recall by Agent Type')
    plt.xticks(x, unique_agents)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kde_clusters_precision_recall.png'))
    
    # Create a cluster distribution analysis
    agent_cluster_counts = pd.crosstab(
        pd.Series(agent_labels, name='Agent'),
        pd.Series(cluster_labels, name='Cluster')
    )
    
    # Plot the distribution
    plt.figure(figsize=(12, 8))
    sns.heatmap(agent_cluster_counts, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Agent Distribution Across KDE Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_kde_cluster_distribution.png'))
    
    # Save the mapping and metrics to a text file
    with open(os.path.join(output_dir, 'kde_cluster_mapping.txt'), 'w') as f:
        f.write("KDE CLUSTER TO AGENT MAPPING\n")
        f.write("===========================\n\n")
        for cluster_id, agent in cluster_to_agent.items():
            f.write(f"Cluster {cluster_id} -> {agent}\n")
        
        f.write(f"\nOverall accuracy: {accuracy:.2%}\n\n")
        
        f.write("Per-agent type metrics:\n")
        for agent in unique_agents:
            f.write(f"  {agent}:\n")
            f.write(f"    Precision: {precision_by_agent[agent]:.2%}\n")
            f.write(f"    Recall: {recall_by_agent[agent]:.2%}\n")
    
    return accuracy

def analyze_kde_clusters(info, output_dir, bandwidth='auto'):
    """
    Perform KDE clustering analysis and evaluate against ground truth
    
    Args:
        info: Tuple containing (df, latent_vectors, unique_agents, agent_labels, future_action_sequences)
        output_dir: Directory to save analysis results
        bandwidth: Bandwidth for MeanShift algorithm, 'auto' for automatic estimation
    
    Returns:
        accuracy: Accuracy of KDE clustering compared to ground truth agent types
    """
    df, latent_vectors, unique_agents, agent_labels, _ = info
    
    # Run KDE clustering
    print("Performing KDE clustering with MeanShift...")
    cluster_labels, cluster_centers = kde_clustering(info, output_dir, bandwidth)
    n_clusters = len(cluster_centers)
    
    # Compute PCA for visualization
    print("Computing PCA for KDE cluster visualization...")
    global pca
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)
    
    # Visualize KDE clusters
    print("Visualizing KDE clusters...")
    visualize_kde_clusters(latent_2d_pca, cluster_labels, cluster_centers, n_clusters, output_dir)
    
    # Evaluate KDE clusters against ground truth
    print("Evaluating KDE clusters against ground truth agent types...")
    accuracy = evaluate_kde_clusters(cluster_labels, agent_labels, unique_agents, output_dir)
    
    print(f"KDE clustering analysis complete. Found {n_clusters} clusters with accuracy {accuracy:.2%}")
    
    # Compare KDE vs K-means clustering
    plt.figure(figsize=(10, 6))
    plt.title("KDE Clustering vs Ground Truth Agent Types")
    plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(label='KDE Cluster ID')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.savefig(os.path.join(output_dir, 'kde_clustering_scatter.png'))
    
    return accuracy, cluster_labels


