import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def bic_score(data, labels, centers):
    """
    Calculate the Bayesian Information Criterion (BIC) score for clustering.
    
    Args:
        data: The dataset used for clustering
        labels: Cluster labels for each data point
        centers: Cluster centers
        
    Returns:
        bic: The BIC score (lower is better)
    """
    n_samples, n_features = data.shape
    n_clusters = len(np.unique(labels))
    
    # Calculate log-likelihood
    log_likelihood = 0
    for i in range(n_clusters):
        cluster_data = data[labels == i]
        n_cluster_samples = len(cluster_data)
        
        if n_cluster_samples > 0:
            # Calculate within-cluster variance (use mean squared distance to centroid)
            cluster_variance = np.sum((cluster_data - centers[i])**2) / n_cluster_samples
            if cluster_variance <= 0:
                cluster_variance = 1e-10  # Avoid division by zero or log(0)
            
            # Add contribution to log-likelihood
            log_likelihood -= n_cluster_samples * (
                np.log(n_cluster_samples) - 
                n_cluster_samples * np.log(n_samples) -
                0.5 * n_features * np.log(2 * np.pi * cluster_variance) -
                0.5 * n_cluster_samples
            )
    
    # Calculate BIC: -2 * log-likelihood + k * log(n)
    # k = number of parameters = n_clusters * (n_features + 1)
    k = n_clusters * (n_features + 1)
    bic = -2 * log_likelihood + k * np.log(n_samples)
    
    return bic

def kmeans_clustering(info, output_dir, min_clusters=4, max_clusters=10, random_state=42, verbose=True):
    """
    Perform X-means clustering that automatically determines the optimal number of clusters.
    
    Args:
        info: Tuple containing (df, latent_vectors, unique_agents, agent_labels, future_action_sequences)
        output_dir: Directory to save analysis results
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        
    Returns:
        cluster_labels: Array of cluster assignments for each sample
        cluster_centers: Array of cluster center coordinates
        optimal_k: The optimal number of clusters determined by BIC
    """
    df, latent_vectors, _, _, _ = info
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare to store BIC scores
    bic_scores = []
    silhouette_scores = []
    kmeans_models = []
    
    print("Running X-means clustering (K-means with BIC optimization)...")
    
    # Try different numbers of clusters
    for k in range(min_clusters, max_clusters + 1):
        print(f"  Testing k={k}...")
        
        # Run K-means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(latent_vectors)
        centers = kmeans.cluster_centers_
        
        # Calculate BIC score
        bic = bic_score(latent_vectors, labels, centers)
        bic_scores.append(bic)
        
        # Calculate silhouette score (optional, for additional validation)
        if k > 1:  # Silhouette requires at least 2 clusters
            try:
                sil_score = silhouette_score(latent_vectors, labels)
                silhouette_scores.append(sil_score)
            except:
                silhouette_scores.append(0)
        else:
            silhouette_scores.append(0)
        
        # Store the model
        kmeans_models.append(kmeans)
        
        print(f"    BIC score: {bic:.2f}, Silhouette score: {silhouette_scores[-1]:.4f}")
    
    # Find optimal number of clusters (minimum BIC)
    optimal_idx_bic = np.argmin(bic_scores)
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = min_clusters + optimal_idx
    optimal_k_bic = min_clusters + optimal_idx_bic
    
    print(f"Optimal number of clusters by Silhouette: {optimal_k}")
    print(f"Optimal number of clusters by BIC: {optimal_k_bic}")
    
    # Get the best model
    best_kmeans = kmeans_models[optimal_idx]
    best_labels = best_kmeans.labels_
    best_centers = best_kmeans.cluster_centers_
    
    if verbose:
        # Plot BIC scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_clusters, max_clusters + 1), bic_scores, 'o-', color='blue')
        plt.axvline(x=optimal_k, color='red', linestyle='--')
        plt.title('BIC Scores by Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('BIC Score (lower is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'xmeans_bic_scores.png'))
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-', color='green')
        plt.title('Silhouette Scores by Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score (higher is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'xmeans_silhouette_scores.png'))
        
        # Save scores to CSV
        scores_df = pd.DataFrame({
            'num_clusters': range(min_clusters, max_clusters + 1),
            'bic_score': bic_scores,
            'silhouette_score': silhouette_scores
        })
        scores_df.to_csv(os.path.join(output_dir, 'xmeans_scores.csv'), index=False)
    
    return best_labels, best_centers, optimal_k

def fit_gaussians_to_clusters(latent_vectors, cluster_labels):
    """
    Fit a Gaussian distribution to each cluster and return the means and covariances.
    
    Args:
        latent_vectors: The latent space vectors
        cluster_labels: Cluster labels from clustering algorithm
    
    Returns:
        dict: Dictionary of form {cluster_id: {'mean': mean_vector, 'cov': covariance_matrix}}
    """
    unique_clusters = sorted(set(cluster_labels))
    gaussians = {}
    
    for cluster_id in unique_clusters:
        # Extract points belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_points = latent_vectors[cluster_mask]
        
        # Calculate mean vector
        mean_vector = np.mean(cluster_points, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(cluster_points, rowvar=False)
        
        # Store in dictionary
        gaussians[cluster_id] = {
            'mean': mean_vector,
            'cov': cov_matrix
        }
    
    return gaussians

def visualize_cluster_gaussians(latent_vectors, cluster_labels, gaussians, output_dir):
    """
    Visualize the Gaussian distributions fitted to each cluster in PCA space.
    
    Args:
        latent_vectors: The latent space vectors
        cluster_labels: Cluster labels from clustering algorithm
        gaussians: Dictionary of Gaussian parameters {cluster_id: {'mean': mean_vector, 'cov': covariance_matrix}}
        output_dir: Directory to save visualization
    """
    # Compute PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    
    from matplotlib import cm
    from matplotlib.patches import Ellipse
    
    n_clusters = len(gaussians)
    colormap = cm.tab10
    colors = colormap(np.linspace(0, 1, n_clusters))
    
    # Plot the data points
    for cluster_id in sorted(gaussians.keys()):
        mask = cluster_labels == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            label=f'Cluster {cluster_id}',
            color=colors[cluster_id % 10],
            alpha=0.5
        )
    
    # Project Gaussian parameters to PCA space and plot ellipses
    for cluster_id, params in gaussians.items():
        # Project mean to PCA space
        mean_pca = pca.transform([params['mean']])[0]
        
        # Project covariance to PCA space
        # For PCA, we need to transform: cov_pca = components.T @ cov @ components
        cov_pca = pca.components_.dot(params['cov']).dot(pca.components_.T)
        
        # Calculate eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov_pca)
        
        # Sort eigenvalues in decreasing order
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # The angle in degrees from the x-axis to the first eigenvector
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # Width and height are "full width, full height" of ellipse
        # Plot ellipses at 1, 2, and 3 standard deviations
        for n_std in [1, 2, 3]:
            width = 2 * n_std * np.sqrt(eigvals[0])
            height = 2 * n_std * np.sqrt(eigvals[1])
            
            ellipse = Ellipse(
                xy=mean_pca, 
                width=width, 
                height=height, 
                angle=angle,
                edgecolor=colors[cluster_id % 10],
                facecolor='none',
                alpha=0.8/n_std,  # Make outer ellipses more transparent
                linestyle=[':', '--', '-'][n_std-1],  # Different line styles for each std
                linewidth=2
            )
            plt.gca().add_patch(ellipse)
    
    plt.title('Agent Strategy Latent Clusters in PCA Space', fontsize=24)
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=24)
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=24)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_gaussians_pca.png'))
    plt.close()

def visualize_kmeans_clusters(latent_vectors, cluster_labels, cluster_centers, n_clusters, output_dir):
    """
    Visualize X-means clustering results
    
    Args:
        latent_vectors: The latent space vectors
        cluster_labels: Cluster labels from X-means
        cluster_centers: Cluster centers from X-means
        n_clusters: Number of clusters
        output_dir: Directory to save visualizations
    """
    # Compute PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    
    from matplotlib import cm
    colormap = cm.tab10  # Different colormap for X-means
    colors = colormap(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(
            latent_2d_pca[mask, 0], 
            latent_2d_pca[mask, 1], 
            label=f'Cluster {cluster_id}',
            color=colors[cluster_id % 10],  # Use modulo to handle more than 10 clusters
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
    
    plt.title(f'X-means Clusters in Latent Space (PCA) - k={n_clusters}')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xmeans_clusters_pca.png'))
    
    # Compute t-SNE visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d_tsne = tsne.fit_transform(latent_vectors)
    
    # Plot t-SNE clusters
    plt.figure(figsize=(12, 10))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(
            latent_2d_tsne[mask, 0], 
            latent_2d_tsne[mask, 1], 
            label=f'Cluster {cluster_id}',
            color=colors[cluster_id % 10],
            alpha=0.7
        )
    
    plt.title(f'X-means Clusters in Latent Space (t-SNE) - k={n_clusters}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xmeans_clusters_tsne.png'))

def evaluate_kmeans_clusters(cluster_labels, agent_labels, unique_agents, output_dir):
    """
    Evaluate how well X-means clusters align with ground truth agent types
    
    Args:
        cluster_labels: Cluster labels from X-means
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
    plt.xlabel('Predicted Agent Type (via X-means Clusters)')
    plt.ylabel('True Agent Type')
    plt.title(f'X-means Clustering Accuracy: {accuracy:.2%}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xmeans_confusion_matrix.png'))
    
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
    plt.title('X-means Clustering: Precision and Recall by Agent Type')
    plt.xticks(x, unique_agents)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xmeans_precision_recall.png'))
    
    # Create a cluster distribution analysis
    agent_cluster_counts = pd.crosstab(
        pd.Series(agent_labels, name='Agent'),
        pd.Series(cluster_labels, name='Cluster')
    )
    
    # Plot the distribution
    plt.figure(figsize=(12, 8))
    sns.heatmap(agent_cluster_counts, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Agent Distribution Across X-means Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_xmeans_distribution.png'))
    
    # Save the mapping and metrics to a text file
    with open(os.path.join(output_dir, 'xmeans_cluster_mapping.txt'), 'w') as f:
        f.write("X-MEANS CLUSTER TO AGENT MAPPING\n")
        f.write("===============================\n\n")
        for cluster_id, agent in cluster_to_agent.items():
            f.write(f"Cluster {cluster_id} -> {agent}\n")
        
        f.write(f"\nOverall accuracy: {accuracy:.2%}\n\n")
        
        f.write("Per-agent type metrics:\n")
        for agent in unique_agents:
            f.write(f"  {agent}:\n")
            f.write(f"    Precision: {precision_by_agent[agent]:.2%}\n")
            f.write(f"    Recall: {recall_by_agent[agent]:.2%}\n")
    
    return accuracy

def analyze_kmeans_clusters(info, output_dir, min_clusters=3, max_clusters=10):
    """
    Perform k-means clustering analysis and evaluate against ground truth
    
    Args:
        info: Tuple containing (df, latent_vectors, unique_agents, agent_labels, future_action_sequences)
        output_dir: Directory to save analysis results
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
    
    Returns:
        accuracy: Accuracy of X-means clustering compared to ground truth agent types
        cluster_labels: The cluster assignments
    """
    df, latent_vectors, unique_agents, agent_labels, _ = info
    
    # Create output directory for X-means
    kmeans_dir = os.path.join(output_dir, 'kmeans')
    os.makedirs(kmeans_dir, exist_ok=True)
    
    # Run k-means clustering
    print("Performing k-means clustering analysis...")
    cluster_labels, cluster_centers, optimal_k = kmeans_clustering(
        info, kmeans_dir, min_clusters, max_clusters
    )

    gaussians = fit_gaussians_to_clusters(latent_vectors, cluster_labels)
    
    # Visualize k-means clusters
    print("Visualizing k-means clusters...")
    visualize_kmeans_clusters(latent_vectors, cluster_labels, cluster_centers, optimal_k, kmeans_dir)

    visualize_cluster_gaussians(latent_vectors, cluster_labels, gaussians, kmeans_dir)
    
    # Evaluate k-means clusters against ground truth
    print("Evaluating k-means clusters against ground truth agent types...")
    accuracy = evaluate_kmeans_clusters(cluster_labels, agent_labels, unique_agents, kmeans_dir)
    
    print(f"k-means clustering analysis complete. Found {optimal_k} clusters with accuracy {accuracy:.2%}")
    
    # Add kmeans cluster labels to the dataframe
    df['kmeans_cluster'] = cluster_labels
    df.to_csv(os.path.join(kmeans_dir, 'latent_data_with_kmeans.csv'), index=False)

    import pickle
    with open(os.path.join(kmeans_dir, "gaussians.pkl"), "wb") as f:
        pickle.dump(gaussians, f)

    print(f"Saved Gaussian parameters to {os.path.join(kmeans_dir, 'gaussians.pkl')}")

    return accuracy, cluster_labels, gaussians
