import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, completeness_score
from unlda import *
from unrtlda import *
from unrtlda_a import *
from untrlda import *
from Methods.untrlda_a import *
from Methods.swulda import *
from unrtcdlda import *
from untrcdlda import *
from sdapc import *
from Methods.unkfdapc import *
def main():

    # Generate synthetic data
    n_samples = 100
    n_clusters = 3
    n_features = 100
    random_state = 1234
    dispersion = 4
    max_iter=20
    Npc = 4

    # generation base filename 
    base = f"new_n{n_samples}_c{n_clusters}_it{max_iter}_disp{dispersion}"

    random.seed(random_state)
    np.random.seed(random_state)

    print("Generating synthetic data...")
    data, labels = generate_synthetic_data(n_samples=n_samples,
                                           n_clusters=n_clusters,
                                           n_features=n_features,
                                           random_state=random_state,
                                           dispersion=dispersion)
    print(data)

    embeddings = {}


    # Apply Un-LDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-LDA-Km...")
    T0, G0, W0, _ = un_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                          center=True, gamma=1e-6)
    print(T0)
    print(G0)
    embeddings["Un-LDA-Km"] = {"T": T0, "W": W0, "G": G0}

    # Apply Un-RTLDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-RTLDA...")
    T, G, W, _ = un_rtlda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                            center=True, gamma=1e-6)
    print(T)
    embeddings["Un-RTLDA"] = {"T": T, "W": W, "G": G}

    # Un-TRLDA
    print("\nRunning Un-TRLDA...")
    T2, G2, W2, _ = un_trlda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                             center=True)
    print(T2)
    embeddings["Un-TRLDA"] = {"T": T2, "W": W2, "G": G2}

    #SWULDA
    print("\nRunning SWULDA...")
    T3, G3, W3, _ = swulda(data, n_clusters, Npc=Npc, tol=1e-6, max_iter=max_iter, center=False)
    print(T3)
    embeddings["SWULDA"] = {"T": T3, "W": W3, "G": G3}

    # Un-RT(CD)LDA
    print("\nRunning Un-RT(CD)LDA...")
    T4, G4, W4, _ = un_rt_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                             center=True,cd_clustering=True)
    print(T4)
    embeddings["Un-RT(CD)LDA"] = {"T": T4, "W": W4, "G": G4}

    # Un-TR(CD)LDA
    print("\nRunning Un-TR(CD)LDA...")
    T5, G5, W5, _ = un_tr_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, max_iter=max_iter, Ntry=10,
                                 center=True, cd_clustering=True)
    print(T5)
    embeddings["Un-TR(CD)LDA"] = {"T": T5, "W": W5, "G": G5}

    print("\nRunning Un-RT(A)LDA...")
    T7, G7, W7, _ = un_rtlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                          center=True, gamma=1e-6)
    print(T7)
    embeddings["Un-RT(A)LDA"] = {"T": T7, "W": W7, "G": G7}

    # Un-TRLDA
    print("\nRunning Un-TR(A)LDA...")
    T8, G8, W8, _ = un_trlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=500, Ntry=30,
                             center=True)
    print(T8)
    embeddings["Un-TR(A)LDA"] = {"T": T8, "W": W8, "G": G8}

    # sDAPC
    print("\nRunning sDAPC...")
    sdapc_results,_ = sdapc(data, labels=None, prop_pc_var=0.5, max_n_clust=6, n_pca_min=10, n_pca_max=100,
                          n_pca_interval=10)
    embeddings["Semisupervised-DAPC"] = sdapc_results["Semisupervised-DAPC"]

    sdapc_results,_ = sdapc(data, labels=labels, prop_pc_var=0.5, max_n_clust=6, n_pca_min=10, n_pca_max=100, n_pca_interval=10)

    embeddings["Supervised-DAPC"] = sdapc_results["Supervised-DAPC"]

    # Call plot_embeddings on simulated data
    print("Plotting embeddings...")
    plot_embedded_clusters(embeddings, labels, filename=f"{base}_da.png")
    plot_pca_clusters(embeddings, data, labels, filename=f"{base}_pca.png")

    # Compute clustering performance metrics
    print("\nClustering metrics:")
    print_metrics(embeddings, labels, filename=f"{base}.txt")


# legend not working

def plot_pca_clusters(embeddings, dataset, labels, filename="pca_clusters.png", no_pca=False):
    """
    Plot clusters in PCA space and save to a PNG file.

    Args:
        embeddings (dict): Dictionary of embeddings with keys as method names
        and values as dicts with "T", "G", and "W".
        dataset (numpy array): Original dataset of shape (n_samples, n_features).
        labels (list): Original population labels for each sample.
        filename (str): Name of the output PNG file containing the plots.
        no_pca (bool): If True, use the first two dimensions of the dataset instead of PCA.
    """
    if no_pca:
        X = dataset[:, :2]
    else:
        pca = PCA(n_components=2)
        X = pca.fit_transform(dataset)

    df = pd.DataFrame(X, columns=[f"PC{i + 1}" for i in range(X.shape[1])])
    df["Original_Population"] = labels

    n_embeddings = len(embeddings)
    n_cols = 3
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for idx, (method, emb) in enumerate(embeddings.items()):
        G = emb["G"]
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]
        scatter = sns.scatterplot(ax=ax, data=df, x="PC1", y="PC2", hue=G, style="Original_Population", palette="deep",
                                  legend=False)
        ax.set_title(f"{method} Clusters on PCA Embeddings")
        ax.set_aspect('equal')

        # Adding cluster labels
        for cluster in np.unique(G):
            cluster_data = df[G == cluster]
            centroid = cluster_data.mean(axis=0)
            ax.text(centroid["PC1"], centroid["PC2"], str(cluster),
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                    ha='center', va='center', fontsize=10, weight='bold')

            # Drawing ellipses
            cov = np.cov(cluster_data[['PC1', 'PC2']].values.T)
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                print(f"Skipping ellipse for cluster {cluster} due to invalid covariance matrix.")
                continue
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = plt.matplotlib.patches.Ellipse(xy=(centroid["PC1"], centroid["PC2"]),
                                                 width=lambda_[0] * 2, height=lambda_[1] * 2,
                                                 angle=np.rad2deg(np.arccos(v[0, 0])), color='black', fill=False)
            ax.add_artist(ell)

    # Remove empty subplots
    for i in range(n_embeddings, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, format='png')
    plt.close(fig)


def plot_embedded_clusters(embeddings, labels, filename="embedded_clusters.png"):
    """
    Plot clusters in embedded space and save to a PNG file.

    Args:
        embeddings (dict): Dictionary of embeddings with keys as method names
        and values as dicts with "T", "G", and "W".
        labels (list): Original population labels for each sample.
        filename (str): Name of the output PNG file containing the plots.
    """
    n_embeddings = len(embeddings)
    n_cols = 3
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for idx, (method, emb) in enumerate(embeddings.items()):
        T = emb["T"]
        G = emb["G"]
        W = emb.get("W", None)

        # Debugging: Log the shapes and contents of the matrices
        print(f"Method: {method}")
        print(f"T shape: {T.shape}")

        if method not in ["Semisupervised-DAPC", "Supervised-DAPC"]:
            if W is not None:
                print(f"W shape: {W.shape}")

                # Compute eigenvalues (variance captured by each axis)
                eigenvalues = np.var(T, axis=0)
                sorted_indices = np.argsort(eigenvalues)[::-1]

                # Ensure we only select available dimensions within bounds of T
                sorted_indices = [i for i in sorted_indices if i < T.shape[1]]
                max_dims = min(2, len(sorted_indices))
                sorted_indices = sorted_indices[:max_dims]

                # Debugging: Log the final sorted indices
                print(f"Final sorted indices: {sorted_indices}")

                sorted_T = T[:, sorted_indices]
            else:
                # If W is None, use the first two dimensions by default
                print("W is None, using first two dimensions of T")
                sorted_T = T[:, :2]
        else:
            # For "Semisupervised-DAPC" and "Supervised-DAPC", use T directly
            sorted_T = T[:, :2]

        df2 = pd.DataFrame(sorted_T, columns=[f"DA{i + 1}" for i in range(sorted_T.shape[1])])
        df2["Cluster"] = G
        df2["Original_Population"] = labels

        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]
        if sorted_T.shape[1] > 1:
            scatter = sns.scatterplot(ax=ax, data=df2, x="DA1", y="DA2", hue="Cluster", style="Original_Population",
                                      palette="deep", legend=False)
            ax.set_title(f"{method} Embeddings")

            # Adding cluster labels and ellipses
            for cluster in np.unique(G):
                cluster_data = df2[df2["Cluster"] == cluster]
                centroid = cluster_data.mean(axis=0)
                ax.text(centroid["DA1"], centroid["DA2"], str(cluster),
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                        ha='center', va='center', fontsize=10, weight='bold')

                # Drawing ellipses
                cov = np.cov(cluster_data[['DA1', 'DA2']].values.T)
                if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                    print(f"Skipping ellipse for cluster {cluster} due to invalid covariance matrix.")
                    continue
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                ell = Ellipse(xy=(centroid["DA1"], centroid["DA2"]),
                              width=lambda_[0] * 2, height=lambda_[1] * 2,
                              angle=np.rad2deg(np.arccos(v[0, 0])), color='black', fill=False)
                ax.add_artist(ell)
        else:
            sns.kdeplot(ax=ax, x="DA1", hue="Cluster", data=df2, fill=None, common_norm=False, palette="deep", zorder=1)
            df2["y"] = 0.1
            sns.scatterplot(ax=ax, data=df2, x="DA1", y="y", hue="Cluster", style="Original_Population", palette="deep",
                            legend=False)
            ax.legend(fontsize="small")
            ax.set_title(f"{method} Embeddings (1 DA Axis)")

    # Remove empty subplots
    for i in range(n_embeddings, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, format='png')
    plt.close(fig)


def print_metrics(embeddings, labels, filename="metrics_results.txt"):
    """
    Calculate and print various clustering metrics for multiple embeddings.

    Args:
        embeddings (dict): Dictionary of embeddings with keys as method names
                           and values as dicts with "T", "G", and "W".
        labels (list): True labels for each sample.

    Returns:
        None
    """
    metrics = ['Adjusted Rand Index', 'Normalized Mutual Information',
               'Silhouette Score', 'Fowlkes-Mallows Index',
               'Completeness Score']
    results = {metric: [] for metric in metrics}
    embedding_names = []

    for method, emb in embeddings.items():
        T = emb["T"]
        G = emb["G"]

        ari = adjusted_rand_score(labels, G)
        nmi = normalized_mutual_info_score(labels, G)
        silhouette = silhouette_score(T, G)
        fmi = fowlkes_mallows_score(labels, G)
        completeness = completeness_score(labels, G)

        results['Adjusted Rand Index'].append(ari)
        results['Normalized Mutual Information'].append(nmi)
        results['Silhouette Score'].append(silhouette)
        results['Fowlkes-Mallows Index'].append(fmi)
        results['Completeness Score'].append(completeness)

        embedding_names.append(method)

    results_df = pd.DataFrame(results, index=embedding_names).T
    print(results_df)

    # Export to txt file
    with open(filename, 'w') as f:
        f.write(results_df.to_string())

def generate_synthetic_data(n_samples=1000, n_clusters=4, n_features=50,
                            random_state=None, dispersion=1):
    """
    Generate synthetic data with specified number of samples, clusters, and
    features.

    Args:
        n_samples (int, optional): The number of samples in the generated
                                   dataset. Defaults to 1000.
        n_clusters (int, optional): The number of clusters in the
                                    generated dataset. Defaults to 4.
        n_features (int, optional): The number of features in the generated
                                    dataset. Defaults to 50.
        random_state (int, optional): The random seed for reproducibility.
                                      Defaults to None.
        dispersion (float, optional): The dispersion of the clusters. Controls
                                      the standard deviation of the clusters.
                                      Defaults to 1.

    Returns:
        tuple: A tuple containing the generated data (numpy array) and the
               corresponding labels (numpy array).
    """
    # Define the minimum and maximum standard deviation for the clusters
    min_std = 0.1
    max_std = 1

    # Calculate the standard deviation for the clusters based on the
    # dispersion parameter
    cluster_std = min_std + (max_std - min_std) * dispersion

    # Generate synthetic data with `n_clusters` clusters
    data, labels = make_blobs(n_samples=n_samples, centers=n_clusters,
                              n_features=n_features, random_state=random_state,
                              cluster_std=cluster_std)

    # Standardize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, labels


if __name__ == "__main__":
    main()
