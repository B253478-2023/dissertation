import random
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, completeness_score
from unrtlda import *
from untrlda import *
from swulda_3 import *
from unrtcdlda import *
from untrcdlda import *

def main():

    # Generate synthetic data
    n_samples = 1000
    n_clusters = 9
    n_features = 100
    random_state = 1234
    dispersion = 18
    max_iter=20

    # generation base filename 
    base = f"c{n_clusters}_it{max_iter}_disp{dispersion}"

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

    # Apply Un-RTLDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-RTLDA...")
    T, G, W, _ = un_rtlda(data, n_clusters, Ninit=10, max_iter=max_iter, Ntry=10,
                            center=True, gamma=0.001)
    print(T)
    embeddings["Un-RTLDA"] = {"T": T, "W": W, "G": G}

    # Un-TRLDA
    print("\nRunning Un-TRLDA...")
    T2, G2, W2, _ = un_trlda(data, n_clusters, Ninit=10, max_iter=max_iter, Ntry=10,
                             center=True)
    print(T2)
    embeddings["Un-TRLDA"] = {"T": T2, "W": W2, "G": G2}

    #SWULDA
    print("\nRunning SWULDA...")
    T3, G3, W3, _ = swulda(data, n_clusters, max_iter=max_iter, center=False)
    print(T3)
    embeddings["SWULDA"] = {"T": T3, "W": W3, "G": G3}

    # Un-RT(CD)LDA
    print("\nRunning Un-RT(CD)LDA...")
    T4, G4, W4, _ = un_rt_cd_lda(data, n_clusters, Ninit=10, max_iter=100, Ntry=10,
                             center=True,cd_clustering=True)
    print(T4)
    embeddings["Un-RT(CD)LDA"] = {"T": T4, "W": W4, "G": G4}

    # Un-TR(CD)LDA
    print("\nRunning Un-TR(CD)LDA...")
    T5, G5, W5, _ = un_tr_cd_lda(data, n_clusters, Ninit=10, max_iter=100, Ntry=10,
                                 center=True, cd_clustering=True)
    print(T5)
    embeddings["Un-TR(CD)LDA"] = {"T": T5, "W": W5, "G": G5}

    # Call plot_embeddings on simulated data
    print("Plotting embeddings...")
    plot_embeddings(embeddings, data, labels, filename=f"{base}.pdf")

    # Compute clustering performance metrics
    print("\nClustering metrics:")
    print_metrics(embeddings, labels, filename=f"{base}.txt")


# legend not working
def plot_embeddings(embeddings, dataset, labels,
                    filename="embeddings_plots.pdf", no_pca=False):
    """
    Plot a grid of clusters and embeddings and save to a PDF.

    Args:
        embeddings (dict): Dictionary of embeddings with keys as method names
        and values as dicts with "T", "G", and "W".
        dataset (numpy array): Original dataset of shape (n_samples,
                               n_features).
        labels (list): Original population labels for each sample.
        filename (str): Name of the output PDF file containing the plots.
        no_pca (bool): If True, use the first two dimensions of the dataset
                       instead of PCA.
    """
    if no_pca:
        X = dataset[:, :2]
    else:
        pca = PCA(n_components=2)
        X = pca.fit_transform(dataset)

    # original data in PCA space
    df = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(X.shape[1])])
    df["Original_Population"] = labels

    n_embeddings = len(embeddings)
    n_cols = n_embeddings
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 10))

    for idx, (method, emb) in enumerate(embeddings.items()):
        T = emb["T"]
        G = emb["G"]
        df2 = pd.DataFrame(T, columns=[f"DA{i+1}" for i in range(T.shape[1])])
        df2["Cluster"] = G
        df2["Original_Population"] = labels

        # Plot clusters in PCA space
        ax = axes[0, idx]
        sns.scatterplot(ax=ax, data=df, x="PC1", y="PC2", hue=G, style="Original_Population", palette="deep", legend=False)
        ax.set_title(f"{method} Clusters on PCA Embeddings")

        # Plot clusters in embedded space
        ax = axes[1, idx]
        if T.shape[1] > 1:
            sns.scatterplot(ax=ax, data=df2, x="DA1", y="DA2", hue="Cluster", style="Original_Population", palette="deep", legend=False)
            ax.set_title(f"{method} Embeddings")
        else:
            sns.kdeplot(ax=ax, x="DA1", hue="Cluster", data=df2, fill=None, common_norm=False, palette="deep", zorder=1)
            df2["y"] = 0.1
            sns.scatterplot(ax=ax, data=df2, x="DA1", y="y", hue="Cluster", style="Original_Population", palette="deep", legend=False)
            ax.legend(fontsize="small")
            ax.set_title(f"{method} Embeddings (1 DA Axis)")

    # Gather legend handles and labels from the last plot to use for the figure legend
    handles, labels = ax.get_legend_handles_labels()

    # Add a single legend at the top of the figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=n_cols)

    # Adjust layout and save the plot to a PDF
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)
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
