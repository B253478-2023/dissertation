import numpy as np
import pandas as pd
from Bio.PopGen.GenePop import read
from collections import defaultdict
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
from swulda import *
from unrtcdlda import *
from untrcdlda import *
# unkfdapc import *
def main():

    labels = pd.read_csv('../H3N2/H3N2pop.csv', index_col=0).values.ravel()
    H3N2data = pd.read_csv('../H3N2/H3N2data.csv', index_col=0).values
    print('labels array:', labels.dtype, labels.shape)
    print('H3N2data array:', H3N2data.dtype, H3N2data.shape)
    n_clusters = 6
    Npc = 150
    max_iter = 100

    test(H3N2data, labels, n_clusters, Npc, max_iter, 'H3N2data')
def test(data,labels,n_clusters,Npc,max_iter,datetype):

    embeddings = {}

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
    #print("\nRunning Un-RT(CD)LDA...")
    #T4, G4, W4, _ = un_rt_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
    #                         center=True,cd_clustering=True)
    #print(T4)
    #embeddings["Un-RT(CD)LDA"] = {"T": T4, "W": W4, "G": G4}

    # Un-TR(CD)LDA
    #print("\nRunning Un-TR(CD)LDA...")
    #T5, G5, W5, _ = un_tr_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, max_iter=max_iter, Ntry=10,
    #                             center=True, cd_clustering=True)
    #print(T5)
    #embeddings["Un-TR(CD)LDA"] = {"T": T5, "W": W5, "G": G5}

    #print("\nRunning Un-KFDAPC...")
    #T6, G6, W6, _ = unkfdapc(data, n_clusters, Ninit=10, gamma=1e-6, tol=1e-6, max_iter=max_iter, Ntry=10, center=True, no_pca=False, alpha=0.5, beta=0.5, sigma=1.0,
    #          mu=1e-12, lambda_param=1e8)
    #print(T6)
    #embeddings["Un-KFDAPC"] = {"T": T6, "W": W6, "G": G6}

    # Call plot_embeddings on simulated data
    print("Plotting embeddings...")
    plot_embeddings(embeddings, data, labels, filename=f"{datetype}_maxiter={max_iter}.pdf")

    # Compute clustering performance metrics
    print("\nClustering metrics:")
    print_metrics(embeddings, labels, file_name=f'{datetype}_maxiter={max_iter}_results.txt')


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


def print_metrics(embeddings, labels, file_name="metrics_results.txt"):
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
    with open(file_name, 'w') as f:
        f.write(results_df.to_string())

if __name__ == "__main__":
    main()