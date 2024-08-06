import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, \
    completeness_score
from matplotlib.patches import Ellipse
from unlda import *
from unrtlda import *
from unrtlda_a import *
from untrlda import *
from untrlda_a import *
from swulda import *
from unrtcdlda import *
from untrcdlda import *
from unkfdapc import *
from sdapc import *



def main():
    # generation base filename

    labels_insular = pd.read_csv('../coala/labels_insular_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                     1]
    labels_cline = pd.read_csv('../coala/labels_cline_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                   1]
    labels_weak = pd.read_csv('../coala/labels_weak_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 1]
    labels_strong = pd.read_csv('../coala/labels_strong_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                       1]

    obs_labels_insular = pd.read_csv('../coala/labels_insular_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                       2]
    obs_labels_cline = pd.read_csv('../coala/labels_cline_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                       2]
    obs_labels_weak = pd.read_csv('../coala/labels_weak_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                      2]
    obs_labels_strong = pd.read_csv('../coala/labels_strong_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:,
                        2]

    insulardata = pd.read_csv('../coala/sim_insular_div_0.9_rep_1.csv', index_col=0).values
    clinedata = pd.read_csv('../coala/sim_cline_div_0.9_rep_1.csv', index_col=0).values
    weakdata = pd.read_csv('../coala/sim_weak_div_0.9_rep_1.csv', index_col=0).values
    strongdata = pd.read_csv('../coala/sim_strong_div_0.9_rep_1.csv', index_col=0).values


    n_clusters = 3
    Npc = 2
    max_iter = 500


    data = insulardata
    labels = labels_insular
    obs_labels = obs_labels_insular
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    obs_labels = label_encoder.fit_transform(obs_labels)
    base = "coala_insular"

    #print(labels)
    embeddings = {}

    # Apply Un-LDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-LDA-Km...")
    n_clusters = 3
    Npc = 2
    T0, G0, W0, _ = un_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                           center=True, gamma=1e-6)
    print(T0)
    print(G0)
    embeddings["Un-LDA-Km"] = {"T": T0, "W": W0, "G": G0}

    # Apply Un-RTLDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-RTLDA...")
    n_clusters = 3
    Npc = 2
    T, G, W, _ = un_rtlda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                          center=True, gamma=1e-6)
    print(T)
    embeddings["Un-RTLDA"] = {"T": T, "W": W, "G": G}

    # Un-TRLDA
    print("\nRunning Un-TRLDA...")
    n_clusters = 2
    Npc = 7
    T2, G2, W2, _ = un_trlda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                             center=True)
    print(T2)
    embeddings["Un-TRLDA"] = {"T": T2, "W": W2, "G": G2}

    # SWULDA
    print("\nRunning SWULDA...")
    n_clusters = 3
    Npc = 2
    T3, G3, W3, _ = swulda(data, n_clusters, Npc=Npc, tol=1e-6, max_iter=max_iter, center=False)
    print(T3)
    embeddings["SWULDA"] = {"T": T3, "W": W3, "G": G3}

    # Un-RT(CD)LDA
    print("\nRunning Un-RT(CD)LDA...")
    n_clusters = 3
    Npc = 3
    T4, G4, W4, _ = un_rt_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                                 center=True, cd_clustering=True)
    print(T4)
    embeddings["Un-RT(CD)LDA"] = {"T": T4, "W": W4, "G": G4}

    # Un-TR(CD)LDA
    print("\nRunning Un-TR(CD)LDA...")
    n_clusters = 2
    Npc = 3
    T5, G5, W5, _ = un_tr_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, max_iter=max_iter, Ntry=10,
                                 center=True, cd_clustering=True)
    print(T5)
    embeddings["Un-TR(CD)LDA"] = {"T": T5, "W": W5, "G": G5}

    print("\nRunning Un-KFDAPC...")
    n_clusters = 2
    Npc = 2
    T6, G6, W6, _ = unkfdapc(data, n_clusters, Npc=Npc, Ninit=50, gamma=1e-6, tol=1e-8, max_iter=max_iter, Ntry=50,
                             center=True, no_pca=False, alpha=1.0, beta=1.0, sigma=0.1, mu=1e-12,
                             lambda_param=1e8)
    print(T6)
    embeddings["Un-KFDA"] = {"T": T6, "W": W6, "G": G6}

    print("\nRunning Un-RT(A)LDA...")
    n_clusters = 3
    Npc = 2
    T7, G7, W7, _ = un_rtlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                               center=True, gamma=1e-6)
    print(T7)
    embeddings["Un-RT(A)LDA"] = {"T": T7, "W": W7, "G": G7}

    # Un-TRLDA
    print("\nRunning Un-TR(A)LDA...")
    n_clusters = 2
    Npc = 7
    T8, G8, W8, _ = un_trlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=500, Ntry=30,
                               center=True)
    print(T8)
    embeddings["Un-TR(A)LDA"] = {"T": T8, "W": W8, "G": G8}

    # sDAPC
    print("\nRunning sDAPC...")
    sdapc_results, _ = sdapc(data, labels=None, prop_pc_var=0.5, max_n_clust=6, n_pca_min=10, n_pca_max=100,
                             n_pca_interval=10)
    embeddings["Semisupervised-DAPC"] = sdapc_results["Semisupervised-DAPC"]

    sdapc_results, _ = sdapc(data, labels=obs_labels, prop_pc_var=0.5, max_n_clust=6, n_pca_min=10, n_pca_max=100,
                             n_pca_interval=10)

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
        df2 = pd.DataFrame(T, columns=[f"DA{i + 1}" for i in range(T.shape[1])])
        df2["Cluster"] = G
        df2["Original_Population"] = labels

        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]
        if T.shape[1] > 1:
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