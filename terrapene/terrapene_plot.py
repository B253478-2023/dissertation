import pysam
import os
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

    # Paths to the example files
    vcf_path = "terrapene.filtered.vcf.gz"
    popmap_path = "terrapene_popmap.csv"

    # Extract genotype matrix and population labels
    genotype_matrix, pop_labels, samples = vcf_to_matrix(vcf_path, popmap_path)

    label_encoder = LabelEncoder()
    pop_labels_encoded = label_encoder.fit_transform(pop_labels)


    n_clusters = 3
    Npc = 2
    max_iter = 500


    data = genotype_matrix
    labels = pop_labels_encoded
    obs_labels = pop_labels_encoded
    base = "terrapene_new"

    #print(labels)
    embeddings = {}

    # Apply Un-LDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-LDA-Km...")
    n_clusters = 14
    Npc = 50
    T0, G0, W0, _ = un_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                           center=True, gamma=1e-6)
    print(T0)
    print(G0)
    embeddings["Un-LDA-Km"] = {"T": T0, "W": W0, "G": G0}

    # Apply Un-RTLDA and obtain the reduced-dimensional representation and cluster assignments
    print("\nRunning Un-RTLDA...")
    n_clusters = 21
    Npc = 50
    T, G, W, _ = un_rtlda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                          center=True, gamma=1e-6)
    print(T)
    embeddings["Un-RTLDA"] = {"T": T, "W": W, "G": G}

    # Un-TRLDA
    print("\nRunning Un-TRLDA...")
    n_clusters = 2
    Npc = 300
    T2, G2, W2, _ = un_trlda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                             center=True)
    print(T2)
    embeddings["Un-TRLDA"] = {"T": T2, "W": W2, "G": G2}

    # SWULDA
    print("\nRunning SWULDA...")
    n_clusters = 4
    Npc = 100
    T3, G3, W3, _ = swulda(data, n_clusters, Npc=Npc, tol=1e-6, max_iter=max_iter, center=False)
    print(T3)
    embeddings["SWULDA"] = {"T": T3, "W": W3, "G": G3}

    # Un-RT(CD)LDA
    print("\nRunning Un-RT(CD)LDA...")
    n_clusters = 2
    Npc = 50
    T4, G4, W4, _ = un_rt_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                                 center=True, cd_clustering=True)
    print(T4)
    embeddings["Un-RT(CD)LDA"] = {"T": T4, "W": W4, "G": G4}

    # Un-TR(CD)LDA
    print("\nRunning Un-TR(CD)LDA...")
    n_clusters = 2
    Npc = 100
    T5, G5, W5, _ = un_tr_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, max_iter=max_iter, Ntry=10,
                                 center=True, cd_clustering=True)
    print(T5)
    embeddings["Un-TR(CD)LDA"] = {"T": T5, "W": W5, "G": G5}


    print("\nRunning Un-RT(A)LDA...")
    n_clusters = 21
    Npc = 50
    T7, G7, W7, _ = un_rtlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                               center=True, gamma=1e-6)
    print(T7)
    embeddings["Un-RT(A)LDA"] = {"T": T7, "W": W7, "G": G7}

    # Un-TRLDA
    print("\nRunning Un-TR(A)LDA...")
    n_clusters = 2
    Npc = 300
    T8, G8, W8, _ = un_trlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=500, Ntry=30,
                               center=True)
    print(T8)
    embeddings["Un-TR(A)LDA"] = {"T": T8, "W": W8, "G": G8}

    # sDAPC
    print("\nRunning sDAPC...")
    sdapc_results, _ = sdapc(data, labels=None, prop_pc_var=0.5, max_n_clust=5, n_pca_min=50, n_pca_max=300,
                             n_pca_interval=50)
    embeddings["Semisupervised-DAPC"] = sdapc_results["Semisupervised-DAPC"]

    sdapc_results, _ = sdapc(data, labels=obs_labels, prop_pc_var=0.5, max_n_clust=5, n_pca_min=50, n_pca_max=300,
                             n_pca_interval=50)

    embeddings["Supervised-DAPC"] = sdapc_results["Supervised-DAPC"]

    predicted_labels(embeddings, samples, filename=f"{base}.csv")
    # Call plot_embeddings on simulated data
    print("Plotting embeddings...")
    plot_embedded_clusters(embeddings, labels, filename=f"{base}_da.png")

    plot_pca_clusters(embeddings, data, labels, filename=f"{base}_pca.png")

    # Compute clustering performance metrics
    print("\nClustering metrics:")
    print_metrics(embeddings, labels, filename=f"{base}.txt")


def vcf_to_matrix(vcf_path, popmap_path):
    # Create an index with tabix if it doesn't exist
    if not (os.path.exists(vcf_path + '.tbi') or os.path.exists(vcf_path + '.csi')):
        pysam.tabix_index(vcf_path, preset='vcf')

    # Read popmap to create a dictionary for individual labels
    popmap = pd.read_csv(popmap_path, header=None, names=["ind", "pop"])
    popmap_dict = pd.Series(popmap["pop"].values, index=popmap["ind"]).to_dict()

    # Open VCF file using pysam
    vcf = pysam.VariantFile(vcf_path)
    samples = list(vcf.header.samples)
    num_individuals = len(samples)
    num_variants = sum(1 for _ in vcf.fetch())

    # Initialize matrix and labels list
    genotype_matrix = np.zeros((num_individuals, num_variants), dtype=int)
    pop_labels = [popmap_dict.get(sample, 'Unknown') for sample in samples]

    # Re-open VCF to iterate over it again
    vcf = pysam.VariantFile(vcf_path)
    variant_idx = 0
    for record in vcf.fetch():
        for ind_idx, ind in enumerate(samples):
            sample = record.samples[ind]
            gt = sample['GT']
            if gt == (0, 0):
                genotype_matrix[ind_idx, variant_idx] = 0  # Homozygous reference
            elif gt == (0, 1) or gt == (1, 0):
                genotype_matrix[ind_idx, variant_idx] = 1  # Heterozygous
            elif gt == (1, 1):
                genotype_matrix[ind_idx, variant_idx] = 2  # Homozygous alternate
        variant_idx += 1

    return genotype_matrix, pop_labels, samples


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


def predicted_labels(embeddings, samples, filename="predicted_labels.csv"):
    results = pd.DataFrame({"ind": samples})

    for method, emb in embeddings.items():
        T = emb["T"]
        G = emb["G"]

        # Ensure G is a numpy array
        G = np.array(G)

        print(f"Method: {method}, G shape: {G.shape}, Results shape: {results.shape}")

        if len(G) != len(results):
            raise ValueError(f"Length of G ({len(G)}) does not match length of results ({len(results)})")

        results[method + " Predicted Label"] = G

    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()