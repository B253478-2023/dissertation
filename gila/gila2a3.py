import pysam
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, completeness_score
from unlda import *
from unrtlda import *
from unrtlda_a import *
from untrlda import *
from untrlda_a import *
from swulda import *
from unrtcdlda import *
from untrcdlda import *
from unkfdapc import *
def main():

    # Paths to the example files
    vcf_path = "fig2a.final.vcf.gz"
    popmap_path = "fig2a.popmap.csv"

    # Extract genotype matrix and population labels
    genotype_matrix, pop_labels = vcf_to_matrix(vcf_path, popmap_path)

    label_encoder = LabelEncoder()
    pop_labels = label_encoder.fit_transform(pop_labels)

    # Print the shape of the matrix and some labels as a sanity check
    print("Genotype Matrix Shape:", genotype_matrix.shape)
    print("Genotype Matrix:\n", genotype_matrix)
    print("Population Labels:", pop_labels[:10])

    max_iter = 500
    k_range = range(2, 6)
    Npc_range = range(20, 301, 20)

    #grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_rtlda')
    #grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_trlda')
    grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_lda')

    #grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_rtalda')
    #grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_tralda')
    grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='swulda')

    #grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_rtcdlda')
    #grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_trcdlda')
    grid_search_clustering(genotype_matrix, pop_labels, k_range, Npc_range, max_iter, datatype='gila2a', method='un_kfdapc')
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

    return genotype_matrix, pop_labels

def grid_search_clustering(data, labels, k_range, Npc_range, max_iter,datatype='Island', method='un_rtlda', n_splits=5):
    """
    Performs a grid search over number of clusters and number of principal components with k-fold cross-validation.

    Args:
        data (array-like): The dataset to cluster.
        labels (array-like): The true labels for computing evaluation metrics.
        k_range (range): A range of values for the number of clusters.
        Npc_range (range): A range of values for the number of principal components to retain.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        dict: Best parameters based on silhouette score and other metrics.
    """
    silhouette_best_score = -1
    silhouette_best_params = {'nPC': None, 'k': None}

    results = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for Npc in Npc_range:
        for k in k_range:
            print('Npc:', Npc, 'k:', k)
            nmi_scores = []
            ari_scores = []
            silhouette_scores = []
            fmi_scores = []
            completeness_scores = []

            for train_index, test_index in kf.split(data):
                train_data, test_data = data[train_index], data[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]

                nmi, ari, silhouette, fmi, completeness = test(train_data, train_labels, k, Npc, max_iter, method)

                nmi_scores.append(nmi)
                ari_scores.append(ari)
                silhouette_scores.append(silhouette)
                fmi_scores.append(fmi)
                completeness_scores.append(completeness)

            avg_nmi = np.mean(nmi_scores)
            avg_ari = np.mean(ari_scores)
            avg_silhouette = np.mean(silhouette_scores)
            avg_fmi = np.mean(fmi_scores)
            avg_completeness = np.mean(completeness_scores)

            results.append({
                'Npc': Npc,
                'k': k,
                'NMI': avg_nmi,
                'ARI': avg_ari,
                'Silhouette': avg_silhouette,
                'fmi': avg_fmi,
                'completeness': avg_completeness
            })

            # Update the best params based on a chosen metric, e.g., nmi
            if avg_silhouette > silhouette_best_score:
                silhouette_best_score = avg_silhouette
                silhouette_best_params = {'silhouette best params: Npc': Npc, 'k': k, 'NMI': avg_nmi, 'ARI': avg_ari,
                                   'Silhouette': avg_silhouette, 'fmi': avg_fmi, 'completeness': avg_completeness}

    with open(f'{datatype}_{method}_grid_search_results.txt', 'w') as f:
        f.write(json.dumps(silhouette_best_params) + "\n")
        f.flush()
        for result in results:
            f.write(json.dumps(result) + "\n")
            f.flush()

    return silhouette_best_params, results

def test(data, labels, n_clusters, Npc, max_iter, method='un_rtlda'):

    embeddings = {}
    # Apply Un-RTLDA and obtain the reduced-dimensional representation and cluster assignments
    if method == 'un_rtlda':
        print("\nRunning Un-RTLDA...")
        T, G, W, _ = un_rtlda(data, n_clusters, Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                                center=True, gamma=1e-6)
        #print(T)
        embeddings["Un-RTLDA"] = {"T": T, "W": W, "G": G}
    elif method == 'un_trlda':
    # Un-TRLDA
        print("\nRunning Un-TRLDA...")
        T2, G2, W2, _ = un_trlda(data, n_clusters, Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                                 center=True)
        #print(T2)
        embeddings["Un-TRLDA"] = {"T": T2, "W": W2, "G": G2}

    elif method == 'swulda':
        #SWULDA
        print("\nRunning SWULDA...")
        T3, G3, W3, _ = swulda(data, n_clusters, Npc, tol=1e-6, max_iter=max_iter, center=True)
        #print(T3)
        embeddings["SWULDA"] = {"T": T3, "W": W3, "G": G3}

    elif method == 'un_lda':
        # Apply Un-LDA and obtain the reduced-dimensional representation and cluster assignments
        print("\nRunning Un-LDA...")
        T0, G0, W0, _ = un_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                              center=True, gamma=1e-6)
        #print(T0)
        embeddings["Un-LDA"] = {"T": T0, "W": W0, "G": G0}
    elif method == 'un_rtcdlda':
        # Un-RT(CD)LDA
        print("\nRunning Un-RT(CD)LDA...")
        T4, G4, W4, _ = un_rt_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                                 center=True,cd_clustering=True)
        #print(T4)
        embeddings["Un-RT(CD)LDA"] = {"T": T4, "W": W4, "G": G4}

    elif method == 'un_trcdlda':
        # Un-TR(CD)LDA
        print("\nRunning Un-TR(CD)LDA...")
        T5, G5, W5, _ = un_tr_cd_lda(data, n_clusters, Npc=Npc, Ninit=100, max_iter=max_iter, Ntry=10,
                                     center=True, cd_clustering=True)
        #print(T5)
        embeddings["Un-TR(CD)LDA"] = {"T": T5, "W": W5, "G": G5}

    elif method == 'un_kfdapc':
        print("\nRunning Un-KFDAPC...")
        T6, G6, W6, _ = unkfdapc(data, n_clusters, Npc=Npc, Ninit=50, gamma=1e-6, tol=1e-8, max_iter=max_iter, Ntry=50,
                                       center=True, no_pca=False, alpha=1.0, beta=1.0, sigma=0.1, mu=1e-12,
                                       lambda_param=1e8)
        #print(T6)
        embeddings["Un-KFDAPC"] = {"T": T6, "W": W6, "G": G6}

    elif method == 'un_rtalda':
        print("\nRunning Un-RTLDA_A...")
        T7, G7, W7, _ = un_rtlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=max_iter, Ntry=30,
                              center=True, gamma=1e-6)
        #print(T7)
        embeddings["Un-RTLDA_A"] = {"T": T7, "W": W7, "G": G7}

    elif method == 'un_tralda':
        # Un-TRLDA-A
        print("\nRunning Un-TRLDA_A...")
        T8, G8, W8, _ = un_trlda_a(data, n_clusters, Npc=Npc, Ninit=100, tol=1e-6, max_iter=500, Ntry=30,
                                 center=True)
        #print(T8)
        embeddings["Un-TRLDA_A"] = {"T": T8, "W": W8, "G": G8}

    # Call plot_embeddings on simulated data
    #print("Plotting embeddings...")
    #plot_embeddings(embeddings, data, labels, filename=f"{datetype}_maxiter={max_iter}.pdf")

    # Compute clustering performance metrics
    #print("\nClustering metrics:")
    nmi, ari, silhouette, fmi, completeness = print_metrics(embeddings, labels, file_name=f'{method}_maxiter={max_iter}_results.txt')

    return nmi, ari, silhouette, fmi, completeness

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
    #with open(file_name, 'w') as f:
    #    f.write(results_df.to_string())

    return nmi, ari, silhouette, fmi, completeness

if __name__ == "__main__":
    main()