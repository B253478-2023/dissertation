import numpy as np
import scipy
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.linalg import eigh
from sklearn.cluster import AgglomerativeClustering

def gevd(A, B):
    """
    Generalized eigendecomposition of two symmetric square matrices A and B.

    Args:
        A (numpy array): A symmetric square matrix of shape (n, n).
        B (numpy array): A symmetric square matrix of shape (n, n).

    Returns:
        dict: A dictionary containing the sorted eigenvectors ('W') and a diagonal matrix of the sorted eigenvalues ('D').
    """
    # Compute the generalized eigenvectors and eigenvalues
    eigvals, eigvecs = eigh(A, B)

    # Sort the eigenvalues and eigenvectors in ascending order
    ind = np.argsort(eigvals)
    eigvals_sorted = eigvals[ind]
    eigvecs_sorted = eigvecs[:, ind]

    # Create a dictionary with the sorted eigenvectors and eigenvalues
    model = {'W': eigvecs_sorted, 'D': np.diag(eigvals_sorted)}

    return model


# Define the Un-RTLDA function
def un_rtlda_a(X, c, Npc, Ninit=10, gamma=1e-6, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False):
    """
    Implement the Unsupervised Ratio-Trade Linear Discriminant Analysis (Un-RTLDA) algorithm for clustering.

    Args:
        X (numpy array): Input data of shape (n_samples, n_features).
        c (int): Number of clusters.
        Ninit (int, optional): Number of initializations for KMeans. Defaults to 10.
        gamma (float, optional): Regularization parameter for the within-class scatter matrix. Defaults to 1e-6.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        Ntry (int, optional): Number of attempts to find the best clustering. Defaults to 10.
        center (bool, optional): Whether to center the data. Defaults to True.
        no_pca (bool, optional): Whether to disable PCA initialization. Defaults to False.

    Returns:
        T (numpy array): Un-RTLDA embeddings of shape (n_samples, n_components).
        Ypre (list): Cluster assignments for each sample.
        W2 (numpy array): Eigenvectors matrix of shape (n_features, n_components).
    """
    n, d = X.shape  # Number of samples

    if center:
        H = np.eye(n) - np.ones((n, n)) / n
    else:
        H = np.eye(n)

    St = X.T @ H @ X  # Compute the within-class scatter matrix St
    Stt = St + gamma * np.eye(d)  # Add regularization term to St

    it = 0  # Initialize the iteration counter

    obj_old = -np.inf  # Initialize the old objective value
    obj_new = 0.0  # Initialize the new objective value
    Ypre = None  # Initialize the predicted cluster labels
    T_old = None
    W2_old = None
    T = None

    # Initialize W using PCA
    # m = min(d, c - 1, Npc)
    m = Npc
    pca = PCA(n_components=m)

    if no_pca:
        W = X.T[:, :m]
    else:
        W = pca.fit_transform(X.T @ H)
    W2 = W  # Initialize W2 with W

    obj_log = []

    # Iterate until convergence or maxIter is reached
    while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:

        it += 1
        obj_old = obj_new

        # Calculate the intermediate matrix product
        T = (scipy.linalg.expm(-0.5 * np.linalg.inv(W2.T @ Stt @ W2)) @ W2.T @ X.T @ H).T
        # T = (fractional_matrix_power(W2.T @ Stt @ W2, -0.5) @ W2.T @ X.T @ H).T

        best_obj_tmp = float('inf')
        best_Ypre = None


        clustering = AgglomerativeClustering(n_clusters=c)  # Initialize Agglomerative clustering
        Ypre = clustering.fit_predict(T)  # Cluster the data and obtain labels

        # Update Yp matrix
        Yp = np.eye(c)[Ypre]

        # Compute the between-class scatter matrix Sb
        Sb = X.T @ H @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ H.T @ X

        # Perform generalized eigenvalue decomposition and update W2
        model = gevd(Sb, Stt)
        W2 = model['W'][:, -m:]

        # Update the new objective value
        # obj_new = np.trace((W2.T @ Stt @ W2) ** -1 @ W2.T @ Sb @ W2)
        pinv_term = np.linalg.pinv(W2.T @ Stt @ W2)
        obj_new = np.trace(pinv_term @ W2.T @ Sb @ W2)

        obj_log.append(obj_new)

    # Print a warning if the algorithm did not converge within maxIter iterations
    if it == max_iter:
        print(f"Warning: The un_rtlda did not converge within {max_iter} iterations!")

    return T, Ypre, W2, obj_log