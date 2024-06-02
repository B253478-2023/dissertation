import sys
import numpy as np
from numpy.linalg import inv, eig
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def swulda(X, c, tol=1e-6, max_iter=100, center=True, no_pca=False):
    """
    Implement the Self-Weighted Unsupervised Linear Discriminant Analysis (SWULDA) algorithm for clustering.

    Args:
        X (numpy array): Input data of shape (n_samples, n_features).
        c (int): Number of clusters.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        center (bool, optional): Whether to center the data. Defaults to True.
        no_pca (bool, optional): Whether to disable PCA initialization. Defaults to False.

    Returns:
        T (numpy array): SWULDA embeddings of shape (n_samples, n_components).
        Ypre (list): Cluster assignments for each sample.
        W (numpy array): Eigenvectors matrix of shape (n_features, n_components).
    """

    n, d = X.shape  # Number of samples

    if center:
        X = X - np.mean(X, axis=0)  # 中心化处理

    St = X.T @ X # Compute the total scatter matrix St

    it = 0  # Initialize the iteration counter

    obj_old = -np.inf  # Initialize the old objective value
    obj_new = 0.0 # Initialize the new objective value
    Ypre = None  # Initialize the predicted cluster labels
    T = None

    # Initialize random indicator matrix G
    G = np.zeros((n, c))
    rand_indices = np.random.choice(c, n)
    G[np.arange(n), rand_indices] = 1

    # Initialize random weight parameter Lambda
    Lambda = np.random.rand()

    # Initialize W using PCA
    m = min(d, c - 1)
    if no_pca:
        W = np.random.rand(d, m)  # Random initialization
    else:
        pca = PCA(n_components=m)
        W = pca.fit(X).components_.T  # W should be d x m

    obj_log = []

    # Iterate until convergence or max_iter is reached
    while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:

        it += 1
        obj_old = obj_new

        # Update W
        print(f"Iteration {it}:")
        print(f"W.T shape: {W.T.shape}")  # Expecting (m, d)
        print(f"X shape: {X.shape}")      # Expecting (n, d)
        print(f"G shape: {G.shape}")      # Expecting (n, c)
        print(f"inv(G.T @ G) shape: {inv(G.T @ G).shape}")  # Expecting (c, c)
        F = W.T @ X.T  @ G @ inv(G.T @ G)
        print(f"F shape: {F.shape}")     # Expecting (m, c)

        sys.exit()

        A = (Lambda**2 - Lambda) * np.eye(d)  # d x d
        B = Lambda**2 * G @ inv(G.T @ G) @ G.T
        eigvals, eigvecs = eig(X @ (A - B) @ X.T)
        W = eigvecs[:, np.argsort(eigvals)[:m]]

        # Update Lambda
        FGt = F @ G.T
        Lambda = np.trace(W.T @ X @ X.T @ W) / (2 * np.linalg.norm(W.T @ X - FGt)**2)

        # Update G
        for i in range(n):
            distances = np.linalg.norm(W.T @ X[i, :, None] - F, axis=0)
            G[i, :] = 0
            G[i, np.argmin(distances)] = 1

        # Check for convergence
        obj_new = Lambda**2 * np.linalg.norm(W.T @ X - FGt, 'fro')**2 - Lambda * np.trace(W.T @ X @ X.T @ W)

    # Calculate embeddings T
    T = X @ W

    # Determine cluster assignments Ypre
    Ypre = np.argmax(G, axis=1).tolist()

    return T, Ypre, W
