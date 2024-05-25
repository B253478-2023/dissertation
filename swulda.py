import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.linalg import eigh


def swulda(X, c, Ninit=10, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False):
    """
    Implement the Self-Weighted Unsupervised Linear Discriminant Analysis (SWULDA) algorithm for clustering.

    Args:
        X (numpy array): Input data of shape (n_samples, n_features).
        c (int): Number of clusters.
        Ninit (int, optional): Number of initializations for KMeans. Defaults to 10.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        Ntry (int, optional): Number of attempts to find the best clustering. Defaults to 10.
        center (bool, optional): Whether to center the data. Defaults to True.
        no_pca (bool, optional): Whether to disable PCA initialization. Defaults to False.

    Returns:
        G (numpy array): Indicator matrix of shape (n_samples, n_clusters).
        W (numpy array): Projection matrix of shape (n_features, n_clusters).
        Lambda (float): Adaptive weight.
        obj_log (list): Log of objective values during iterations.
    """
    n, d = X.shape  # Number of samples

    if center:
        H = np.eye(n) - np.ones((n, n)) / n
        X_centered = H @ X
    else:
        X_centered = X

    St = X_centered.T @ X_centered  # Compute the total scatter matrix St

    # Initialize random indicator matrix G
    G = np.zeros((n, c))
    rand_indices = np.random.choice(c, n)
    G[np.arange(n), rand_indices] = 1

    # Initialize random weight parameter Lambda
    Lambda = np.random.rand()

    # Initialize W using PCA
    m = min(d, c - 1)
    if no_pca:
        W = X_centered.T[:, :m]
    else:
        pca = PCA(n_components=m)
        W = pca.fit_transform(X_centered.T @ H).T

    obj_log = []

    # Iterate until convergence or max_iter is reached
    for it in range(max_iter):
        # Update W by solving the eigenvalue problem
        A = (Lambda ** 2 - Lambda) * np.eye(d)
        B = Lambda ** 2 * G @ np.linalg.inv(G.T @ G) @ G.T
        M = X_centered @ (A - B) @ X_centered.T
        eigenvalues, W = eigh(M, eigvals=(0, c - 1))

        # Update Lambda
        F = W.T @ X_centered @ G @ np.linalg.inv(G.T @ G)
        Lambda_new = np.trace(W.T @ X_centered @ X_centered.T @ W) / (
                    2 * np.linalg.norm(W.T @ X_centered - F @ G.T, 'fro') ** 2)

        # Update G
        D = np.zeros_like(G)
        for i in range(n):
            distances = np.linalg.norm(W.T @ X_centered[i, :].reshape(-1, 1) - F, axis=0)
            min_index = np.argmin(distances)
            D[i, min_index] = 1

        # Check convergence
        if np.abs(Lambda_new - Lambda) < tol:
            break

        Lambda = Lambda_new
        G = D

        # Update objective function log
        obj_new = np.trace(W.T @ St @ W) / np.trace(W.T @ X_centered @ (A - B) @ X_centered.T @ W)
        obj_log.append(obj_new)

    return G, W, Lambda, obj_log


# Example usage with random data
np.random.seed(0)
X = np.random.rand(100, 10)
num_clusters = 3
G, W, Lambda, obj_log = swulda(X, num_clusters, center=True)

print("Indicator Matrix G:")
print(G)
print("Projection Matrix W:")
print(W)
print("Adaptive Weight Lambda:")
print(Lambda)
print("Objective Log:")
print(obj_log)