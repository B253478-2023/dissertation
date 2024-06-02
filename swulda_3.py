import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import PCA
from scipy.linalg import eigh

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

def swulda(X, c, tol=1e-6, max_iter=100, center=True, no_pca=False):
    n, d = X.shape  # Number of samples and features

    if center:
        X = X - np.mean(X, axis=0)  # Center the data

    it = 0  # Initialize the iteration counter

    obj_old = -np.inf  # Initialize the old objective value
    obj_new = 0.0  # Initialize the new objective value
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

        # Print shapes before updating W
        print(f"Iteration {it}:")
        print(f"W.T shape: {W.T.shape}")  # Expecting (m, d)
        print(f"X shape: {X.shape}")      # Expecting (n, d)
        print(f"G shape: {G.shape}")      # Expecting (n, c)
        print(f"inv(G.T @ G) shape: {inv(G.T @ G).shape}")  # Expecting (c, c)

        # Update F
        F = W.T @ X.T @ G @ inv(G.T @ G)
        print(f"F shape: {F.shape}")     # Expecting (m, c)

        # Compute scatter matrices
        H = np.eye(n) - (1/n) * np.ones((n, n))  # Centering matrix
        print(f"H shape: {H.shape}")      # Expecting (n, n)

        Sw = X.T @ H @ X  # Within-class scatter matrix
        # Sw = (Sw + Sw.T) / 2  # Ensuring symmetry
        print(f"Sw shape: {Sw.shape}")    # Expecting (d, d)

        Sb = X.T @ H @ G @ inv(G.T @ G) @ G.T @ H @ X  # Between-class scatter matrix
        # Sb = (Sb + Sb.T) / 2  # Ensuring symmetry
        print(f"Sb shape: {Sb.shape}")    # Expecting (d, d)

        # Update W using generalized eigenvalue decomposition
        model = gevd(Sb, Sw)
        W = model['W'][:, -m:]  # Select top m eigenvectors
        print(f"W shape: {W.shape}")      # Expecting (d, m)

        # Update Lambda
        FGt = F @ G.T  # Dimension: (m, c) x (c, n) = (m, n)
        print(f"FGt shape: {FGt.shape}")  # Expecting (m, n)

        Lambda = np.trace(W.T @ X.T @ X @ W) / (2 * np.linalg.norm(W.T @ X.T - FGt)**2)
        print(f"Lambda: {Lambda}")

        # Check for convergence
        obj_new = Lambda**2 * np.linalg.norm(W.T @ X.T - FGt, 'fro')**2 - Lambda * np.trace(W.T @ X.T @ X @ W)
        obj_log.append(obj_new)

    # Calculate embeddings T
    T = X @ W

    # Determine cluster assignments Ypre
    Ypre = np.argmax(G, axis=1).tolist()

    return T, Ypre, W, obj_log
