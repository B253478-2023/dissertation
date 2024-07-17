import sys
import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import PCA
from scipy.linalg import eigh


def swulda(X, c, Npc, tol=1e-6, max_iter=100, center=True, no_pca=False):
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
    #m = min(d, c - 1, Npc)
    m = Npc
    if no_pca:
        W = np.random.rand(d, m)  # Random initialization
    else:
        pca = PCA(n_components=m)
        W = pca.fit(X).components_.T  # W should be d x m
    
    #print(f"X shape: {X.shape}")      # Expecting (n, d)
    #print(f"n: {n}")
    #print(f"d: {d}")
    #print(f"m: {m}")
    #print(f"c: {c}")
    #print(f"W shape: {W.shape}")  # Expecting (m, d))
    #print(f"G shape: {G.shape}")  # Expecting (n, c))

    obj_log = []

    # Iterate until convergence or max_iter is reached
    while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:
        it += 1
        obj_old = obj_new

        # Print shapes before updating W
        #print(f"Iteration {it}:")
        #print(f"W.T shape: {W.T.shape}")  # Expecting (m, d)
        #print(f"G shape: {G.shape}")      # Expecting (n, c)
        #print(f"inv(G.T @ G) shape: {np.linalg.pinv(G.T @ G).shape}")  # Expecting (c, c)

        # Update W 
        A = (Lambda**2 - Lambda) * np.eye(n)
        #print(f"A shape: {A.shape}")  # Expecting (n, n)

        B = Lambda**2 * G @ np.linalg.pinv(G.T @ G) @ G.T
        #print(f"B shape: {B.shape}")  # Expecting (n, n)

        M = X.T @ (A - B) @ X
        #print(f"M shape: {M.shape}")  # Expecting (d, d)

        eigvals, eigvecs = np.linalg.eigh(M)
        W = eigvecs[:, :m]
        #print(f"W shape: {W.shape}")  # Expecting (d, m)

        # Update F
        # F = W^T X G (G^T G)^-1
        F = W.T @ X.T @ G @ np.linalg.pinv(G.T @ G)
        #print(f"F shape: {F.shape}")     # Expecting (m, c)

        # Update Lambda
        FGt = F @ G.T  # Dimension: (m, c) x (c, n) = (m, n)
        #print(f"FGt shape: {FGt.shape}")  # Expecting (m, n)

        # Update G
        for i in range(n):
            distances = np.linalg.norm(W.T @ X[i, :].reshape(-1, 1) - F, axis=0)
            G[i, :] = 0
            G[i, np.argmin(distances)] = 1

        Lambda = np.trace(W.T @ X.T @ X @ W) / (2 * np.linalg.norm(W.T @ X.T - FGt)**2)
        #print(f"Lambda: {Lambda}")

        # Check for convergence
        obj_new = Lambda**2 * np.linalg.norm(W.T @ X.T - FGt, 'fro')**2 - Lambda * np.trace(W.T @ X.T @ X @ W)
        obj_log.append(obj_new)

    # Calculate embeddings T
    T = X @ W
    #print(f"T shape: {T.shape}") # Expecting n,m

    # Determine cluster assignments Ypre
    Ypre = np.argmax(G, axis=1).tolist()

    return T, Ypre, W, obj_log