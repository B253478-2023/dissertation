import numpy as np
import scipy
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def coordinate_descent_clustering(A, num_clusters, atol=1e-6, max_iter=100):
    """
    Perform coordinate descent clustering.

    Parameters:
    A: np.ndarray
        The input squared matrix of shape (n, n).
    num_clusters: int
        The number of clusters (c).
    atol: float, optional
        Absolute tolerance for convergence. Default is 1e-6.
    max_iter: int, optional
        Maximum number of iterations. Default is 100.

    Returns:
    np.ndarray
        The optimal cluster indicator matrix G.
    """

    n = A.shape[0]
    c = num_clusters

    # Randomly initialize the cluster indicator matrix G
    G = np.zeros((n, c))
    for i in range(n):
        G[i, np.random.randint(0, c)] = 1

    # Computing g_l^T A g_l  and g_l^T g_l for all l = 1, 2, ..., c
    g_A_g = [G[:, l].T @ A @ G[:, l] for l in range(c)]
    g_g = [G[:, l].T @ G[:, l] for l in range(c)]

    for iter_num in range(max_iter):
        G_old = G.copy()

        for i in range(n):
            m = np.argmax(G[i, :])  # Record the cluster index m

            if sum(G[:, m]) == 1:
                continue  # If the ith row is the only 1 in the m-th column, continue

            # Generate G^(0)
            G_0 = G.copy()
            G_0[i, :] = 0

            D = np.zeros(c)
            for k in range(c):
                G_k = G_0.copy()
                G_k[i, k] = 1

                if k == m:
                    g_k_A_g_k = g_A_g[m]
                    g_k_g_k_0 = g_g[m] - 1
                else:
                    g_k_A_g_k = g_A_g[k] + 2 * (G[:, k].T @ A[i, :] - A[i, i])
                    g_k_g_k_0 = g_g[k] + 1

                D[k] = (g_k_A_g_k / g_k_g_k_0) - (g_A_g[m] / (g_g[m] - 1))

            best_k = np.argmax(D)

            if best_k != m:
                G[i, :] = 0
                G[i, best_k] = 1
                g_g[best_k] += 1
                g_g[m] -= 1
                g_A_g[best_k] += 2 * (G[:, best_k].T @ A[i, :] - A[i, i])
                g_A_g[m] -= 2 * (G[:, m].T @ A[i, :] - A[i, i])

        if np.linalg.norm(G - G_old) < atol:
            break

    return G


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
def un_rt_cd_lda(X, c, Ninit=10, gamma=1e-6, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False, cd_clustering=True):
    """
    Implement the Unsupervised Ratio-Trade (Coordinate Descent) Linear Discriminant Analysis (Un-RT(CD)LDA) algorithm for clustering.

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
        cd_clustering (bool, optional): Whether to enable CD clustering. Defaults to True.

    Returns:
        T (numpy array): Un-RTLDA embeddings of shape (n_samples, n_components).
        Ypre (list): Cluster assignments for each sample.
        W2 (numpy array): Eigenvectors matrix of shape (n_features, n_components).
    """
    n, d = X.shape  # Number of samples

    if center:
        X = X - np.mean(X, axis=0)  # Center the data

    St = X.T @ X  # Compute the within-class scatter matrix St
    Stt = St + gamma * np.eye(d)  # Add regularization term to St

    it = 0  # Initialize the iteration counter

    obj_old = -np.inf  # Initialize the old objective value
    obj_new = 0.0  # Initialize the new objective value
    Ypre = None  # Initialize the predicted cluster labels
    T_old = None
    W2_old = None
    T = None

    # Initialize W using PCA
    m = min(d, c - 1)
    pca = PCA(n_components=m)

    if no_pca:
        W = X.T[:, :m]
    else:
        W = pca.fit_transform(X.T)
    W2 = W  # Initialize W2 with W
    #print(f'W2 shape: {W2.shape}')
    obj_log = []

    # Iterate until convergence or maxIter is reached
    while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:

        it += 1
        obj_old = obj_new

        # Calculate the intermediate matrix product
        T = (scipy.linalg.expm(-0.5 * np.linalg.inv(W2.T @ Stt @ W2)) @ W2.T @ X.T).T
        # T = (fractional_matrix_power(W2.T @ Stt @ W2, -0.5) @ W2.T @ X.T @ H).T
        #print(f'T shape: {T.shape}')
        best_obj_tmp = float('inf')
        best_Ypre = None

        if cd_clustering:

            intermediate = np.linalg.inv(W2.T @ X.T @ X @ W2)
            #print(f"intermediate shape: {intermediate.shape}")  # Expecting (m, m)
            AA = X @ W2 @ intermediate @ W2.T @ X.T
            #print(f"AA shape: {AA.shape}") # Expecting (n, n)
            GG = coordinate_descent_clustering(AA, atol=tol, num_clusters=c, max_iter=max_iter)
            Ypre = np.argmax(GG, axis=1)
        else:
            # Loop through Ntry times to find the best clustering
            for j in range(Ntry):
                kmeans = KMeans(n_clusters=c, tol=tol, max_iter=max_iter, n_init=Ninit)  # Initialize KMeans clustering
                Ypre_temp = kmeans.fit_predict(T)  # Cluster the data and obtain labels
                obj_tmp = kmeans.inertia_  # Store the within-cluster sum of squares
                # Update Ypre if the new clustering is better than the previous one
                if obj_tmp < best_obj_tmp:
                    best_obj_tmp = obj_tmp
                    best_Ypre = Ypre_temp
            Ypre = best_Ypre

        # Update Yp matrix
        Yp = np.eye(c)[Ypre]
        #print(f"Yp shape: {Yp.shape}")  # Expecting (n, c)
        #print(Yp)
        # Compute the between-class scatter matrix Sb
        Sb = X.T @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ X

        #print(f"Sb shape: {Sb.shape}")  # Expecting (d, d)

        # Perform generalized eigenvalue decomposition and update W2
        model = gevd(Sb, Stt)
        W2 = model['W'][:, -m:]
        # Print shapes
        #print(f"W2 shape: {W2.shape}")  # Expecting (d, min(d, c-1))

        # Update the new objective value
        #obj_new = np.trace((W2.T @ Stt @ W2) ** -1 @ W2.T @ Sb @ W2)
        pinv_term = np.linalg.pinv(W2.T @ Stt @ W2)
        obj_new = np.trace(pinv_term @ W2.T @ Sb @ W2)

        obj_log.append(obj_new)

    # Print a warning if the algorithm did not converge within maxIter iterations
    if it == max_iter:
        print(f"Warning: The un_rt(cd)lda did not converge within {max_iter} iterations!")

    return T, Ypre, W2, obj_log