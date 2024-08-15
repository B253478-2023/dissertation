import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def trace_ratio(A, B, dim, is_max=True):
    """
    Solve the Trace Ratio problem: max_{W'*W=I} tr(W'*A*W)/tr(W'*B*W)

    Args:
        A (numpy array): Symmetric matrix A.
        B (numpy array): Positive semi-definite matrix B.
        dim (int): Number of components to retain.
        is_max (bool, optional): Whether to maximize the trace ratio. Defaults to True.

    Returns:
        W (numpy array): Projection matrix of shape (n_features, dim).
        obj (float): Objective value of the trace ratio.
    """
    n = A.shape[0]
    W = np.eye(n, dim)
    ob = np.trace(W.T @ A @ W) / np.trace(W.T @ B @ W)

    counter = 1
    obd = 1
    obj = []

    while obd > 1e-6 and counter < 20:
        M = A - ob * B
        M = np.maximum(M, M.T)
        M = (M + M.T) / 2  # Ensure M is symmetric
        eigvals, eigvecs = np.linalg.eigh(M)

        if is_max:
            idx = np.argsort(eigvals)[::-1]
        else:
            idx = np.argsort(eigvals)

        W = eigvecs[:, idx[:dim]]
        obd = np.abs(np.sum(eigvals[idx[:dim]]))
        ob = np.trace(W.T @ A @ W) / np.abs(np.trace(W.T @ B @ W))
        obj.append(ob)
        counter += 1

    if counter == 20:
        print('Warning: the trace ratio did not converge!')

    return W, obj[-1]

def un_trlda(X, c, Npc, Ninit=10, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False):
    # describetion need to be changed
    """
    Implement the Unsupervised Trace-Ratio Linear Discriminant Analysis (Un-TRLDA) algorithm for clustering.

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
        T (numpy array): Un-TRLDA embeddings of shape (n_samples, n_components).
        Ypre (list): Cluster assignments for each sample.
        W2 (numpy array): Eigenvectors matrix of shape (n_features, n_components).
    """
    n, d = X.shape  # Number of samples

    if center:
        H = np.eye(n) - np.ones((n, n)) / n
    else:
        H = np.eye(n)

    St = X.T @ H @ X  # Compute the within-class scatter matrix St

    it = 0  # Initialize the iteration counter

    obj_old = -np.inf  # Initialize the old objective value
    obj_new = 0.0  # Initialize the new objective value
    Ypre = None  # Initialize the predicted cluster labels
    T_old = None
    W2_old = None
    T = None

    # Initialize W using PCA
    #m = min(d, c - 1,Npc)
    m = Npc
    pca = PCA(n_components=Npc)

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
        T = (W2.T @ X.T @ H).T
        # T = (fractional_matrix_power(W2.T @ Stt @ W2, -0.5) @ W2.T @ X.T @ H).T

        best_obj_tmp = float('inf')
        best_Ypre = None

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

        # Compute the between-class scatter matrix Sb
        Sb = X.T @ H @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ H.T @ X

        # Perform Trace Ratio optimization and update W2
        W2, _ = trace_ratio(Sb, St, Npc)

        obj_new = np.trace(W2.T @ Sb @ W2) / np.trace(W2.T @ St @ W2)

        obj_log.append(obj_new)

    # Print a warning if the algorithm did not converge within maxIter iterations
    if it == max_iter:
        print(f"Warning: The un_trlda did not converge within {max_iter} iterations!")

    return T, Ypre, W2, obj_log