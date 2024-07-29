import numpy as np
import pandas as pd
import sys
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

def initialize_r_environment():
    # Activate the automatic conversion of pandas DataFrame to R data frame
    pandas2ri.activate()
    
    # Import necessary R packages
    adegenet = importr('adegenet')
    base = importr('base')
    utils = importr('utils')
    stats = importr('stats')
    
    # Define the DAPC cross-validation function in R
    ro.r('''
        dapc_xval <- function(dat, grp, prefix, n_pca_min, n_pca_max, n_pca_interval){
            mat <- tab(dat, NA.method="mean")
            n_pca_seq <- seq(n_pca_min, n_pca_max, by=n_pca_interval)
            cx <- xvalDapc(mat, grp, n.pca.max = n_pca_max, training.set = 0.8,
                           result = "groupMean", center = TRUE, scale = TRUE,
                           n.pca = n_pca_seq, n.rep = 30, xval.plot = FALSE)
            
            pc_retain <- as.numeric(as.vector(names(cx[5][[1]])))[which.min(unlist(cx[5]))]
            
            dapc1 <- dapc(dat, grp, n.pca=pc_retain, n.da=length(unique(grp)))
            
            eig <- dapc1$eig
            pov <- eig / sum(eig) * 100
            
            return(list(dapc=dapc1, pov=pov))
        }
    ''')
    return adegenet

def sdapc(X, labels=None, prop_pc_var=0.5, max_n_clust=20, n_pca_min=20, n_pca_max=300, n_pca_interval=20):
    embeddings = {}
    
    # Initialize the R environment
    adegenet = initialize_r_environment()
    
    # Convert the numpy array to an R matrix
    X_df = pd.DataFrame(X)
    r_X = pandas2ri.py2rpy(X_df)
    
    # Convert data to genind object
    genind_obj = adegenet.df2genind(r_X, ploidy=1)
    genind_obj2 = adegenet.df2genind(r_X, ploidy=1)
    
    # Semi-supervised DAPC (using find.clusters)
    ro.globalenv['genind_obj'] = genind_obj
    ro.globalenv['genind_obj2'] = genind_obj2
    ro.globalenv['prop_pc_var'] = prop_pc_var
    ro.globalenv['max_n_clust'] = max_n_clust
    ro.globalenv['n_pca_min'] = n_pca_min
    ro.globalenv['n_pca_max'] = n_pca_max
    ro.globalenv['n_pca_interval'] = n_pca_interval
    ro.r('grp <- find.clusters(genind_obj, perc.pca=prop_pc_var, pca.select="percVar", criterion="diffNgroup", max.n.clust=max_n_clust, choose.n.clust=FALSE)')
    grp = ro.r('grp$grp')
    ro.globalenv['grp'] = grp
    semi_dapc_result = ro.r('dapc_xval(genind_obj, grp, "semisupervised", n_pca_min, n_pca_max, n_pca_interval)')
    
    semi_dapc = semi_dapc_result.rx2('dapc')
    semi_pov = semi_dapc_result.rx2('pov')
    print(semi_pov)
    
    T_semi = np.array(semi_dapc.rx2('tab'))
    G_semi = np.array(semi_dapc.rx2('grp'))
    W_semi = None

    embeddings["Semisupervised-DAPC"] = {"T": T_semi.T, "W": W_semi, "G": G_semi}
    
    # Supervised DAPC (using provided labels)
    if labels is not None:
        str_labels = [str(label) for label in labels]
        ro.globalenv['labels'] = ro.FactorVector(StrVector(str_labels))
        ro.r('genind_obj2@pop <- labels')
        supervised_dapc_result = ro.r('dapc_xval(genind_obj2, labels, "supervised", n_pca_min, n_pca_max, n_pca_interval)')
        
        supervised_dapc = supervised_dapc_result.rx2('dapc')
        supervised_pov = supervised_dapc_result.rx2('pov')
        print(supervised_pov)
        
        T_sup = np.array(supervised_dapc.rx2('tab'))
        G_sup = np.array(supervised_dapc.rx2('grp'))
        W_sup = None
        
        embeddings["Supervised-DAPC"] = {"T": T_sup.T, "W": W_sup, "G": G_sup}
    
    return embeddings
