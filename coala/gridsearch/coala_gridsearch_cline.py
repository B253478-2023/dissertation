from unlda import *
from unrtlda import *
from unrtlda_a import *
from untrlda import *
from unrtcdlda import *
from untrcdlda import *
from sdapc import *
from Methods.grid_search import *

def main():
    labels_insular = pd.read_csv('../datasets/labels_insular_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 1].str.replace('pop', '').astype(int).values.ravel()
    labels_cline = pd.read_csv('../datasets/labels_cline_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 1].str.replace('pop', '').astype(int).values.ravel()
    labels_weak = pd.read_csv('../datasets/labels_weak_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 1].str.replace('pop', '').astype(int).values.ravel()
    labels_strong = pd.read_csv('../datasets/labels_strong_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 1].str.replace('pop', '').astype(int).values.ravel()

    obs_labels_insular = pd.read_csv('../datasets/labels_insular_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 2].str.replace('pop', '').astype(int).values.ravel()
    obs_labels_cline = pd.read_csv('../datasets/labels_cline_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 2].str.replace('pop', '').astype(int).values.ravel()
    obs_labels_weak = pd.read_csv('../datasets/labels_weak_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 2].str.replace('pop', '').astype(int).values.ravel()
    obs_labels_strong = pd.read_csv('../datasets/labels_strong_div_0.9_rep_1.csv', skiprows=1, header=None).iloc[:, 2].str.replace('pop', '').astype(int).values.ravel()

    insulardata = pd.read_csv('../datasets/sim_insular_div_0.9_rep_1.csv', index_col=0).values
    clinedata = pd.read_csv('../datasets/sim_cline_div_0.9_rep_1.csv', index_col=0).values
    weakdata = pd.read_csv('../datasets/sim_weak_div_0.9_rep_1.csv', index_col=0).values
    strongdata = pd.read_csv('../datasets/sim_strong_div_0.9_rep_1.csv', index_col=0).values

    max_iter = 500
    k_range =range(2,5)
    Npc_range = range(2,11)

    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_rtlda')
    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_trlda')
    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='swulda')

    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_lda')
    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_kfdapc')
    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_rtalda')
    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_tralda')

    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_rtcdlda')
    #grid_search_clustering_k(clinedata, labels_cline, k_range, Npc_range,max_iter, datatype='clinedata', method='un_trcdlda')
    k_range=[1]
    Npc_range=[1]
    grid_search_clustering(clinedata, labels_cline, k_range, Npc_range, max_iter, datatype='clinedata',method='sdapc',obeserved_lables=obs_labels_cline)
    grid_search_clustering(clinedata, labels_cline, k_range, Npc_range, max_iter, datatype='clinedata', method='semi-DAPC')

    grid_search_clustering(insulardata, labels_insular, k_range, Npc_range, max_iter, datatype='insulardata',method='sdapc',obeserved_lables=obs_labels_insular)
    grid_search_clustering(insulardata, labels_insular, k_range, Npc_range, max_iter, datatype='insulardata', method='semi-DAPC')

    grid_search_clustering(strongdata, labels_strong, k_range, Npc_range, max_iter, datatype='clinedata',method='sdapc',obeserved_lables=obs_labels_strong)
    grid_search_clustering(strongdata, labels_strong, k_range, Npc_range, max_iter, datatype='clinedata', method='semi-DAPC')
    grid_search_clustering(weakdata, labels_weak, k_range, Npc_range, max_iter, datatype='clinedata',method='sdapc',obeserved_lables=obs_labels_weak)
    grid_search_clustering(weakdata, labels_weak, k_range, Npc_range, max_iter, datatype='clinedata', method='semi-DAPC')
if __name__ == "__main__":
    main()