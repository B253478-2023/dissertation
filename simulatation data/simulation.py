import numpy as np
import pandas as pd
from Bio.PopGen.GenePop import read
from collections import defaultdict

def read_gene(genepop_file):
    # 读取Genepop文件
    with open(genepop_file) as f:
        Island_1 = read(f)

    # 计算行数（即个体总数）
    num_individuals = sum(len(pop) for pop in Island_1.populations)

    # 提取所有的locus和alleles
    loci_alleles = defaultdict(set)
    for pop in Island_1.populations:
        #print('pop:', pop)
        for ind in pop:
            #print('ind:', ind)
            for i, allele_pair in enumerate(ind[1]):
                #print('i:', i)
                #print('allele_pair:', allele_pair)
                locus_name = f'locus{i+1}'
                loci_alleles[locus_name].update(allele_pair)

    # 创建列名
    columns = []
    for locus, alleles in loci_alleles.items():
        for allele in sorted(alleles):
            columns.append(f'{locus}.{allele}')

    # 初始化 DataFrame
    data = pd.DataFrame(0, index=range(num_individuals), columns=columns)

    # 初始化种群数组
    pop_array = np.zeros(num_individuals, dtype=int)
    pop_index = 0
    # 填充数据
    row_index = 0
    for pop in Island_1.populations:
        #print('pop:',pop)
        for ind in pop:
            #print('ind:', ind)
            pop_array[row_index] = pop_index
            for i, allele_pair in enumerate(ind[1]):
                #print('allele_pair:', allele_pair)
                locus_name = f'locus{i+1}'
                #print('locus_name:', locus_name)
                for allele in allele_pair:
                    #print('allele:', allele)
                    col_name = f'{locus_name}.{allele}'
                    #print('col_name:', col_name)
                    if col_name in data.columns:
                        data.at[row_index, col_name] = 1
            row_index += 1
        pop_index += 1

    data_array = data.values

    return data_array,pop_array

def main():
    Islanddata,pop_array = read_gene("16PopIsland_1_1.gen")
    HierIslanddata,_ = read_gene("16PopHierISland_1_1.gen")
    Steppingstonedata,_ = read_gene("16PopSteppingstone_1_1.gen")
    Hiersteppingstonedata,_ = read_gene("16PopHiersteppingstone_1_1.gen")



if __name__ == "__main__":
    main()