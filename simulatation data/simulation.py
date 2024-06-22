import numpy as np
import pandas as pd
from Bio.PopGen.GenePop import read
from collections import defaultdict

# 读取Genepop文件
genepop_file = "16PopIsland_1_1.gen"
with open(genepop_file) as f:
    Island_1 = read(f)

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
data = pd.DataFrame(0, index=range(480), columns=columns)

# 填充数据
row_index = 0
for pop in Island_1.populations:
    print('pop:',pop)
    for ind in pop:
        print('ind:', ind)
        for i, allele_pair in enumerate(ind[1]):
            print('allele_pair:', allele_pair)
            locus_name = f'locus{i+1}'
            print('locus_name:', locus_name)
            for allele in allele_pair:
                print('allele:', allele)
                col_name = f'{locus_name}.{allele}'
                print('col_name:', col_name)
                if col_name in data.columns:
                    data.at[row_index, col_name] = 1
        row_index += 1

# 查看结果
print(data)
data.to_csv('data.csv')