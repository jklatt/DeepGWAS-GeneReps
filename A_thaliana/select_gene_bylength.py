import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from bed_reader import open_bed
from utils import *

with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/gene2snp/gene2snps_0.05.pkl', 'rb') as f:
	 df=pickle.load(f)

file_name = "/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_0.05.bed"
bed = open_bed(file_name)
bed_mat = bed.read()
chr_ = load_file('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/chr_gen_pos_dictionary.pkl')
bim = pd.read_csv("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_0.05.bim", sep = '\t', header = None, names = ['chrom', 'name', '-', 'pos', '/', '.'])
gene_persample = dict()
pos = bim['pos'].values
chromosome = bim['chrom'].values

selected_length=20#the varible of selected gene length
selected_gene=[]
for i in range(len(df)):
    if len(df[list(df.keys())[i]])==selected_length:
        selected_gene.append(list(df.keys())[i])

#there are 521 genes has length 20
random.seed(1)
#without replacement select 5 genes
select_5_genes=random.sample(selected_gene,5)
print("selected gene names", pd.DataFrame(select_5_genes))

# chromo=range(1,6)

gene_by_sample_prop_list=[]
snp_index_list=[]
selected_gene_samples=[]
bed_mat=np.matrix.transpose(bed_mat)
for gene in select_5_genes:
    for i in range(1,6):
        chromo=i
        num=chromo-1
        chrom='Chr{}'.format(chromo)
        details = chr_[chrom]
        gene_list = details[1]
        if gene in gene_list:
            index = np.where(gene_list == gene)
            gene_limit = details[0][index]
            snp_index = ((pos >= gene_limit[0][0]) & (pos <= gene_limit[0][1]) & (chromosome == num+1))

            if (snp_index.sum() > 0):
                snp_index_list.append(np.array(snp_index))        
                gene_by_sample_=bed_mat[snp_index]
                gene_by_sample_=np.matrix.transpose(gene_by_sample_)
                gene_by_sample_=gene_by_sample_/2
                selected_gene_samples.append(gene_by_sample_)
print(selected_gene_samples)
save_file("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/selected_gene_sample_snplength{}.pkl".format(selected_length),selected_gene_samples)





