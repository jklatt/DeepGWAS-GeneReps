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

selected_length=50#the varible of selected gene length
wanted_present_num=2029*0.2
selected_gene=[]
for i in range(len(df)):
    if len(df[list(df.keys())[i]])==selected_length:
        selected_gene.append(list(df.keys())[i])

# there are 521 genes has length 20
# random.seed(100)
# without replacement select 5 genes
# select_5_genes=random.sample(selected_gene,10)
select_5_genes=selected_gene
print("number of gene with length", selected_length, len(select_5_genes))

#selected gene names snp20 by hand
# select_5_genes=["AT1G80000", "AT2G03690", "AT1G56340", "AT5G66680", "AT4G05230"]
# print("selected gene names", pd.DataFrame(select_5_genes))

# chromo=range(1,6)

gene_by_sample_prop_list=[]
snp_index_list=[]
selected_gene_samples=[]
selected_gene_names=[]
bed_mat=np.matrix.transpose(bed_mat)
selected_snps=[1, 2, 18]
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
                # snp_index_list.append(np.array(snp_index))        
                gene_by_sample_=bed_mat[snp_index]
                gene_by_sample_=np.matrix.transpose(gene_by_sample_)
                gene_by_sample_=gene_by_sample_/2

                selected_SNP_position=gene_by_sample_[:,selected_snps]
                count_present=np.sum(selected_SNP_position, 1)
                all_present_num=np.count_nonzero(count_present==3)
                if all_present_num>wanted_present_num:
                    selected_gene_samples.append(gene_by_sample_)
                    selected_gene_names.append(gene)

                
                # print(selected_SNP_position0.sum())
                
                
                

print("Number of genes that fit the criteria", len(selected_gene_names))
random.seed(50)
select_5_genes_names=random.sample(selected_gene_names,5)
print("selected gene names", pd.DataFrame(select_5_genes_names))
selectedgene_index=[i for i, names in enumerate(selected_gene_names) if names in select_5_genes_names]
selectedsamples=[ sample for i, sample in enumerate(selected_gene_samples) if i in selectedgene_index]

# print(selected_gene_samples)
save_file("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/selected_gene_sample_snplength{}_alogPICK.pkl".format(selected_length),selectedsamples)


#seed 2 genes
# selected gene names            0
# 0  AT1G30140
# 1  AT1G61780
# 2  AT1G56210
# 3  AT4G19730
# 4  AT2G19560

#seed 1 genes
# selected gene names            0
# 0  AT1G80000
# 1  AT1G36670
# 2  AT3G28230
# 3  AT1G69760
# 4  AT5G60620

# HAND PICKS
# selected gene names            0
# 0  AT1G80000
# 1  AT5G48270
# 2  AT1G56340
# 3  AT5G08600
# 4  AT4G05230


# algo pick snp20
# selected gene names            0
# 0  AT4G02310
# 1  AT1G75080
# 2  AT2G45630
# 3  AT5G47960
# 4  AT1G64640


# algo pick snp50
# selected gene names            0
# 0  AT5G48440
# 1  AT3G52970
# 2  AT2G36570
# 3  AT4G10350
# 4  AT2G16676

# snp200 only has 5 gene so we take them to have a look
