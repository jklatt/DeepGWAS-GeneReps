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

selected_length=200#the varible of selected gene length
wanted_present_num=2029*0.2
selected_gene=[]
for i in range(len(df)):
    if len(df[list(df.keys())[i]])==selected_length:
        selected_gene.append(list(df.keys())[i])

print("number of gene with length", selected_length, len(selected_gene))
print("the name of the genes", selected_gene)
selectedsamples=[]
bed_mat=np.matrix.transpose(bed_mat)

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

for gene in selected_gene:
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
                gene_by_sample_=bed_mat[snp_index]
                gene_by_sample_=np.matrix.transpose(gene_by_sample_)
                gene_by_sample_=gene_by_sample_/2
                selectedsamples.append(gene_by_sample_)

                #select the most present SNP
                count_present=np.sum(gene_by_sample_, 0)
                top_1_snp=sort_index(count_present)[:1]
                print("the most present snp is", top_1_snp[0])

                #get the sample with top1 SNP are present
                top_1_present_ind=[i for i, sample in enumerate(gene_by_sample_) if sample[top_1_snp]==1]

                #within already top1 present sample count the next most present sample. 
                top_1_present_samples=gene_by_sample_[top_1_present_ind]
                count_present_1=np.sum(top_1_present_samples, 0)
                top_1_snp_second=sort_index(count_present_1)[1:2]
                print("the next most present snp is", top_1_snp_second[0])

                # within top1 top2 present sample chose the most present sample.
                top_2_present_ind=[i for i, sample in enumerate(top_1_present_samples) if sample[top_1_snp_second]==1]
                top_2_present_samples=top_1_present_samples[top_2_present_ind]
                count_present_2=np.sum(top_2_present_samples, 0)
                top_2_snp=sort_index(count_present_2)[:3]
                print("the next next most present snp is", top_2_snp)


# save_file("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/selected_gene_sample_snplength{}_alogPICK.pkl".format(selected_length),selectedsamples)














