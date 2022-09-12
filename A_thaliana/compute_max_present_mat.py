from lib2to3.pgen2.pgen import DFAState
from pandas_plink import read_plink
from pandas_plink import get_data_folder
from os.path import join
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time
import argparse
from joblib import Parallel, delayed
from bed_reader import open_bed


parser = argparse.ArgumentParser()
# parser.add_argument('--maf',required=True)
# parser.add_argument('--chr',type=int,required=True)
parser.add_argument('--maf',default=0.1)
parser.add_argument('--chr',type=int,default=1)
args = parser.parse_args()


file_name = "/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_{}.bed".format(args.maf)
bed = open_bed(file_name)
bed_mat = bed.read()


# this function is to find the per gene per sample level statistics.
chr_ = load_file('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/chr_gen_pos_dictionary.pkl')
# (_, fam,_) = read_plink(join(get_data_folder(), "/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_{}.bed").format(args.maf),verbose=False) 
# bed_mat=bed.compute()

bim = pd.read_csv("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_{}.bim".format(args.maf), sep = '\t', header = None, names = ['chrom', 'name', '-', 'pos', '/', '.'])
print(bed_mat.shape)


gene_persample = dict()
pos = bim['pos'].values
chromosome = bim['chrom'].values

# for num, chrom in enumerate(np.array(['Chr1', 'Chr2', 'Chr3', 'Chr4', 'Chr5'])):

# start = time.time()
num=args.chr-1
chrom='Chr{}'.format(args.chr)

gene_by_sample_prop_list=[]
details = chr_[chrom]
gene_list = details[1]

for gene in gene_list:
    index = np.where(gene_list == gene)
    gene_limit = details[0][index]
    snp_index = ((pos >= gene_limit[0][0]) & (pos <= gene_limit[0][1]) & (chromosome == num+1))

    if (snp_index.sum() > 0): # just keeping genes with SNPs mapped on them
        gene_by_sample_times2=np.matmul(bed_mat, np.array(snp_index))
        gene_by_sample=gene_by_sample_times2/2
        gene_by_sample_prop=gene_by_sample/snp_index.sum()
        gene_by_sample_prop_list.append(gene_by_sample_prop)




# print(list(np.concatenate(gene_by_sample_prop_list)))
# end = time.time()
# print(end - start)      
# overll_minor_persample_pergene=list(np.concatenate(overll_minor_persample_pergene))

#outputting the file
# save_file('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat_bed_reader/minor_present_percentage_{}MAF_chr{}.pkl'.format(args.maf, args.chr), gene_by_sample_prop_list)




# save_file('/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/minor_present_percentage_all_{}MAF.pkl'.format(args.maf), overll_minor_persample_pergene)
# print(pd.DataFrame(overll_minor_persample_pergene.describe()))


