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


parser = argparse.ArgumentParser()
parser.add_argument('--maf',required=True)
parser.add_argument('--chr',type=int,required=True)
# parser.add_argument('--maf',default=0.1)
# parser.add_argument('--chr',default=1)
args = parser.parse_args()

# this function is to find the per gene per sample level statistics.
chr_ = load_file('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/chr_gen_pos_dictionary.pkl')

(_, fam, bed) = read_plink(join(get_data_folder(), "/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_{}.bed").format(args.maf),verbose=False) 
bim = pd.read_csv("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_{}.bim".format(args.maf), sep = '\t', header = None, names = ['chrom', 'name', '-', 'pos', '/', '.'])
bed_mat=bed.compute()
print(bed_mat.shape)


gene_persample = dict()
pos = bim['pos'].values
chromosome = bim['chrom'].values

num=args.chr-1
chrom='Chr{}'.format(args.chr)

gene_by_sample_prop_list=[]
details = chr_[chrom]
gene_list = details[1]
num_snp_list=[]

for gene in gene_list:
    index = np.where(gene_list == gene)
    gene_limit = details[0][index]
    snp_index = ((pos >= gene_limit[0][0]) & (pos <= gene_limit[0][1]) & (chromosome == num+1))

    if (snp_index.sum() > 0):
        num_snp_list.append(snp_index.sum())


save_file('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat/number_snp_on_gene_MAF{}_chr{}.pkl'.format(args.maf, args.chr),num_snp_list)
