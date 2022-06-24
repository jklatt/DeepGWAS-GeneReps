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


parser = argparse.ArgumentParser()
# parser.add_argument('--chr',type=int, required=True)
parser.add_argument('--maf',required=True)
args = parser.parse_args()

# this function is to find the per gene per sample level statistics.
chr_ = load_file('/home/zixshu/DeepGWAS/A_thaliana/chr_gen_pos_dictionary.pkl')

(_, fam, bed) = read_plink(join(get_data_folder(), "/home/zixshu/DeepGWAS/A_thaliana/X_genic/X_genic_{}.bed").format(args.maf),verbose=False) 
bim = pd.read_csv("/home/zixshu/DeepGWAS/A_thaliana/X_genic/X_genic_{}.bim".format(args.maf), sep = '\t', header = None, names = ['chrom', 'name', '-', 'pos', '/', '.'])
bed_mat=bed.compute()
print(bed_mat.shape)


gene_persample = dict()
pos = bim['pos'].values
chromosome = bim['chrom'].values
overll_minor_persample_pergene=[]


for num, chrom in enumerate(np.array(['Chr1', 'Chr2', 'Chr3', 'Chr4', 'Chr5'])):
# chrom='Chr{}'.format(args.chr)
# num=args.chr-1
    details = chr_[chrom]
    gene_list = details[1]

    for gene in gene_list:
        index = np.where(gene_list == gene)
        gene_limit = details[0][index]
        snp_index = ((pos >= gene_limit[0][0]) & (pos <= gene_limit[0][1]) & (chromosome == num+1))

        if (snp_index.sum() > 0): # just keeping genes with SNPs mapped on them
            snps_insample_list=[]
            prop_minor_list=[]

            for sample in range(fam.shape[0]):
                # print(snp_index.sum())		
                sample_arr=np.array(bed_mat[:,sample])
                snps_insample=sample_arr[np.array(snp_index, dtype=bool)]

                # snps_insample_list.append(snps_insample)
                minor_count=list(snps_insample).count(2)
                prop_minor=minor_count/len(snps_insample)
                prop_minor_list.append(prop_minor)

            overll_minor_persample_pergene.append(prop_minor_list)

                
overll_minor_persample_pergene=list(np.concatenate(overll_minor_persample_pergene))
# save_file('/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/minor_present_percentage_{}MAF_chr{}.pkl'.format(args.maf, args.chr), overll_minor_persample_pergene)
save_file('/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/minor_present_percentage_all_{}MAF.pkl'.format(args.maf), overll_minor_persample_pergene)

print(pd.DataFrame(overll_minor_persample_pergene.describe()))


