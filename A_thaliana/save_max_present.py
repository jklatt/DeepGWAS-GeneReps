import pickle
import os
import numpy as np
import pandas as pd
from utils import *

num_snp_vs_present_avg=[]
for maf in [0.05]:
	for chr in [1, 2, 3, 4, 5]:
		with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat/number_snp_on_gene_MAF{}_chr{}.pkl'.format(maf, chr), 'rb') as f:
			num_snp_ongene=pickle.load(f)

		with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat_bed_reader/minor_present_percentage_{}MAF_chr{}.pkl'.format(maf,chr), 'rb') as f:
			minor_present=pickle.load(f)

		for idx, gene_allsamples in enumerate(minor_present):
			present_avg=np.max(gene_allsamples)
			num_snp_vs_present_avg+=[[num_snp_ongene[idx],present_avg, maf]]

num_snp_vs_present_avg=pd.DataFrame(num_snp_vs_present_avg)
num_snp_vs_present_avg.columns=['num_snp','present','maf']
print(num_snp_vs_present_avg.sort_values('num_snp').head())
print(num_snp_vs_present_avg.sort_values('num_snp').tail())
print(pd.DataFrame(num_snp_vs_present_avg.groupby('num_snp')['present'].max()))

out_file=num_snp_vs_present_avg.groupby('num_snp')['present'].max()
# out_file=1-out_file
# out_file.to_csv('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/num_snp_by_present_maf0.05_max_bed_reader.csv')

