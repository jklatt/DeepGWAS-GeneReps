import pickle
import os
from turtle import color 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_list= os.listdir('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat_bed_reader')
present_prop=[]
PATH='/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat_bed_reader/'
for file in file_list:
	if '0.05MAF' in file:
		with open(PATH+file, 'rb') as f:
			df=pickle.load(f)
			present_prop.append(df)
		
present_prop=np.concatenate(present_prop)

present_prop=np.concatenate(present_prop)
print(pd.DataFrame(present_prop).describe())
plt.hist(present_prop,bins=100)
plt.title('minor-allele SNPs present 0.05MAF')
plt.savefig('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/minor-allele_present_0.05MAF.png')


# num_snp_vs_present_avg=[]
# for maf in [0.01, 0.05, 0.1]:
# 	for chr in [1, 2, 3, 4, 5]:
# 		with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat/number_snp_on_gene_MAF{}_chr{}.pkl'.format(maf, chr), 'rb') as f:
# 			num_snp_ongene=pickle.load(f)

# 		with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/max_present_stat_bed_reader/minor_present_percentage_{}MAF_chr{}.pkl'.format(maf,chr), 'rb') as f:
# 			minor_present=pickle.load(f)

# 		for idx, gene_allsamples in enumerate(minor_present):
# 			present_avg=np.max(gene_allsamples)
# 			num_snp_vs_present_avg+=[[num_snp_ongene[idx],present_avg, maf]]

# num_snp_vs_present_avg=pd.DataFrame(num_snp_vs_present_avg)
# num_snp_vs_present_avg.columns=['num_snp_ongene','present_avg','maf']
# num_snp_vs_present_avg.to_csv("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/num_snp_vs_present_max_bed_reader.csv")
# print(num_snp_vs_present_avg.shape)






#scatter plot
# plt.scatter(num_snp_vs_present_avg['num_snp_ongene'], num_snp_vs_present_avg['present_avg'],color=num_snp_vs_present_avg['maf'])
# plt.title('num_snp vs minor present percentatge averaging by samples')
# plt.ylabel("minor_present_percentage")
# plt.xlabel("number of SNP on gene")
# plt.savefig('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/num_snp_vs_present_max_bed_reader.png')


