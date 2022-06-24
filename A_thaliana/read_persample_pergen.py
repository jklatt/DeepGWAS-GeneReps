import pickle
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# file_list= os.listdir('/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/')
# present_prop=[]
# PATH='/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/'
# for file in file_list:
# 	if '0.1MAF' in file:
# 		with open(PATH+file, 'rb') as f:
# 			df=pickle.load(f)
# 			present_prop.append(df)
		
# present_prop=np.concatenate(present_prop)

# present_prop=np.concatenate(present_prop)
# print(pd.DataFrame(present_prop).describe())
# plt.hist(present_prop,bins=100)
# plt.title('minor-allele SNPs present 0.1MAF')
# plt.savefig('/home/zixshu/DeepGWAS/A_thaliana/minor-allele_present_0.1MAF.png')


file_list= os.listdir('/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/')
present_prop=[]
PATH='/home/zixshu/DeepGWAS/A_thaliana/max_present_stat/'
for file in file_list:
	if '0.1MAF' in file:
		with open(PATH+file, 'rb') as f:
			df=pickle.load(f)
			present_prop.append(df)
		
present_prop=np.concatenate(present_prop)

present_prop=np.concatenate(present_prop)
print(pd.DataFrame(present_prop).describe())
plt.hist(present_prop,bins=100)
plt.title('minor-allele SNPs present 0.1MAF')
plt.savefig('/home/zixshu/DeepGWAS/A_thaliana/minor-allele_present_0.1MAF.png')