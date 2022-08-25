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

# get the present SNP number amount all SNP positions
# with open('/home/zixshu/DeepGWAS/A_thaliana/X_genic_full/X_genic_0.05.pkl', 'rb') as f:
# 	 df=pickle.load(f)

# present_prop=[]
# for i in range(len(list(df.keys()))):
# 	gene_n=df[list(df.keys())[i]]
# 	present_prop.append(gene_n.sum()/len(gene_n))

# print(pd.DataFrame(present_prop).describe())


# this function read the realted bim/fam/bed file 
## MAF 0.1
# (bim, fam, bed) = read_plink(join(get_data_folder(), "/home/zixshu/DeepGWAS/A_thaliana/X_genic/X_genic_0.1.bed"),verbose=False)                       
# print(bim.head())
# print(fam.head())
# print(bed.compute())

#MAF 0.05
# (bim, fam, bed) = read_plink(join(get_data_folder(), "/home/zixshu/DeepGWAS/A_thaliana/X_genic/X_genic_0.05.bed"),verbose=False)                     
# print(bim.head())
# print(fam.head())
# print(bed.compute())

# MAF 0.01
(bim, fam, bed) = read_plink(join(get_data_folder(), "/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_0.01.bed"),verbose=False)                  
print(bim.head())
print(fam.head())
print(bim.shape)
print(fam.shape)

bed_mat=bed.compute()
print(type(bed))
print(bed_mat.shape)
unique, counts = np.unique(np.array(bed_mat[:,2]), return_counts=True)
print(dict(zip(unique, counts)))




    









