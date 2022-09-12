import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random

with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/gene2snp/gene2snps_0.05.pkl', 'rb') as f:
	 df=pickle.load(f)

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






