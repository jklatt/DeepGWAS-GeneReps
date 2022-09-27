import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/gene2snp/gene2snps_0.01.pkl', 'rb') as f:
	 df=pickle.load(f)


num_snp_pergene=[]
for i in range(len(df)):
	num_snp_pergene.append(len(df[list(df.keys())[i]]))

print(pd.DataFrame(num_snp_pergene).describe())

#ploting
# plt.hist(num_snp_pergene,bins=100,color="#00BA38")#maf0.05
plt.hist(num_snp_pergene,bins=100,color="#F8766D")#maf 0.01
# plt.hist(num_snp_pergene,bins=100,color="#619CFF")#maf 0.1
plt.title('num_snp_pergene_MAF0.01')
plt.savefig('/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/num_snp_pergene_MAF0.01.png')

