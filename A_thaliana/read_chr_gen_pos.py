import pickle
import pandas as pd
import matplotlib.pyplot as plt
with open('/home/zixshu/DeepGWAS/A_thaliana/chr_gen_pos_dictionary.pkl', 'rb') as f:
	 df=pickle.load(f)

# print(df['Chr1'][0].shape)#(8433, 2)
# print(df['Chr2'][0].shape)#(5513, 2)
# print(df['Chr3'][0].shape)#(6730, 2)
# print(df['Chr4'][0].shape)#(5140, 2)
# print(df['Chr5'][0].shape)#(7507, 2)

# print(df['Chr1'])
# print(df['Chr1'][0][0][1]-df['Chr1'][0][0][0])

gene_length=[]
for i in range(1,6):
	chr_name='Chr{}'.format(i)
	for j in range(df[chr_name][0].shape[0]):
		gene_length.append(abs(df[chr_name][0][j][1]-df[chr_name][0][j][0]))

print(pd.DataFrame(gene_length).describe())


plt.hist(gene_length,bins=100)
plt.title('gene_length_stat')
plt.xlim(min(gene_length), max(gene_length))
plt.savefig('/home/zixshu/DeepGWAS/A_thaliana/gene_length.png')

