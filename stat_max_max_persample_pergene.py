import pandas as pd

df=pd.read_csv("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/num_snp_by_present_maf0.05_max_bed_reader.csv")
print(df['present'].describe())