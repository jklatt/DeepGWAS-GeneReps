from pickle import TRUE
from utils import *
import random
import os
import pandas as pd

present_df=load_file("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/selected_gene_sample_snplength20.pkl")
selected_length=20
interaction=False
gene=0 #range{0,5}
random.seed(1)
def gene_data_gen(gene:int, present_df, selected_length:int, interaction:bool):
    # for  in range(len(present_df)):
    data_list=[]
    bag_label_list=[]
    single_labels_list=[]
    bag_label_list=[]
    gene_present=present_df[gene]
    snp_identifier=list(range(selected_length))
    snp_type=np.random.randint(0,4,selected_length)
    target_mutation_pos=random.sample(range(selected_length),3)

    for sample in gene_present:
        single_labels=[]
        data=[]
        for i in range(len(sample)):
            item=[snp_identifier[i],snp_type[i],sample[i]]
            data.append(item)
            if ((sample[i]==1) & (i in target_mutation_pos)):
                label=True
            else:
                label=False    
            single_labels.append(label)
        data_list.append(data)
        single_labels_list.append(single_labels)
        if interaction:
            bag_label_check=[sample[j] for j in target_mutation_pos]
            bag_label=all(bag_label_check)
        else:
            bag_label_check=[sample[j] for j in target_mutation_pos]
            bag_label=any(bag_label_check)
        bag_label_list.append(bag_label)

    print("The prevalence of the data is",bag_label_list)

    return bag_label_list, data_list, single_labels_list

bag_label_list, data_list, single_labels_list=gene_data_gen(gene,present_df, selected_length, interaction)
print(bag_label_list)
print(data_list)
print(single_labels_list)

    # save_file=zip(data_list,single_labels_list, bag_label_list)
    # SAVE_PATH="/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/select_gene_samples"
    # os.makedirs(SAVE_PATH,exist_ok=True)
    # save_file(SAVE_PATH+"/"+"selected_snplength{}_intaraction{}_gene{}".format(selected_length,interaction,gene),save_file)
