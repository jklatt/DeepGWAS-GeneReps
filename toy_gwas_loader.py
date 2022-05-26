import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

from main_toygwas_jk import train_test_split


gene_length=10
target_mutation_pos=[3,5]
target_mutation_val=0
num_genes=200
train=True
num_in_train=2

def generate_samples(gene_length, max_present,num_casual_snp, num_genes_train,num_genes_test, interaction=False):
    # generate some toy sample of GWAS data with integers
    # gene_length=np.int(np.random.normal(gene_length_mean,gene_length_var,1))
    num_genes=num_genes_train+num_genes_test

    target_mutation_pos=random.sample(range(gene_length),num_casual_snp)
    
    data_list=[]
    label_list=[]
    bag_label_list=[]
    num_casual_snp_list=random.choices(range(1,max_present),k=num_genes)
    
    for k in range(0,num_genes):
        data=[[]]*gene_length
        label=[]
        
        present_snp=random.sample(range(gene_length), num_casual_snp_list[k])
        present_list=np.zeros(gene_length)
        for l in present_snp:
            present_list[l]=1

        for i in range(0,gene_length):
            values = randint(0, 3)
            present=present_list[i]

            if  i in target_mutation_pos and present==1:
                # single_label=values==target_mutation_val
                single_label=True
            else:
                single_label= False

            item=[[i,values,present]]
            # item=[[values,present]]


            # item=np.expand_dims(item,axis=0)           
            label.append(single_label)
            data[i]=item

        if interaction:
            casual_label=[label[j] for j in target_mutation_pos]

            if all(casual_label):
                bag_label=[True]     
            else:
                bag_label=[False]

        else:
            if True in label:
                bag_label=[True]
                
            else:
                bag_label=[False]
            

        data_list.append(data)
        label_list.append(label)
        bag_label_list.append(bag_label)

    train_data_list=data_list[:num_genes_train]
    test_data_list=data_list[num_genes_train:]

    train_label_list= label_list[:num_genes_train]
    test_label_list= label_list[num_genes_train:]

    train_bag_label_list= bag_label_list[:num_genes_train]
    test_bag_label_list = bag_label_list[num_genes_train:]


    
    # TODO:
    # if true_prob<0.2: 
    #     ind=[i for i, x in enumerate(bag_label_list) if x]
    #     ind_withreplace=random.choice(ind, size=num_genes*0.2)

    return train_data_list, train_bag_label_list, train_label_list, test_data_list, test_bag_label_list,test_label_list, 

# train_data_list, train_bag_label_list, train_label_list, test_data_list, test_label_list, test_bag_label_list=generate_samples(gene_length=10,
# max_present=8,num_casual_snp=2,num_genes_train=200,num_genes_test=200,interaction=True)

# train_data=TensorDataset(torch.tensor(data_list),torch.tensor(bag_label_list),torch.tensor(label_list))
# train_loader =DataLoader(train_data,batch_size=num_in_train, shuffle=False)
       
                           
# for _, data in enumerate(train_loader):
#     genes=data[0]
#     print(genes)
#     labels=data[1]
#     print(labels)

def get_weight(bag_label_list):

    true_prob=np.array(bag_label_list).sum()/len(bag_label_list)
    false_prob=1-true_prob
    bag_class_weight=[1/true_prob,1/false_prob]

    return bag_class_weight



