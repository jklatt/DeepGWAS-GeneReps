import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import random


gene_length=10
target_mutation_pos=[3,5]
target_mutation_val=0
num_genes=200
train=True
num_in_train=2


def generate_samples(gene_length, target_mutation_val, target_mutation_pos, num_genes,train=True):
    # generate some toy sample of GWAS data with integers
    # gene_length=np.int(np.random.normal(gene_length_mean,gene_length_var,1))

    if train==True:
        data_list=[]*num_genes
        label_list=[]
        bag_label_list=[]
        for k in range(0,num_genes):
            data=[[]]*gene_length
            label=[]
            # seed(num_genes+k)
            present_list=np.random.choice(2, gene_length)

            for i in range(0,gene_length):
                values = randint(0, 4)

                if present_list[i]==1:
                   present=1
                else: 
                   present=0


                if  i in target_mutation_pos and present==1:
                    single_label=values==target_mutation_val
                else:
                    single_label= False

                item=[i,values,present]
                item=np.expand_dims(item,axis=0)           
                label.append(single_label)
                data[i]=item
     
            if True in label:
                bag_label=[True]
                
            else:
                bag_label=[False]
                

            data_list.append(data)
            label_list.append(label)
            bag_label_list.append(bag_label)
            

    elif train==False:
        data_list=[]*num_genes
        label_list=[]
        bag_label_list=[]

        for k in range(0,num_genes):

            data=[[]]*gene_length
            label=[]
            # seed(num_genes+10000+k)
            present_list=np.random.choice(2, gene_length)

            for i in range(0,gene_length):
                values = randint(0, 4)

                if present_list[i]==1:
                    present=1
                else: 
                    present=0

                if  i in target_mutation_pos and present==1:
                    single_label=values==target_mutation_val
                else:
                    single_label= False

                item=[i,values,present] 
                item=np.expand_dims(item,axis=0)             
                label.append(single_label)
                data[i]=item

            if True in label:
                bag_label=True
               
            else:
                bag_label=False

            data_list.append(data)
            label_list.append(label)
            bag_label_list.append(bag_label)
    

    return data_list, bag_label_list, label_list

# data_list, bag_label_list, label_list=generate_samples(gene_length=10,target_mutation_val=0,target_mutation_pos=[3,5],num_genes=200)

# train_data=TensorDataset(torch.tensor(data_list),torch.tensor(bag_label_list),torch.tensor(label_list))
# train_loader =DataLoader(train_data,batch_size=num_in_train, shuffle=False)
       
                           
# for _, data in enumerate(train_loader):
#     genes=data[0]
#     print(genes)
#     labels=data[1]
#     print(labels)

def generate_samples_twoSNPs(gene_length_mean,gene_length_var, target_mutation_val1, target_mutation_pos1,target_mutation_val2, target_mutation_pos2, num_genes,train=True):
    # generate some toy sample of GWAS data with integers
    
    gene_length=np.int(np.random.normal(gene_length_mean,gene_length_var,1))
    if train==True:
        data_list=[]*num_genes
        label_list=[]
        bag_label_list=[]
        for k in range(0,num_genes):
            data=[[]]*gene_length
            label=[]
            seed(num_genes+k)

            for i in range(0,gene_length):
                values = randint(0, 4)
                if i==target_mutation_pos1:
                    single_label = values==target_mutation_val1 
                    
                elif i==target_mutation_pos2:
                    single_label = values==target_mutation_val2 

                else:
                    single_label= False

                if 3>i>0 or i>8:
                    encoding_region=1
                else: 
                    encoding_region=0

                item=[i,values,encoding_region]
                item=np.expand_dims(item,axis=0)           
                label.append(single_label)
                data[i]=item
     
            if True in label:
                bag_label=[True]
                
            else:
                bag_label=[False]
                

            data_list.append(data)
            label_list.append(label)
            bag_label_list.append(bag_label)
            

    elif train==False:
        data_list=[]*num_genes
        label_list=[]
        bag_label_list=[]

        for k in range(0,num_genes):

            data=[[]]*gene_length
            label=[]
            seed(num_genes+10000+k)

            for i in range(0,gene_length):
                values = randint(0, 4)
                if i==target_mutation_pos1:
                    single_label = values==target_mutation_val1 
                    
                elif i==target_mutation_pos2:
                    single_label = values==target_mutation_val2 

                else:
                    single_label= False

                if 3>i>0 or i>8:
                    encoding_region=1
                else: 
                    encoding_region=0

                item=[i,values,encoding_region] 
                item=np.expand_dims(item,axis=0)             
                label.append(single_label)
                data[i]=item

            if True in label:
                bag_label=True
               
            else:
                bag_label=False

            data_list.append(data)
            label_list.append(label)
            bag_label_list.append(bag_label)
    

    return data_list, bag_label_list, label_list

# data_list, bag_label_list, label_list=generate_samples_twoSNPs(gene_length_mean=13, gene_length_var=4,target_mutation_val1=0,target_mutation_pos1=3,target_mutation_val2=3,target_mutation_pos2=7,num_genes=200)

# train_data=TensorDataset(torch.tensor(data_list),torch.tensor(bag_label_list),torch.tensor(label_list))
# train_loader =DataLoader(train_data,batch_size=num_in_train, shuffle=False)
       
                           
# for _, data in enumerate(train_loader):
#     genes=data[0]
#     print(genes)
#     labels=data[1]
#     print(labels)




