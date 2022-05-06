from tkinter import LabelFrame
from matplotlib.cbook import flatten
import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader

gene_length=10
target_mutation=3
num_genes=200
train=True
num_in_train=2


def generate_samples(gene_length, target_mutation,num_genes,train=True):
    # generate some toy sample of GWAS data with integers
    
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
                single_label=values==target_mutation
                
                if 3>i>0 or i>8:
                    encoding_region=1
                else: 
                    encoding_region=0

                item=[i,values,encoding_region]
                # item=np.expand_dims(item,axis=0) 
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
                single_label=values==target_mutation

                if 3>i>0 or i>8:
                    encoding_region=1
                else: 
                    encoding_region=0

                item=[i,values,encoding_region] 
                # item=np.expand_dims(item,axis=0)
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

# data_list, bag_label_list, label_list=generate_samples(gene_length=10, target_mutation=3,num_genes=200)

# train_data=TensorDataset(torch.tensor(data_list),torch.tensor(bag_label_list),torch.tensor(label_list))
# train_loader =DataLoader(train_data,batch_size=num_in_train, shuffle=False)
       
                           
# for _, data in enumerate(train_loader):
#     genes=data[0]
#     print(genes)
#     labels=data[1]
#     print(labels)



