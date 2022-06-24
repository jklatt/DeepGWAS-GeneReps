from idna import valid_label_length
import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import random


from utils import gen_binary_list_at_least_one_one, gen_binary_list_non_mutation_one, gen_binary_list_all_mutation_one, gen_binary_list_non_all_mutation_one


gene_length=10
target_mutation_pos=[3,5]
target_mutation_val=0
num_genes=200
train=True
num_in_train=2

def generate_samples(gene_length, max_present,num_casual_snp, num_genes_train,num_genes_test, interaction=False):
    # generate some toy sample of GWAS data with integers
    # gene_length=np.int(np.random.normal(gene_length_mean,gene_length_var,1))
    random.seed(1)
    np.random.seed(4)
    num_genes=num_genes_train+num_genes_test+num_genes_test

    target_mutation_pos=random.sample(range(gene_length),num_casual_snp)
    
    data_list=[]
    label_list=[]
    bag_label_list=[]
    num_casual_snp_list=random.choices(range(0,max_present+1),k=num_genes)
    
    for k in range(0,num_genes):
        data=[[]]*gene_length
        label=[]
        
        present_snp=random.sample(range(gene_length), num_casual_snp_list[k])
        present_list=np.zeros(gene_length)
        for l in present_snp:
            present_list[l]=1

        for i in range(0,gene_length):
            values = randint(0, 4)
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
    test_data_list=data_list[num_genes_train:num_genes_train+num_genes_test]
    val_data_list=data_list[num_genes_train+num_genes_test:]


    train_label_list= label_list[:num_genes_train]
    test_label_list= label_list[num_genes_train:num_genes_train+num_genes_test]
    val_label_list= label_list[num_genes_train+num_genes_test:]


    train_bag_label_list= bag_label_list[:num_genes_train]
    test_bag_label_list = bag_label_list[num_genes_train:num_genes_train+num_genes_test]
    val_bag_label_list=bag_label_list[num_genes_train+num_genes_test:]



    return train_data_list, train_bag_label_list, train_label_list, test_data_list, test_bag_label_list,test_label_list,val_data_list,val_bag_label_list,val_label_list

# train_data_list, train_bag_label_list, train_label_list, test_data_list, test_label_list, test_bag_label_list=generate_samples(gene_length=10,
# max_present=8,num_casual_snp=2,num_genes_train=200,num_genes_test=200,interaction=True)

# train_data=TensorDataset(torch.tensor(data_list),torch.tensor(bag_label_list),torch.tensor(label_list))
# train_loader =DataLoader(train_data,batch_size=num_in_train, shuffle=False)
       
                           
# for _, data in enumerate(train_loader):
#     genes=data[0]
#     print(genes)
#     labels=data[1]
#     print(labels)





def generate_samples_prev(gene_length, max_present,num_casual_snp, num_genes_train,num_genes_test, prevalence, interaction=False):
    #this function generate the true and false sample seperately to achieve the predefined prevalence
    random.seed(1)
    np.random.seed(4)

    # define the total number of genes and calculate the number of true and false gene
    num_genes=num_genes_train+num_genes_test+num_genes_test
    num_genes_true=int(prevalence*num_genes)
    num_genes_false=int((1-prevalence)*num_genes)

    # generate the casual snp position list
    target_mutation_pos=random.sample(range(gene_length+1),num_casual_snp)

    data_list_true=[]
    label_list_true=[]
    bag_label_list_true=[]
    data_list_false=[]
    label_list_false=[]
    bag_label_list_false=[]

    # generate true samples
    for i in range(0, num_genes_true):
        single_labels=[]
        data=[[]]*gene_length
        
        if interaction:
            num_snp_present=random.choices(range(len(target_mutation_pos),max_present+1),k=1)[0]
            present_list=gen_binary_list_all_mutation_one(gene_length,target_mutation_pos, num_snp_present)
            bag_label_check=[present_list[i] for i in target_mutation_pos]
            bag_label=all(bag_label_check)
            

        else:
            num_snp_present=random.choices(range(1,max_present+1),k=1)[0]
            present_list=gen_binary_list_at_least_one_one(gene_length,target_mutation_pos, num_snp_present)
            bag_label_check=[present_list[i] for i in target_mutation_pos]
            bag_label=any(bag_label_check)

            
        for index in range(gene_length):
            values = randint(0, 4)
            item= [[index,values,present_list[index]]]
            data[index]=item

        
        for g in range(gene_length):
            if ((present_list[g]==1) & (g in target_mutation_pos)):
               label=True
            else:
               label=False    
            single_labels.append(label)
        
        

        data_list_true.append(data)
        label_list_true.append(single_labels)
        bag_label_list_true.append(bag_label)
        if not all(bag_label_list_true):
            print("there is something wrong with true bag")
        

    # generate false samples
    for j in range(0, num_genes_false):
        single_labels=[]
        data=[[]]*gene_length
        num_snp_present=random.choices(range(0,max_present+1),k=1)[0]
        
        if interaction:
           present_list=gen_binary_list_non_all_mutation_one(gene_length,target_mutation_pos, num_snp_present)
           bag_label_check=[present_list[i] for i in target_mutation_pos]
           bag_label=all(bag_label_check)

        else:
           present_list=gen_binary_list_non_mutation_one(gene_length,target_mutation_pos, num_snp_present)
           bag_label_check=[present_list[i] for i in target_mutation_pos]
           bag_label=any(bag_label_check)
            
        for index in range(gene_length):
            values = randint(0, 3)
            item= [[index,values,present_list[index]]]
            data[index]=item

        for g in range(gene_length):
            if ((present_list[g]==1) & (g in target_mutation_pos)):
               label=True
            else:
               label=False         
            single_labels.append(label)

        

        data_list_false.append(data)
        label_list_false.append(single_labels)
        bag_label_list_false.append(False)
        if any(bag_label_list_false):
            print("there is something wrong with false bag")

    #Combine the true and false back then shuffle them
    data_list= data_list_true + data_list_false 
    label_list=label_list_true+label_list_false
    bag_label_list=bag_label_list_true+bag_label_list_false


    temp=list(zip(data_list, label_list, bag_label_list))
    random.shuffle(temp)
    data_list_out, label_list_out, bag_label_list_out=zip(*temp)
    data_list_out, label_list_out, bag_label_list_out=list(data_list_out), list(label_list_out), list(bag_label_list_out)



    # output the train and test set seperately
    train_data_list=data_list_out[:num_genes_train]
    test_data_list=data_list_out[num_genes_train:num_genes_train+num_genes_test]
    val_data_list=data_list_out[num_genes_train+num_genes_test:]
    

    train_label_list= label_list_out[:num_genes_train]
    test_label_list= label_list_out[num_genes_train:num_genes_train+num_genes_test]
    valid_label_list=label_list_out[num_genes_train+num_genes_test:]

    train_bag_label_list= bag_label_list_out[:num_genes_train]
    test_bag_label_list =bag_label_list_out[num_genes_train:num_genes_train+num_genes_test]
    val_bag_label_list=bag_label_list_out[num_genes_train+num_genes_test:]

    return train_data_list, train_bag_label_list, train_label_list, test_data_list, test_bag_label_list,test_label_list,val_data_list,valid_label_list,val_bag_label_list


# gene_length=10
# max_present=8
# num_casual_snp=4
# num_genes_train=10
# num_genes_test=8
# prevalence=0.2
# interaction=False


# train_data_list, train_bag_label_list, train_label_list, test_data_list, test_bag_label_list,test_label_list=generate_samples_prev(gene_length, max_present ,num_casual_snp, num_genes_train,num_genes_test, prevalence, interaction=True)

# print(train_data_list[0])
# print(test_data_list[0])





