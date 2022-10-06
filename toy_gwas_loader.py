from idna import valid_label_length
import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import pandas as pd

from utils import gen_binary_list_at_least_one_one, gen_binary_list_non_mutation_one, gen_binary_list_all_mutation_one, gen_binary_list_non_all_mutation_one
from sklearn.model_selection import train_test_split

gene_length=10
target_mutation_pos=[3,5]
target_mutation_val=0
num_genes=200
train=True
num_in_train=2

def generate_samples(gene_lenggen_binary_list_non_mutation_oneth, max_present,num_casual_snp, num_genes_train,num_genes_test, interaction=False):
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





def generate_samples_prev(gene_length, max_present,num_casual_snp, num_genes_train,num_genes_test, prevalence, interaction=False, seed=1,non_causal=0):
    #this function generate the true and false sample seperately to achieve the predefined prevalence
    random.seed(seed)
    np.random.seed(seed)

    # define the total number of genes and calculate the number of true and false gene
    num_genes=num_genes_train+num_genes_test+num_genes_test
    num_genes_true=int(prevalence*num_genes)
    num_genes_false=int((1-prevalence)*num_genes)

    # generate the casual snp position list
    target_mutation_pos=random.sample(range(gene_length),num_casual_snp)
    print("--The target mutation position is", sorted(target_mutation_pos))

    values_list=np.random.randint(0,4,gene_length)

    data_list_true=[]
    label_list_true=[]
    bag_label_list_true=[]
    data_list_false=[]
    label_list_false=[]
    bag_label_list_false=[]
    causal_ind_list_true=[]
    causal_ind_list_false=[]
    casualsnp_freq_list_true=[]
    casualsnp_freq_list_false=[]
    actual_present_true=[]
    actual_present_false=[]

    # generate true samples
    for i in range(0, num_genes_true):
        single_labels=[]
        data=[[]]*gene_length

        
        if interaction:
            num_snp_present=random.choices(range(num_casual_snp,max_present+1),k=1)[0]
            present_list=gen_binary_list_all_mutation_one(gene_length,target_mutation_pos, num_snp_present)

            actual_present=present_list.count(1)


            bag_label_check=[present_list[i] for i in target_mutation_pos]
            bag_label=all(bag_label_check)
            casualsnp_freq_true=bag_label_check.count(1)
            

        else:
            num_snp_present=random.choices(range(1,max_present+1),k=1)[0]
            present_list=gen_binary_list_at_least_one_one(gene_length,target_mutation_pos, num_snp_present)
            actual_present=present_list.count(1)


            bag_label_check=[present_list[i] for i in target_mutation_pos]
            bag_label=any(bag_label_check)
            casualsnp_freq_true=bag_label_check.count(1)


            
        for index in range(gene_length):
            values=values_list[index]
            item= [[index,values,present_list[index]]]
            data[index]=item

            if ((present_list[index]==1) & (index in target_mutation_pos)):
                label=True
            else:
                label=False    
            single_labels.append(label)

            
        causal_ind=[l for l, x in enumerate(single_labels) if x]
        causal_ind_list_true.append(causal_ind)
        casualsnp_freq_list_true.append(casualsnp_freq_true)
        actual_present_true.append(actual_present)



        data_list_true.append(data)
        
        label_list_true.append(single_labels)
        bag_label_list_true.append(bag_label)

    if not all(bag_label_list_true):
        print("!!!!!!!there is something wrong with true bag!!!!!!")

    print("-----------------------------------------------------------------------------")
    print("-----Table of casual SNP present stats true bag-------", pd.DataFrame(np.concatenate(causal_ind_list_true),columns =['causal_ind']).groupby(['causal_ind']).size())
    print("-----Frequency of casual SNP in True bag-----",pd.DataFrame(casualsnp_freq_list_true).describe())    
    print("-----Actual number of SNP present True bag----", pd.DataFrame(np.array(actual_present_true)/gene_length).describe())

    # generate false samples
    for j in range(0, num_genes_false):
        single_labels=[]        
        data=[[]]*gene_length

   
        if interaction:
            num_snp_present=random.choices(range(1,max_present+1),k=1)[0]
            present_list=gen_binary_list_non_all_mutation_one(gene_length,target_mutation_pos, num_snp_present)
            actual_present=present_list.count(1)

            bag_label_check=[present_list[i] for i in target_mutation_pos]
            bag_label=all(bag_label_check)
            casualsnp_freq_false=bag_label_check.count(1)

        
        else:
            if (max_present+len(target_mutation_pos))>gene_length:
                num_snp_present=random.choices(range(1,gene_length+1-len(target_mutation_pos)),k=1)[0]
            else:
                num_snp_present=random.choices(range(1,max_present+1),k=1)[0]
            present_list=gen_binary_list_non_mutation_one(gene_length,target_mutation_pos, num_snp_present)
            actual_present=present_list.count(1)

            bag_label_check=[present_list[i] for i in target_mutation_pos]
            bag_label=any(bag_label_check)
            casualsnp_freq_false=bag_label_check.count(1)
            
        for index in range(gene_length):
            

            values=values_list[index]
            item= [[index,values,present_list[index]]]
            data[index]=item

            if ((present_list[index]==1) & (index in target_mutation_pos)):
                label=True
            else:
                label=False         
            single_labels.append(label)


        causal_ind=[l for l, x in enumerate(single_labels) if x]
        causal_ind_list_false.append(causal_ind)
        casualsnp_freq_list_false.append(casualsnp_freq_false)
        actual_present_false.append(actual_present)

        data_list_false.append(data)

        label_list_false.append(single_labels)
        bag_label_list_false.append(bag_label)

    if any(bag_label_list_false):
        print("!!!!!!there is something wrong with false bag!!!!!!!")
    
    print("----------------------------------------------------------------------------------------")
    print("------Table of casual SNP present stats false bag-----", pd.DataFrame(np.concatenate(causal_ind_list_false),columns =['causal_ind']).groupby(['causal_ind']).size())
    print("------Frequency of casual SNP in False bag------",pd.DataFrame(casualsnp_freq_list_false).describe())    
    print("------Actual number of SNP present False bag-------", pd.DataFrame(np.array(actual_present_false)/gene_length).describe())




    #Combine the true and false back then shuffle them
    data_list= data_list_true + data_list_false 
    label_list=label_list_true+label_list_false
    if non_causal==1:
        # bag_label_list=random.choices([True, False],k=len(label_list))
        bag_label_list=[True]*num_genes_true+[False]*num_genes_false
        random.shuffle(bag_label_list)


    else:
        bag_label_list=bag_label_list_true+bag_label_list_false


    temp=list(zip(data_list, label_list, bag_label_list))
    random.shuffle(temp)
    data_list_out, label_list_out, bag_label_list_out=zip(*temp)
    data_list_out, label_list_out, bag_label_list_out=list(data_list_out), list(label_list_out), list(bag_label_list_out)

    train_data_list, valtest_data, train_bag_label_list, valtest_baglabel, train_label_list,valtest_label=train_test_split(data_list_out,bag_label_list_out, 
                                                                                                            label_list_out, test_size=1/3, random_state=1,stratify=bag_label_list_out)

    
    test_data_list, val_data_list, test_bag_label_list, val_bag_label_list, test_label_list, valid_label_list=train_test_split(valtest_data, valtest_baglabel, valtest_label, test_size=0.5, random_state=1, stratify= valtest_baglabel)                                                          

    
    #get present casual index 
    # causal_ind_list_train=[l for one_label in train_label_list for l, x in enumerate(one_label) if x]
    # causal_ind_list_test=[l for one_label in test_label_list for l, x in enumerate(one_label) if x]
    # causal_ind_list_val=[l for one_label in valid_label_list for l, x in enumerate(one_label) if x]
    
    # print("----------------------------------------------------------------------------")
    # print("------Casual SNP present stats TRAIN SET-----", pd.DataFrame(causal_ind_list_train,columns =['causal_ind']).groupby(['causal_ind']).size())
    # print("------Casual SNP present stats TEST SET-----", pd.DataFrame(causal_ind_list_test,columns =['causal_ind']).groupby(['causal_ind']).size())
    # print("------Casual SNP present stats VALID SET----", pd.DataFrame(causal_ind_list_val,columns =['causal_ind']).groupby(['causal_ind']).size())
    
    #casual snp combination
    causal_ind_list_train=[l for one_label in train_label_list for l, x in enumerate(one_label) if x]
    causal_ind_list_test=[l for one_label in test_label_list for l, x in enumerate(one_label) if x]
    causal_ind_list_val=[l for one_label in valid_label_list for l, x in enumerate(one_label) if x]

   
    #get present SNP frequency
    causal_snp_freq_train=[sum(one_label) for one_label in train_label_list]
    causal_snp_freq_test=[sum(one_label) for one_label in test_label_list]
    causal_snp_freq_val=[sum(one_label) for one_label in valid_label_list]


    print("----------------------------------------------------------------------------")
    print("-----Casual snp frequency for train bag is---------", pd.DataFrame(causal_snp_freq_train).describe())
    print("-----Casual snp frequency for test bag is----------", pd.DataFrame(causal_snp_freq_test).describe())
    print("-----Casual snp frequency for validation bag is----------", pd.DataFrame(causal_snp_freq_val).describe())

    #casual snp combination
    causal_ind_by_sample_list_train=[[l for l, x in enumerate(one_label) if x] for one_label in train_label_list]
    causal_ind_by_sample_list_test=[[l for l, x in enumerate(one_label) if x] for one_label in test_label_list]
    causal_ind_by_sample_list_val=[[l for l, x in enumerate(one_label) if x] for one_label in valid_label_list]

    print("----------------------------------------------------------------------------")
    print("------Casual SNP present stats TRAIN SET-----", pd.DataFrame(np.concatenate(causal_ind_by_sample_list_train),columns =['causal_ind'],dtype=int).groupby(['causal_ind']).size())
    print("------Casual SNP present stats TEST SET-----", pd.DataFrame(np.concatenate(causal_ind_by_sample_list_test),columns =['causal_ind'],dtype=int).groupby(['causal_ind']).size())
    print("------Casual SNP present stats VALID SET----", pd.DataFrame(np.concatenate(causal_ind_by_sample_list_val),columns =['causal_ind'],dtype=int).groupby(['causal_ind']).size())
    

    causal_ind_by_sample_list_train=[[''.join(str(x) for x in k)]for k in causal_ind_by_sample_list_train]
    causal_ind_by_sample_list_test=[[''.join(str(x) for x in k)]for k in causal_ind_by_sample_list_test]
    causal_ind_by_sample_list_val=[[''.join(str(x) for x in k)]for k in causal_ind_by_sample_list_val]

    print("----------------------------------------------------------------------------")
    print("-----Casual SNP combination stats TRAIN SET----------",pd.DataFrame(causal_ind_by_sample_list_train,columns=['casual combination']).groupby(['casual combination']).size())
    print("-----Casual SNP combination stats TEST SET----------", pd.DataFrame(causal_ind_by_sample_list_test,columns=['casual combination']).groupby(['casual combination']).size())
    print("-----Casual SNP combination stats VALID SET--------",pd.DataFrame(causal_ind_by_sample_list_val,columns=['casual combination']).groupby(['casual combination']).size())



    
    return train_data_list, train_bag_label_list, train_label_list, test_data_list, test_bag_label_list,test_label_list,val_data_list,valid_label_list,val_bag_label_list, sorted(target_mutation_pos)


# gene_length=10
# max_present=8
# num_casual_snp=4
# num_genes_train=20
# num_genes_test=8
# prevalence=0.35
# interaction=False


# train_data_list, train_bag_label_list, train_label_list, test_data_list, test_bag_label_list,test_label_list,val_data_list,valid_label_list,val_bag_label_list=generate_samples_prev(gene_length, max_present ,num_casual_snp, num_genes_train,num_genes_test, prevalence, interaction=interaction,non_causal=1)

# print(train_data_list[0])
# print(test_data_list[0])





