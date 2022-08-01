from more_itertools import one
import numpy as np
import random
from typing import List
import pickle

def get_weight(bag_label_list):

    true_prob=np.array(bag_label_list).sum()/len(bag_label_list)
    false_prob=1-true_prob
    bag_class_weight=[1/true_prob,1/false_prob]

    return bag_class_weight

def gen_binary_list_non_all_mutation_one(gene_length: int, mutation_positions: List[int], num_snp_present:int) -> List[int]:

    # generate random list bases on the max limit
    present_pos=random.sample(range(gene_length),num_snp_present)
    random_list=np.zeros(gene_length).tolist()

    for pos in present_pos:
        random_list[pos]=1

    # check mutation constraint
    multation_label=[random_list[mutation_position] for mutation_position in mutation_positions]

    # if all the one -- flip random number of indices
    if all(v==1 for v in multation_label):
        #flipping for creating false bags
        num_flip=np.random.choice(range(1, len(mutation_positions)+1),1)
        flip_pos=random.sample(mutation_positions,int(num_flip))

        #also unflipping the corresponding number of SNPs
        zero_ind=[k for k, x in enumerate(random_list) if x == 0]
        if len(zero_ind)>int(num_flip):
            unflip_pos=random.sample(zero_ind, int(num_flip))
        else:
            unflip_pos=zero_ind

        for i in flip_pos:
            random_list[i]=0

        for j in unflip_pos:
            random_list[j]=1

    return random_list


def gen_binary_list_at_least_one_one(gene_length: int, mutation_positions: List[int], num_snp_present:int) -> List[int]:

    # generate random list bases on the max limit
    present_pos=random.sample(range(gene_length),num_snp_present)
    random_list=np.zeros(gene_length).tolist()

    for pos in present_pos:
        random_list[pos]=1

    # check mutation constraint
    mutation_label=[random_list[mutation_position] for mutation_position in mutation_positions]

    # if all of the casual mutation is false--- flip it to true
    if all(v==0 for v in mutation_label):
        #flipping for the true bags
        num_flip=np.random.choice(range(1, len(mutation_positions)+1),1)
        flip_pos=random.sample(mutation_positions,int(num_flip))

        #unflipping the corresponding number
        one_ind=[k for k, x in enumerate(random_list) if x == 1]
        if len(one_ind)>int(num_flip):
            unflip_pos=random.sample(one_ind, int(num_flip))
        else:
            unflip_pos=one_ind


        for i in flip_pos:
            random_list[i]=1

        for j in unflip_pos:
            random_list[j]=0

    return random_list

def gen_binary_list_all_mutation_one(gene_length: int, mutation_positions: List[int],num_snp_present:int) -> List[int]:
    
    all_position=list(range(gene_length))
    present_list=np.zeros(gene_length).tolist()   
    for j in mutation_positions:
        present_list[j]=1
        
    possible_present=[all_position[k] for k in all_position if k not in mutation_positions]

    present_position=random.sample(possible_present,(num_snp_present-len(mutation_positions)))

    for k in present_position:
        present_list[k]=1
    return present_list




def gen_binary_list_non_mutation_one(gene_length: int, mutation_positions: List[int],num_snp_present:int) -> List[int]:

    all_position=list(range(gene_length))
    present_list=np.zeros(gene_length).tolist()  

    possible_present=[all_position[k] for k in all_position if k not in mutation_positions]
       
    present_position=random.sample(possible_present,num_snp_present)
    for k in present_position:
        present_list[k]=1
    return present_list


def save_file(filename, data):
	'''
	Function for saving file in the pickle format
	Input
	-----------
	filename:  name of the output file
	data:      data to save
	Output
	----------
	'''
	with open(filename, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)









