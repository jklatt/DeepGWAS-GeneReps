from __future__ import print_function
import numpy as np
import argparse
import torch
from toy_gwas_loader import generate_samples, generate_samples_prev
import random

import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import collections
from utils import get_weight, save_file

parser = argparse.ArgumentParser(description='PyTorch GWAS Toy')

parser.add_argument('--epochs', type=int, default=500,)
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5,
                    help='weight decay')
parser.add_argument('--num_bags_train', type=int, default=800, 
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=200,
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('-nsnp','--num_snp',type=int, default=20,help='number of SNP in elvery sample')
parser.add_argument('-maxp','--max_present',type=float, default=1,  help='maximun number of present SNP in every sample')
parser.add_argument('-ncsnp','--num_casual_snp', type=int, default=3, help='number of ground truth causal SNP')
parser.add_argument('-int','--interaction',type=int,default=0,  help='if assume there is interaction between casual SNP')
parser.add_argument('-osampling','--oversampling',type=bool,default=True, help='if using upsampling in training')
parser.add_argument('-wloss','--weight_loss',type=bool,default=True, help='if using weighted loss in training')
parser.add_argument('-pre','--prevalence',type=float,default=0.1, help='the ratio of true bag and false bag in generated samples')
parser.add_argument('-cprgevalene','--control_prevalence',type=bool,default=True, help='if we control prevalence when generating samples')
parser.add_argument('--non_causal',type=int,default=0, help='if we want to set casual snp in bag')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

torch.manual_seed(args.seed)

np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#converting tshe 1 0 into boolean value
if args.interaction==1:
    args.interaction=True
else:
    args.interaction=False

if args.control_prevalence:
    # prevalence as parameter sample generation
    data_list_train, bag_label_list_train, label_list_train, data_list_test, bag_label_list_test,label_list_test,val_data_list,val_label_list, val_bag_label_list,target_mutation=generate_samples_prev(gene_length=args.num_snp, 
    max_present=int(args.max_present*args.num_snp) ,num_casual_snp=args.num_casual_snp, num_genes_train=args.num_bags_train,num_genes_test=args.num_bags_test, prevalence=args.prevalence, interaction=args.interaction,seed=args.seed, non_causal=args.non_causal)

else:
    # without controling prevalence sample generation
    data_list_train, bag_label_list_train, label_list_train, data_list_test,bag_label_list_test,label_list_test,val_data_list,val_bag_label_list,val_label_list,target_mutation=generate_samples(gene_length=args.num_snp,
    max_present=int(args.max_present*args.num_snp),num_casual_snp=args.num_casual_snp,num_genes_train=args.num_bags_train,num_genes_test=args.num_bags_test,interaction=args.interaction,seed=args.seed, non_causal=args.non_causal)

bag_class_weight_train=get_weight(bag_label_list_train)
bag_class_weight_test=get_weight(bag_label_list_test)
bag_class_weight_val=get_weight(val_bag_label_list)
if args.cuda:
    bag_class_weight_train=np.array(bag_class_weight_train)
    bag_class_weight_train=torch.from_numpy(bag_class_weight_train)
    bag_class_weight_val=get_weight(val_bag_label_list)
    bag_class_weight_train.cuda()
    # print("weight is on cuda", bag_class_weight_train.get_device())


overampling=args.oversampling

if (1/bag_class_weight_train[0]<0.5) & (overampling==True):
    print('Using resampling')
    # this here change to upsampling to balance dataset in the training model
    true_bag=[i for i, x in enumerate(bag_label_list_train) if x]
    res_ind=random.choices(true_bag,k=int(len(bag_label_list_train)*(0.5-1/bag_class_weight_train[0])))
    counter=collections.Counter(res_ind)

    print('The three most commom samples', counter.most_common(3),'the total length of append dataset is', len(res_ind))

    data_list_res=[data_list_train[j] for j in res_ind]
    bag_label_list_res=[bag_label_list_train[j] for j in res_ind]
    label_list_train_res=[label_list_train[j] for j in res_ind]

    data_list_train+=data_list_res
    bag_label_list_train+=bag_label_list_res
    label_list_train+=label_list_train_res

    bag_class_weight_train=get_weight(bag_label_list_train)
    

elif 1/bag_class_weight_train[0]<0.5:
    print('Using undersampling')
    false_bag=[i for i, x in enumerate(bag_label_list_train) if x[0]==False]
    drop_ind=random.sample(false_bag,k=int(len(false_bag)*0.5))
    keep_ind=[i for i in range(len(data_list_train)) if i not in drop_ind]
    
    data_list_train=[data_list_train[j] for j in keep_ind]
    bag_label_list_train=[bag_label_list_train[j] for j in keep_ind]
    label_list_train=[label_list_train[j] for j in keep_ind]

    bag_class_weight_train=get_weight(bag_label_list_train)

def extract_moments(data_list_train):
    data_out=[]
    for data in data_list_train: 
        m1=np.mean(data,axis=0)[0][2]
        m2=np.std(data,axis=0)[0][2]**2
        m3=skew(data,axis=0)[0][2]
        m4=kurtosis(data,axis=0)[0][2]

        m5=np.mean(data,axis=0)[0][1]
        m6=np.std(data,axis=0)[0][1]**2
        m7=skew(data,axis=0)[0][1]
        m8=kurtosis(data,axis=0)[0][1]
        data_out.append([m1,m2,m3,m4,m5,m6,m7,m8])
    return data_out

data_moment_train=extract_moments(data_list_train)
data_moment_valid=extract_moments(val_data_list)
data_moment_test=extract_moments(data_list_test)

#fiting random forest
clf = RandomForestClassifier(n_estimators=400,max_depth=50, random_state=1)
clf.fit(data_moment_train, bag_label_list_train)

#fiting logistic regression
# clf = LogisticRegression(random_state=0).fit(data_moment_train, bag_label_list_train)

predictions=clf.predict(data_moment_test)
print(confusion_matrix(bag_label_list_test,predictions))











