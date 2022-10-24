from __future__ import print_function
import numpy as np
import argparse
import torch
import random
import os
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import collections
from utils import get_weight, save_file, load_file
import matplotlib.pyplot as plt
from A_thaliana.gen_semi_natural_setting_inputs import gene_data_gen
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch GWAS Toy')
parser.add_argument('--reg', type=float, default=10e-5,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-int','--interaction',type=int,default=0,  help='if assume there is interaction between casual SNP')
parser.add_argument('-osampling','--oversampling',type=bool,default=True, help='if using upsampling in training')
parser.add_argument('-wloss','--weight_loss',type=bool,default=True, help='if using weighted loss in training')
parser.add_argument('--selected_length',type=int,default=20, help='selected length from nature data')
parser.add_argument('--gene_ind',type=int,default=3, help='selected gene index')
parser.add_argument('--non_causal',type=int,default=0, help='if we want to set casual snp in bag')
parser.add_argument('--model',type=str,default="logistic", help='choosing the baseline model')
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


if args.selected_length<500:
    present_df=load_file("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/selected_gene_sample_snplength{}_alogPICK.pkl".format(args.selected_length))
else:
    present_df=load_file("/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/selected_gene_sample_snplength500greaterthan_alogPICK.pkl")

bag_label_list, data_list, single_labels_list=gene_data_gen(args.gene_ind,present_df, args.selected_length, args.interaction)
data_list_train, valtest_data, bag_label_list_train, valtest_baglabel, label_list_train,valtest_label=train_test_split(data_list,bag_label_list, 
                                                                                                            single_labels_list, test_size=1/3, random_state=1,stratify=bag_label_list)
data_list_test, val_data_list, bag_label_list_test, val_bag_label_list, label_list_test, val_label_list=train_test_split(valtest_data, valtest_baglabel, valtest_label, test_size=0.5, random_state=1, stratify= valtest_baglabel)                                                          


bag_class_weight_train=get_weight(bag_label_list_train)
bag_class_weight_test=get_weight(bag_label_list_test)
bag_class_weight_val=get_weight(val_bag_label_list)
if args.cuda:
    bag_class_weight_train=np.array(bag_class_weight_train)
    bag_class_weight_train=torch.from_numpy(bag_class_weight_train)
    bag_class_weight_val=get_weight(val_bag_label_list)
    bag_class_weight_train.cuda()
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

def extract_moments(data_list_train, bag_label):
    data_out=[]
    bag_out=[]
    for i, data in enumerate(data_list_train): 
        present_identifier=[dat[0] for dat in data if dat[2]==1]
        m1=np.mean(present_identifier)
        m2=np.std(present_identifier)**2
        m3=skew(present_identifier)
        m4=kurtosis(present_identifier)
        if not np.isnan([m1,m2,m3,m4]).any():
            data_out.append([m1,m2,m3,m4])
            bag_out.append(bag_label[i])
    return data_out, bag_out


data_moment_train,bag_label_train=extract_moments(data_list_train,bag_label_list_train)
data_moment_test,bag_label_test=extract_moments(data_list_test,bag_label_list_test)

if args.model=="logistic":
    #fiting logistic regression
    clf = LogisticRegression(random_state=0).fit(data_moment_train, bag_label_train)
    
elif args.model=="random_forest":
    #fiting random forest
    clf = RandomForestClassifier(n_estimators=200,random_state=args.seed)
    clf.fit(data_moment_train, bag_label_list_train)
   


predictions=clf.predict(data_moment_test)
prediction_prob=clf.predict_proba(data_moment_test)[::,1]
print(confusion_matrix(bag_label_test,predictions))


# Matrics and plots bag level
fpr, tpr, threshold_roc=roc_curve(bag_label_test, prediction_prob)
roc_auc = auc(fpr, tpr)

precision, recall, thresholds_prc = precision_recall_curve(bag_label_test, prediction_prob)
prc_avg = average_precision_score(bag_label_test, prediction_prob)

SAVING_METRIC_PATH="/home/zixshu/DeepGWAS/baseline/metrics_fourmoments_semi_{}_manual_all/{}/".format(args.model, args.gene_ind)
os.makedirs(SAVING_METRIC_PATH, exist_ok=True)
EVALUATION_PATH=SAVING_METRIC_PATH+'evaluation_score_selectedlength{}_i{}.pkl'.format(args.selected_length,args.interaction)
evaluation_dict={}
evaluation_dict['fpr']=fpr
evaluation_dict['tpr']=tpr
evaluation_dict['precision']=precision
evaluation_dict['recall']=recall
evaluation_dict['roc_auc']=roc_auc 
evaluation_dict['prc_avg']=prc_avg
save_file(EVALUATION_PATH,evaluation_dict)

figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle('selectedlength{}_i{}.png'.format(args.selected_length,args.interaction), fontsize=16)

axis[0].set_title('Bag level ROC')
axis[0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
axis[0].legend(loc = 'lower right')
axis[0].plot([0, 1], [0, 1],'r--')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1])
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')


# plt.subplot(1, 2, 2) 
axis[1].set_title('Bag level PRC')
axis[1].plot(recall, precision , 'b', label = 'AP = %0.2f' % prc_avg)
axis[1].legend(loc = 'lower left')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].axhline(y=0.35, color='grey', linestyle='dotted')
plt.tight_layout()

SAVING_PATH="/home/zixshu/DeepGWAS/baseline/plots_baselines_fourmoments_semi_{}_manual_all/{}/".format(args.model, args.gene_ind)
os.makedirs(SAVING_PATH, exist_ok=True)

PLOT_PATH=SAVING_PATH+'selectedlength{}_i{}.png'.format(args.selected_length,args.interaction)
plt.savefig(PLOT_PATH)







