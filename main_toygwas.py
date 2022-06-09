from __future__ import print_function
from code import interact
from tkinter import Label

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from toy_gwas_loader import generate_samples, generate_samples_prev
from model_gwas import Attention, GatedAttention
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os

from collections import Counter
import random
import collections
from utils import get_weight

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GWAS Toy Example')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--num_bags_train', type=int, default=800, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=200, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

parser.add_argument('--num_snp',type=int, default=10,help='number of SNP in every sample')
parser.add_argument('--max_present',type=int, default=8, help='maximun number of present SNP in every sample')
parser.add_argument('--num_casual_snp', type=int, default=3, help='number of ground truth causal SNP')
parser.add_argument('--interaction',type=int,default=True, help='if assume there is interaction between casual SNP')
parser.add_argument('--oversampling',type=bool,default=True, help='if using upsampling in training')
parser.add_argument('--weight_loss',type=bool,default=True, help='if using weighted loss in training')
parser.add_argument('--prevalence',type=float,default=0.3, help='the ratio of true bag and false bag in generated samples')
parser.add_argument('--control_prevalence',type=bool,default=True, help='if we control prevalence when generating samples')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.control_prevalence:
    # prevalence as parameter sample generation
    data_list_train, bag_label_list_train, label_list_train, data_list_test, bag_label_list_test,label_list_test=generate_samples_prev(gene_length=args.num_snp, max_present=args.max_present ,num_casual_snp=args.num_casual_snp, num_genes_train=args.num_bags_train,num_genes_test=args.num_bags_test, prevalence=args.prevalence, interaction=args.interaction)

else:
    # without controling prevalence sample generation
    data_list_train, bag_label_list_train, label_list_train, data_list_test,bag_label_list_test,label_list_test=generate_samples(gene_length=args.num_snp,max_present=args.max_present,num_casual_snp=args.num_casual_snp,num_genes_train=args.num_bags_train,num_genes_test=args.num_bags_test,interaction=args.interaction)



bag_class_weight_train=get_weight(bag_label_list_train)
bag_class_weight_test=get_weight(bag_label_list_test)


overampling=args.oversampling

if (1/bag_class_weight_train[0]<0.2) & (overampling==True):
    print('Using resampling')
    true_bag=[i for i, x in enumerate(bag_label_list_train) if x[0]]
    res_ind=random.choices(true_bag,k=int(len(bag_label_list_train)*0.5))
    counter=collections.Counter(res_ind)

    print('The three most commom samples', counter.most_common(3),'the total length of append dataset is', len(res_ind))

    data_list_res=[data_list_train[j] for j in res_ind]
    bag_label_list_res=[bag_label_list_train[j] for j in res_ind]
    label_list_train_res=[label_list_train[j] for j in res_ind]

    data_list_train+=data_list_res
    bag_label_list_train+=bag_label_list_res
    label_list_train+=label_list_train_res

    bag_class_weight_train=get_weight(bag_label_list_train)

elif 1/bag_class_weight_train[0]<0.2:
    print('Using undersampling')
    false_bag=[i for i, x in enumerate(bag_label_list_train) if x[0]==False]
    drop_ind=random.sample(false_bag,k=int(len(false_bag)*0.5))
    keep_ind=[i for i in range(len(data_list_train)) if i not in drop_ind]
    
    data_list_train=[data_list_train[j] for j in keep_ind]
    bag_label_list_train=[bag_label_list_train[j] for j in keep_ind]
    label_list_train=[label_list_train[j] for j in keep_ind]

    bag_class_weight_train=get_weight(bag_label_list_train)



train_data=TensorDataset(torch.tensor(data_list_train),torch.tensor(bag_label_list_train),torch.tensor(label_list_train))

train_loader =DataLoader(train_data,batch_size=1, shuffle=True)

test_data=TensorDataset(torch.tensor(data_list_test,dtype=torch.int32),torch.tensor(bag_label_list_test),torch.tensor(label_list_test))
test_loader =DataLoader(test_data,batch_size=1, shuffle=False)


sharedParams = {'weight_train': bag_class_weight_train,
'weight_test': bag_class_weight_test}


print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch,bag_class_weight_train, weight):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, bag_label, label) in enumerate(train_loader):
        # bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        # print('\ndata: ',data)
        # print('\nlabel:',bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)

        if args.weight_loss:
            if bag_label:
                weighted_loss=bag_class_weight_train[0]*loss
            else:
                weighted_loss=bag_class_weight_train[1]*loss
            train_loss += weighted_loss.data[0]

        else:
            train_loss += loss.data[0]
        
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    return train_loss


def test(PATH):

    #Using checkpoint to evaluate the model
    if args.model=='attention':
        model = Attention()
    elif args.model=='gated_attention':
        model = GatedAttention()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    if args.control_prevalence:
        PATH_LOAD=PATH+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}.pt'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence)

    else:
        PATH_LOAD=PATH+'/nsnp{}_max{}_csnp{}_i{}.pt'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction)

    checkpoint = torch.load(PATH_LOAD)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('We use the model in traing epoch', epoch, 'the loss was', float(loss))

    model.eval()
    test_loss = 0.
    test_error = 0.
    pred_label_list=[]
    true_label_list=[]

    rightattention_count=0.
    total_count=0.

    attention_array_list=[]
    single_labels_list=[]
    y_prob_list=[]


    for batch_idx, (data, bag_label,label) in enumerate(test_loader):
        # bag_label = label[0]
        instance_labels = label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        y_prob,_ ,_ = model.forward(data)
        y_prob=torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)

        true_label_list.append(bag_label)
        pred_label_list.append(predicted_label)
        y_prob_list.append(y_prob.detach().numpy())

        if predicted_label.cpu().data.numpy()[0][0]==1:
           attention_array=attention_weights.cpu().data.numpy()[0]

           #counting for calculating probability of max weight if true label
           max_value=max(attention_array)
           max_attention= [i for i, j in enumerate(attention_array) if j == max_value]
           total_count+=1
           single_labels=instance_labels.numpy()[0].tolist()
           if single_labels[max_attention[0]]:
               rightattention_count+=1 

           #prepare list for instance level ROC
           attention_array_list.append(attention_array)
           single_labels_list.append(single_labels)

           



        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))




    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('confusion matrix:',confusion_matrix(np.concatenate(true_label_list), np.concatenate(pred_label_list)))

    if total_count==0:
        print('The estimated probability of the right largest attention is',rightattention_count/(total_count+0.000000000001))
    else:
        print('The estimated probability of the right largest attention is',rightattention_count/total_count)


    

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))

    #additional matrics and plots bag level

    fpr, tpr, threshold_roc=roc_curve(np.concatenate(true_label_list), np.concatenate(y_prob_list))
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds_prc = precision_recall_curve(np.concatenate(true_label_list), np.concatenate(y_prob_list))
    prc_avg = average_precision_score(np.concatenate(true_label_list),np.concatenate(y_prob_list))

    #instance level evaluations
    instance_level_score=np.concatenate(attention_array_list)
    instance_level_truth=np.concatenate(single_labels_list)

    precision_instance, recall_instance, thresholds_prc_instance = precision_recall_curve(instance_level_truth, instance_level_score)
    prc_avg_instance = average_precision_score(instance_level_truth, instance_level_score)

    fpr_instance, tpr_instance, threshold_roc_instance=roc_curve(instance_level_truth, instance_level_score)
    roc_auc_instance = auc(fpr_instance, tpr_instance)


    figure, axis = plt.subplots(2, 2, figsize=(7, 7))

    axis[0, 0].set_title('Bag level ROC')
    axis[0, 0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    axis[0, 0].legend(loc = 'lower right')
    axis[0, 0].plot([0, 1], [0, 1],'r--')
    axis[0, 0].set_xlim([0, 1])
    axis[0, 0].set_ylim([0, 1])
    axis[0, 0].set_ylabel('True Positive Rate')
    axis[0, 0].set_xlabel('False Positive Rate')


    # plt.subplot(1, 2, 2) 
    axis[0, 1].set_title('Bag level PRC')
    axis[0, 1].plot(recall, precision , 'b', label = 'AP = %0.2f' % prc_avg)
    axis[0, 1].legend(loc = 'lower left')
    axis[0, 1].set_xlim([0, 1])
    axis[0, 1].set_ylim([0, 1])
    axis[0, 1].set_xlabel('Recall')
    axis[0, 1].set_ylabel('Precision')

    axis[1, 0].set_title('Instance level ROC')
    axis[1, 0].plot(fpr_instance, tpr_instance, 'b', label = 'AUC = %0.2f' % roc_auc_instance)
    axis[1, 0].legend(loc = 'lower right')
    axis[1, 0].plot([0, 1], [0, 1],'r--')
    axis[1, 0].set_xlim([0, 1])
    axis[1, 0].set_ylim([0, 1])
    axis[1, 0].set_ylabel('True Positive Rate')
    axis[1, 0].set_xlabel('False Positive Rate')

    axis[1, 1].set_title('Instance level PRC')
    axis[1, 1].plot(recall_instance, precision_instance , 'b', label = 'AP = %0.2f' % prc_avg_instance)
    axis[1, 1].legend(loc = 'lower left')
    axis[1, 1].set_xlim([0, 1])
    axis[1, 1].set_ylim([0, 1])
    axis[1, 1].set_xlabel('Recall')
    axis[1, 1].set_ylabel('Precision')


    plt.tight_layout()

    # plt.show()
    SAVING_PATH=os.getcwd()+'/plots'
    os.makedirs(SAVING_PATH, exist_ok=True)
    if args.prevalence:
        PLOT_PATH=SAVING_PATH+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}.png'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence)

    else:
        PLOT_PATH=SAVING_PATH+'/nsnp{}_max{}_csnp{}_i{}.png'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction)
    plt.savefig(PLOT_PATH)



if __name__ == "__main__":
    print('Start Training')
    print('training weight:', bag_class_weight_train)
    working_dir=os.getcwd() 
    PATH=working_dir+'/checkpoints'

    os.makedirs(PATH, exist_ok=True)
    if args.control_prevalence:
        PATH_SAVE=PATH+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}.pt'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence)

    else:
        PATH_SAVE=PATH+'/nsnp{}_max{}_csnp{}_i{}.pt'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction)
    

    min_loss=100
    for epoch in range(1, args.epochs + 1):
        train_loss=train(epoch,bag_class_weight_train,weight=True)

        if train_loss<min_loss:
            min_loss=train_loss
            epoch_min=epoch

    #save checkpoint of the model
    torch.save({'epoch':epoch_min, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss':min_loss}, PATH_SAVE)

    print('Start Testing')
    print('training weight:', bag_class_weight_test)
    test(PATH)
