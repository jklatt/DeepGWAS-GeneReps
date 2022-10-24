from __future__ import print_function
from ast import arg
from statistics import variance
# from code import interact
# from tkinter import Label

import numpy as np
import argparse
import torch
# import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from toy_gwas_loader import generate_samples, generate_samples_prev
from model_gwas import Attention, GatedAttention, SetTransformer, Attention_onlypresent, GatedAttention_onlypresent
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os

from collections import Counter
import random
import collections
from utils import get_weight, save_file
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

# Training settings
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
parser.add_argument('--model', type=str, default='attention_onlypresent', help='Choose b/w attention and gated_attention')
parser.add_argument('-nsnp','--num_snp',type=int, default=30,help='number of SNP in elvery sample')
parser.add_argument('-maxp','--max_present',type=float, default=0.3,  help='maximun number of present SNP in every sample')
parser.add_argument('-ncsnp','--num_casual_snp', type=int, default=3, help='number of ground truth causal SNP')
parser.add_argument('-int','--interaction',type=int,default=0,  help='if assume there is interaction between casual SNP')
parser.add_argument('-osampling','--oversampling',type=bool,default=True, help='if using upsampling in training')
parser.add_argument('-wloss','--weight_loss',type=bool,default=True, help='if using weighted loss in training')
parser.add_argument('-pre','--prevalence',type=float,default=0.1, help='the ratio of true bag and false bag in generated samples')
parser.add_argument('-cprgevalene','--control_prevalence',type=bool,default=True, help='if we control prevalence when generating samples')
parser.add_argument('--non_causal',type=int,default=0, help='if we want to set casual snp in bag')
parser.add_argument('--onlypresent',type=int,default=1, help='only present indentifier model')
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

if args.onlypresent==0:
    args.onlypresent=False
else:
    args.onlypresent=True


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




train_data=TensorDataset(torch.tensor(data_list_train),torch.tensor(bag_label_list_train),torch.tensor(label_list_train))
train_loader =DataLoader(train_data,batch_size=1, shuffle=True)

test_data=TensorDataset(torch.tensor(data_list_test,dtype=torch.int32),torch.tensor(bag_label_list_test),torch.tensor(label_list_test))
test_loader =DataLoader(test_data,batch_size=1, shuffle=False)

validation_data=TensorDataset(torch.tensor(val_data_list,dtype=torch.int32),torch.tensor(val_bag_label_list),torch.tensor(val_label_list))
validation_loader =DataLoader(validation_data,batch_size=1, shuffle=False)


sharedParams = {'weight_train': bag_class_weight_train,
'weight_test': bag_class_weight_test,
'weight_val':bag_class_weight_val}


print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
elif args.model=='set_transformer':
    model = SetTransformer()
elif args.model=='attention_onlypresent':
    model=Attention_onlypresent()
elif args.model=='gated_attention_onlypresent':
    model=GatedAttention_onlypresent()
    
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch,bag_class_weight_train, weight):
    model.train()
    train_loss = 0.
    train_error = 0.
    
    for batch_idx, (data, bag_label, label) in enumerate(train_loader):
        # bag_label = label[0]
        if args.onlypresent:
            data=data.squeeze(0)
            data=[[dat[0][0]] for dat in data if dat[0][2]==1]
            data=torch.tensor(data)

        if len(data)>0:

            data=data.type(torch.FloatTensor)
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
                # print("data is on", data.get_device())
                # print("bag_label is on", bag_label.get_device())
            data, bag_label = Variable(data), Variable(bag_label)
            


            # print('\ndata: ',data)
            # print('\nlabel:',bag_label)

            # reset gradients
            optimizer.zero_grad()


            # calculate loss and metrics 
            loss, _ = model.calculate_objective(data, bag_label)
            bag_class_weight_train=torch.tensor(bag_class_weight_train)
            bag_class_weight_train=Variable(bag_class_weight_train,requires_grad=True)

            if args.model!="set_transformer":
                if args.weight_loss:
                    if bag_label:
                        weighted_loss=bag_class_weight_train[0]*loss
                    else:
                        weighted_loss=bag_class_weight_train[1]*loss
                    train_loss += weighted_loss.data[0]

                else:
                    train_loss += loss.data[0]

            else:
                if args.weight_loss:
                    if bag_label:
                        weighted_loss=bag_class_weight_train[0]*loss
                    else:
                        weighted_loss=bag_class_weight_train[1]*loss
                    train_loss += weighted_loss

                else:
                    train_loss += loss


            error, _ = model.calculate_classification_error(data, bag_label)
            train_error += error

            # weighted_loss=Variable(weighted_loss,requires_grad=True)

            # backward pass
            weighted_loss.backward()#!!! here changed

            # step
            optimizer.step()
        else:
            pass


    # calculate loss and error for epochl
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    if args.model!="set_transformer":
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    else:
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu(), train_error))
    return train_loss

def val():
    model.eval()
    val_loss = 0.
    val_error = 0.
    for batch_idx, (data, bag_label, label) in enumerate(validation_loader):
        if args.onlypresent:
            data=data.squeeze(0)
            data=[[dat[0][0]] for dat in data if dat[0][2]==1]
            data=torch.tensor(data)

        if len(data)>0:
            data=data.type(torch.FloatTensor)

            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()


            data, bag_label = Variable(data), Variable(bag_label)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            # if args.weight_loss:
            #     if bag_label:
            #         weighted_loss=bag_class_weight_val[0]*loss
            #     else:
            #         weighted_loss=bag_class_weight_val[1]*loss
            #     val_loss += weighted_loss.data[0]

            # else:
            #     val_loss += loss.data[0]
            if args.model!="set_transformer":
                val_loss += loss.data[0]
            else:
                val_loss += loss

            error, predicted_label = model.calculate_classification_error(data, bag_label)
            val_error += error
        else:
            pass

    val_error /= len(validation_loader)
    val_loss /= len(validation_loader)

    return val_loss



def test(PATH):

    #Using checkpoint to evaluate the model
    # if args.model=='attention':
    #     model = Attention()
    # elif args.model=='gated_attention':
    #     model = GatedAttention()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
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

    # if args.cuda:
    #     model.cuda()

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
    attention_array_true_list=[]
    max_attention_list=[]

    identifier_list=[]
    identifier_true_list=[]


    for batch_idx, (data, bag_label,label) in enumerate(test_loader):
        if args.onlypresent:
            data=data.squeeze(0)
            present_list=[i for i, dat in enumerate(data) if dat[0][2]==1]
            label=[label[0][i] for i in present_list]

            data=[[dat[0][0]] for dat in data if dat[0][2]==1]
            
            label=torch.tensor(label)
            data=torch.tensor(data)

        if len(data)>0:

            # bag_label = label[0]
            instance_labels = label
            data=data.type(torch.FloatTensor)
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            loss, attention_weights = model.calculate_objective(data, bag_label)

            if args.model!="set_transformer":
                test_loss += loss.data[0]
                y_prob,_,_ = model.forward(data)

            else:
                test_loss += loss
                y_prob,_ = model.forward(data)
                
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

            y_prob=torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)

            true_label_list.append(bag_label.cpu().data.numpy())
            pred_label_list.append(predicted_label.cpu().data.numpy())
            if args.model!="set_transformer":
                y_prob_list.append(y_prob.cpu().data.numpy()[0])
            else:
                y_prob_list.append(y_prob.cpu().data)
            
            if args.model!="set_transformer":
                if predicted_label.cpu().data.numpy()[0][0]==1:
                    attention_array=attention_weights.cpu().data.numpy()[0]
                    #counting for calculating probability of max weight if true label
                    max_value=max(attention_array)
                    max_attention= [i for i, j in enumerate(attention_array) if j == max_value]
                    total_count+=1
                    if args.onlypresent:
                        single_labels=instance_labels.numpy().tolist()
                    else:
                        single_labels=instance_labels.numpy()[0].tolist()
                    max_attention_list.append(max_attention)

                    if single_labels[max_attention[0]]:
                        rightattention_count+=1 
                        
                    attention_array_true_list.append(attention_array)
                    # single_labels_true_list.append(single_labels)
                    if args.onlypresent:
                        identifier_true_list.append(data.cpu().data.numpy().tolist())


                #prepare list for instance level ROC
                attention_array_list.append(attention_weights.cpu().data.numpy()[0].tolist())
                if args.onlypresent:
                    single_labels_list.append(instance_labels.numpy().tolist())
                else:
                    single_labels_list.append(instance_labels.numpy()[0].tolist())
                identifier_list.append(data.cpu().data.numpy().tolist())


                if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                    bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
                    if args.onlypresent:
                        print(instance_labels.cpu().data.tolist())
                        instance_level = list(zip(instance_labels.numpy().tolist(),
                                            np.round(attention_weights.cpu().data.numpy(), decimals=3).tolist()))
                    else:
                        instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                            np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

                    print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                        'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))
        else:
            pass
            
           

    if args.model!="set_transformer" and args.onlypresent==False and len(max_attention_list)>0:
        max_pos_df=pd.DataFrame(np.concatenate(max_attention_list)).groupby([0]).size().sort_values(ascending=False)
        print("------------------------------------------------------------------------")
        print("Max Attention Position Statistics", max_pos_df)


    if args.model!="set_transformer" and args.onlypresent==False:    
        if len(attention_array_true_list)>0:
            attention_true=pd.DataFrame(attention_array_true_list)
            print("------------------------------------------------------------------------")
            print("The averaging attention weight by position predicted TRUE bags", pd.DataFrame(attention_true.mean(axis=0)).sort_values(by=[0], ascending=False).head(15))

        attention_df=pd.DataFrame(attention_array_list)
        print("------------------------------------------------------------------------")
        print("The averaging attention weight by position ALL bags", pd.DataFrame(attention_df.mean(axis=0)).sort_values(by=[0], ascending=False).head(15))
        print("------------------------------------------------------------------------")


        SAVING_INSTANCE_PATH=os.getcwd()+"/instance_level_results_lr{}_{}_/{}/".format(args.lr, args.model, args.seed)
        avg_prediciton_truebag_label=[]
        avg_prediciton_allbag_label=[]
        if len(attention_array_true_list)>0:
            avg_predict_true_bags=pd.DataFrame(attention_true.mean(axis=0)).sort_values(by=[0], ascending=False)
        avg_predict_all_bags=pd.DataFrame(attention_df.mean(axis=0)).sort_values(by=[0], ascending=False)

        for i in range(avg_predict_all_bags.shape[0]):
            if len(attention_array_true_list)>0:
                if avg_predict_true_bags.index[i] in target_mutation:
                    avg_prediciton_truebag_label.append(True)
                else:
                    avg_prediciton_truebag_label.append(False)

            if avg_predict_all_bags.index[i] in target_mutation:
                avg_prediciton_allbag_label.append(True)
            else:
                avg_prediciton_allbag_label.append(False)



        os.makedirs(SAVING_INSTANCE_PATH,exist_ok=True)
        if len(attention_array_true_list)>0:
            avg_predict_true_bags['label']=avg_prediciton_truebag_label
            save_file(SAVING_INSTANCE_PATH+"/truebags_nsnp{}_max{}_csnp{}_i{}_prevalence{}.pkl".format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence),avg_predict_true_bags)
        
        avg_predict_all_bags['label']=avg_prediciton_allbag_label
        save_file(SAVING_INSTANCE_PATH+"/allbags_nsnp{}_max{}_csnp{}_i{}_prevalence{}.pkl".format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence),avg_predict_all_bags)


        if total_count==0:
            print('The estimated probability of the right largest attention is',rightattention_count/(total_count+0.000000000001))
        else:
            print('The estimated probability of the right largest attention is',rightattention_count/total_count)

    if args.onlypresent==1:

        SAVING_INSTANCE_PATH=os.getcwd()+"/instance_level_results_lr{}_{}_onlypresent{}/{}/".format(args.lr, args.model,args.onlypresent, args.seed)
        os.makedirs(SAVING_INSTANCE_PATH,exist_ok=True)

        predict_all_bags=pd.DataFrame(np.concatenate(attention_array_list))
        predict_all_bags['identifier']=np.concatenate(identifier_list)
        predict_all_bags_mean=predict_all_bags.groupby(['identifier']).mean()
        identifier_labels=[]
        for i in range(predict_all_bags_mean.shape[0]):
            if i in target_mutation:
                label=True
            else:
                label=False
            identifier_labels.append(label)
        predict_all_bags_mean['labels']=identifier_labels

        if len(attention_array_true_list):
            predict_true_bags=pd.DataFrame(np.concatenate(attention_array_true_list))
            predict_true_bags['identifier']=np.concatenate(identifier_true_list)
            predict_true_bags_mean=predict_true_bags.groupby(['identifier']).mean()
            identifier_true_labels=[]
            for j in range(predict_true_bags_mean.shape[0]):
                if j in target_mutation:
                    label=True
                else:
                    label=False
                identifier_true_labels.append(label)

            predict_true_bags_mean['label']=identifier_true_labels
            save_file(SAVING_INSTANCE_PATH+"/truebags_nsnp{}_max{}_csnp{}_i{}_prevalence{}.pkl".format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence),predict_true_bags_mean)
        save_file(SAVING_INSTANCE_PATH+"/allbags_nsnp{}_max{}_csnp{}_i{}_prevalence{}.pkl".format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence),predict_all_bags_mean)


    print('confusion matrix:',confusion_matrix(np.concatenate(true_label_list), np.concatenate(pred_label_list)))
    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    if args.model!="set_transformer":
        print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))
    else:
        print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu(), test_error))

    # Matrics and plots bag level
    fpr, tpr, threshold_roc=roc_curve(np.concatenate(true_label_list), np.concatenate(y_prob_list))
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds_prc = precision_recall_curve(np.concatenate(true_label_list), np.concatenate(y_prob_list))
    prc_avg = average_precision_score(np.concatenate(true_label_list),np.concatenate(y_prob_list))

    if args.model!="set_transformer" and args.onlypresent==False:
        # Matrics and plots instance level
        instance_level_score=np.concatenate(attention_array_list)
        instance_level_truth=np.concatenate(single_labels_list)

        precision_instance, recall_instance, thresholds_prc_instance = precision_recall_curve(instance_level_truth, instance_level_score)
        prc_avg_instance = average_precision_score(instance_level_truth, instance_level_score)

        fpr_instance, tpr_instance, threshold_roc_instance=roc_curve(instance_level_truth, instance_level_score)
        roc_auc_instance = auc(fpr_instance, tpr_instance)


        # saving evaluation scores
        evaluation_dict={}
        evaluation_dict['fpr_instance']=fpr_instance
        evaluation_dict['tpr_instance']=tpr_instance
        evaluation_dict['roc_auc_instance']=roc_auc_instance
        evaluation_dict['precision_instance']=precision_instance
        evaluation_dict['recall_instance']=recall_instance
        evaluation_dict['prc_avg_instance']=prc_avg_instance

        evaluation_dict['fpr_bag']=fpr
        evaluation_dict['tpr_bag']=tpr
        evaluation_dict['roc_auc_bag']=roc_auc
        evaluation_dict['precision_bag']=precision
        evaluation_dict['recall_bag']=recall
        evaluation_dict['prc_avg_bag']=prc_avg


    else:
        # saving evaluation scores
        evaluation_dict={}
        evaluation_dict['fpr_bag']=fpr
        evaluation_dict['tpr_bag']=tpr
        evaluation_dict['roc_auc_bag']=roc_auc
        evaluation_dict['precision_bag']=precision
        evaluation_dict['recall_bag']=recall
        evaluation_dict['prc_avg_bag']=prc_avg
        
   
    if args.model!="set_transformer" and args.onlypresent==False:
        figure, axis = plt.subplots(2, 2, figsize=(7, 7))
        figure.suptitle('nsnp{}_max{}_csnp{}_i{}_prevalence{}'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence), fontsize=16)

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
        axis[0, 1].axhline(y=0.35, color='grey', linestyle='dotted')

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

    else:
        figure, axis = plt.subplots(1, 2, figsize=(7, 7))
        figure.suptitle('nsnp{}_max{}_csnp{}_i{}_prevalence{}'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence), fontsize=16)

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



    # plt.show()
    SAVING_PATH=os.getcwd()+'/plots_bedreader_leakyrelu_reduceplateu_lr{}_twostep_MLP_upsampling_attweight_{}_fixedSNPtype_onlypresent{}_withattention_all_from0/'.format(args.lr,args.model,args.onlypresent)+ str(args.seed)
    os.makedirs(SAVING_PATH, exist_ok=True)

    EVALUATION_SAVINGPATH=os.getcwd()+'/metrics_bedreader_leakyrelu_reduceplateu_lr{}_twostep_MLP_upsampling_attweight_{}_fixedSNPtype_onlypresent{}_withattention_all_from0/'.format(args.lr,args.model,args.onlypresent)+ str(args.seed)
    os.makedirs(EVALUATION_SAVINGPATH, exist_ok=True)

    if args.prevalence:
        PLOT_PATH=SAVING_PATH+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}.png'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence)
        EVALUATION_PATH=EVALUATION_SAVINGPATH+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}.pkl'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence)
        
    else:
        PLOT_PATH=SAVING_PATH+'/nsnp{}_max{}_csnp{}_i{}.png'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction)
        EVALUATION_PATH=EVALUATION_SAVINGPATH+'/nsnp{}_max{}_csnp{}_i{}.pkl'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction)
    plt.savefig(PLOT_PATH)

    save_file(EVALUATION_PATH,evaluation_dict)



#early stopping criteria
n_epochs_stop = 200
if __name__ == "__main__":
    print('Start Training')
    print('training weight:', bag_class_weight_train)
    working_dir=os.getcwd() 
    PATH=working_dir+'/checkpoints_bedreader_leakyrelu_reduceplateu_lr{}_twostep_MLP_upsampling_attweight_{}_fixedSNPtype_onlypresent{}_withattention_all_from0/'.format(args.lr,args.model,args.onlypresent)+ str(args.seed)

    os.makedirs(PATH, exist_ok=True)
    if args.control_prevalence:
        PATH_SAVE=PATH+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}.pt'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence)

    else:
        PATH_SAVE=PATH+'/nsnp{}_max{}_csnp{}_i{}.pt'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction)
    

    min_loss=np.inf
    if args.num_snp<=100:
        #squeuience was 12
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',patience=10, min_lr=0.00001,factor=0.5,verbose=True)

    elif 100<args.num_snp<=2000:  
        # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',patience=10, min_lr=0.000005,factor=0.8,verbose=True)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',patience=8, min_lr=0.00001,factor=0.7,verbose=True)

    elif 500<args.num_snp<=2000:  
        #patience was 10
        # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',patience=8, min_lr=0.00001,factor=0.8,verbose=True)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',patience=8, min_lr=0.00001,factor=0.7,verbose=True)

    for epoch in range(1, args.epochs + 1):
        train_loss=train(epoch,bag_class_weight_train,weight=True)
        val_loss=val()
        print("validation loss:", val_loss)
        
        if args.num_snp<=100:
            scheduler.step(val_loss)
        elif 100<args.num_snp<=2000:  
            scheduler.step(val_loss)

        os.makedirs("./tensorboard_logs_bedreader_leakyrelu_reduceplateu_lr{}_twostep_MLP_upsampling_attweight_{}_fixedSNPtype_onlypresent{}_withattention_all_from0/".format(args.lr,args.model,args.onlypresent)+ str(args.seed), exist_ok=True)
        writer = SummaryWriter('./tensorboard_logs_bedreader_leakyrelu_reduceplateu_lr{}_twostep_MLP_upsampling_attweight_{}_fixedSNPtype_onlypresent{}_withattention_all_from0/'.format(args.lr,args.model,args.onlypresent)+ str(args.seed)+'/nsnp{}_max{}_csnp{}_i{}_prevalence{}'.format(args.num_snp,args.max_present,args.num_casual_snp,args.interaction,args.prevalence))

        writer.add_scalar('training loss',
                            train_loss/ epoch, epoch)
        writer.add_scalar('validation loss',
                            val_loss/ epoch, epoch)
  
        # for saving best model and check early stoping criteria
        if val_loss< min_loss:
            min_loss=val_loss
            epoch_min=epoch
            epochs_no_improve = 0

            model_state=model.state_dict()
            optimizer_state=optimizer.state_dict()

        else:
            epochs_no_improve += 1

        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            break
        else:
            continue

            

    #save checkpoint of the model
    torch.save({'epoch':epoch_min, 'model_state_dict': model_state,'optimizer_state_dict': optimizer_state,'loss':min_loss}, PATH_SAVE)

    print('Start Testing')
    print('training weight:', bag_class_weight_test)
    test(PATH)


