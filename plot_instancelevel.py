from fileinput import filename
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

seeds=range(1,6)
ploting_snp=200
criteria_bag="allbags_"
# criteria_bag="truebags"

attention_mec="gated_attention_onlypresent"
if attention_mec=="attention":
    # #attention
    path="/home/zixshu/DeepGWAS/instance_level_results_lr0.0005_attention_/"
    saving_path="/home/zixshu/DeepGWAS/plot_instances_level_attention_onlypresentFalse/"

elif attention_mec=="gated_attention":
    path="/home/zixshu/DeepGWAS/instance_level_results_lr0.0005_gated_attention_/"
    saving_path="/home/zixshu/DeepGWAS/plot_instances_level_gated_attention_onlypresentFalse/"


elif attention_mec=="attention_onlypresent":
    path="/home/zixshu/DeepGWAS/instance_level_results_lr0.0005_attention_onlypresent_onlypresentTrue/"
    saving_path="/home/zixshu/DeepGWAS/plot_instances_level_attention_onlypresentTrue/"
elif attention_mec=="gated_attention_onlypresent":
    path="/home/zixshu/DeepGWAS/instance_level_results_lr0.0005_gated_attention_onlypresent_onlypresentTrue/"
    saving_path="/home/zixshu/DeepGWAS/plot_instances_level_gated_attention_onlypresentTrue/"
#gated_attention
# path="/home/zixshu/DeepGWAS/instance_level_results_lr0.0005_gated_attention_/"
# saving_path="/home/zixshu/DeepGWAS/plot_gated_attention_instances_level/"

seeds=range(1,6)
selected_seed=range(1,6)
if ploting_snp==20:
    #snp20
    criteria1_base="nsnp20"
    criteria2_base="csnp3"
    criteria3_base="max1.0"
    criteria4_base="prevalence0.35"
elif ploting_snp==200:
    #snp200
    criteria1_base="nsnp200"
    criteria2_base="csnp3"
    criteria3_base="max0.8"
    criteria4_base="prevalence0.35"

def read_result_byseed(seeds, criteria1, criteria2, criteria3, criteria4, get_avg_dic, variating_parameter, path, attention_mec):
    # extracting max_present dictionary
    evaluation_scores_true={}
    evaluation_scores_false={}

    for seed in seeds:
        PATH=path+str(seed)
        filenames=os.listdir(PATH)
        #variating max_present
        for file in filenames:
            if "prevalence" in file:
                if variating_parameter=="prevalence":
                    splited_name=file.split('_')
                    train_parameter=splited_name[5].split('.p')[0]

                if variating_parameter=="csnp":
                    splited_name=file.split('_')
                    train_parameter=splited_name[3]#.split('p')[0]+"p"
                
                if variating_parameter=="max_present":
                    splited_name=file.split('_')
                    train_parameter=splited_name[2]#.split('x')[0]+"x"
            
                if((criteria1 in file) and (criteria2 in file) and (criteria3 in file) and (criteria4 in file)):
                    
                    if ('iTrue' in file) and train_parameter not in list(evaluation_scores_true.keys()) :
                        evaluation_scores_true[train_parameter]={}
                        evaluation_scores_true[train_parameter]['roc_auc']=[]
                        evaluation_scores_true[train_parameter]['prc_avg']=[]



                    elif ("iFalse" in file) and train_parameter not in list(evaluation_scores_false.keys()):
                        evaluation_scores_false[train_parameter]={}
                        evaluation_scores_false[train_parameter]['roc_auc']=[]
                        evaluation_scores_false[train_parameter]['prc_avg']=[]


                    FILE_PATH=PATH+'/'+file
                    with open(FILE_PATH, "rb") as f:
                        evaluation_dict=pickle.load(f)
                    attention_weights=evaluation_dict[0]
                    if "onlypresent" in attention_mec:
                        labels=evaluation_dict['labels']
                    else:
                        labels=evaluation_dict['labels']

                    # Matrics and plots bag level
                    fpr, tpr, threshold_roc=roc_curve(labels,attention_weights)
                    roc_auc = auc(fpr, tpr)

                    precision, recall, thresholds_prc = precision_recall_curve(labels, attention_weights)
                    prc_avg = average_precision_score(labels,attention_weights)

                    if 'iTrue' in file:
                        evaluation_scores_true[train_parameter]['roc_auc'].append(roc_auc)
                        evaluation_scores_true[train_parameter]['prc_avg'].append(prc_avg)

                    else:
                        evaluation_scores_false[train_parameter]['roc_auc'].append(roc_auc)
                        evaluation_scores_false[train_parameter]['prc_avg'].append(prc_avg)


    evaluation_scores_true_avg = get_avg_dic(evaluation_scores_true)
    evaluation_scores_false_avg = get_avg_dic(evaluation_scores_false)

    if variating_parameter=="prevalence":
        true_prevalence=list(evaluation_scores_true_avg.keys())
        false_prevalence=list(evaluation_scores_false_avg.keys())
        for t in range(len(true_prevalence)):
            evaluation_scores_true_avg[true_prevalence[t]]['prc_auc_mean']=evaluation_scores_true_avg[true_prevalence[t]]['prc_auc_mean']/float(true_prevalence[t][10:])
            evaluation_scores_false_avg[false_prevalence[t]]['prc_auc_mean']=evaluation_scores_false_avg[false_prevalence[t]]['prc_auc_mean']/float(false_prevalence[t][10:])


    return evaluation_scores_true_avg, evaluation_scores_false_avg 


def get_avg_dic(evaluation_scores_true):
    evaluation_scores_true_avg={}
    keys=evaluation_scores_true.keys()
    for key in keys:
        evaluation_scores_true_avg[key]={}

        evaluation_scores_true_avg[key]['roc_auc_mean']=np.mean(evaluation_scores_true[key]['roc_auc'])
        evaluation_scores_true_avg[key]['roc_auc_sd']=np.std(evaluation_scores_true[key]['roc_auc'])

        evaluation_scores_true_avg[key]['prc_auc_mean']=np.mean(evaluation_scores_true[key]['prc_avg'])
        evaluation_scores_true_avg[key]['prc_auc_sd']=np.std(evaluation_scores_true[key]['prc_avg'])
    return evaluation_scores_true_avg





reference_setting={}
calculating_avg={}
for seed in selected_seed:
    reference_setting[seed]={}
    reference_setting[seed]["interaction_true"]={}
    reference_setting[seed]["interaction_false"]={}
    calculating_avg[seed]={}
    calculating_avg[seed]["interaction_true"]={}
    calculating_avg[seed]["interaction_false"]={}


def forward_filling(new_list1, list1, list2):
    filled_list1=[[]]*len(new_list1)
    compared_ind=0
    # filled_list1[0]=list2[0]
    for i in range(0,len(new_list1)):
        greater_list=list1>new_list1[i]
        greater_ind=[i for i, x in enumerate(greater_list) if x]
        if len(greater_ind)>0:
            if greater_ind[0]>=1 :
                filled_list1[i]=list2[greater_ind[0]-1]
            else:
                filled_list1[i]=list2[0]
        else:
            filled_list1[i]=list2[-1]
    return filled_list1


# for seed in selected_seed:
#     # reference setting 
#     seedfilepath=path+str(seed)
#     filelist=os.listdir(seedfilepath)
#     for file in filelist:
#         # if criteria1_base in file and criteria2_base in file and criteria3_base in file and criteria4_base in file:
#             FILE_PATH=seedfilepath+"/"+file
#             with open(FILE_PATH, "rb") as f:
#                     standard_seting=pickle.load(f)
#             # print(standard_seting['label'])
#             # print(standard_seting[0])
#             # print(standard_seting)
#             print(standard_seting[0])
#             print(standard_seting['label'])

#             if "True" in file:
#                 #interporating the line on graph
#                 min_pre,max_pre=standard_seting['precision_bag'][0], standard_seting['precision_bag'][-1]
#                 min_recall,max_recall=standard_seting['recall_bag'][0], standard_seting['recall_bag'][-1]
#                 new_a1_x = np.linspace(min_pre, max_pre, 1000)
#                 new_a2_x = np.linspace(min_recall, max_recall, 1000)
#                 # new_a1_y = np.interp(new_a1_x, standard_seting['precision_bag'], standard_seting['recall_bag'])
#                 # new_a2_y = np.interp(new_a2_x, np.flip(standard_seting['recall_bag']),np.flip(standard_seting['precision_bag']))
                
#                 new_a1_y = forward_filling(new_a1_x, standard_seting['precision_bag'], standard_seting['recall_bag'])
#                 if standard_seting['recall_bag'][0]-standard_seting['recall_bag'][-1]>0:
#                     new_a2_y = forward_filling(np.flip(new_a2_x), np.flip(standard_seting['recall_bag']),np.flip(standard_seting['precision_bag']))
#                     calculating_avg[seed]["interaction_true"]['precision']=np.flip(new_a2_y)
#                 else:
#                     new_a2_y = forward_filling(new_a2_x, standard_seting['recall_bag'], standard_seting['precision_bag'])
#                     calculating_avg[seed]["interaction_true"]['precision']=new_a2_y

#                 calculating_avg[seed]["interaction_true"]['recall']=new_a1_y
#                 reference_setting[seed]["interaction_true"]['precision']=standard_seting['precision_bag']
#                 reference_setting[seed]["interaction_true"]['recall']=standard_seting['recall_bag']
#                 reference_setting[seed]["interaction_true"]['prc_avg']=standard_seting['prc_avg_bag']

#             else:
#                 #interporating the line on graph
#                 min_pre,max_pre=standard_seting['precision_bag'][0], standard_seting['precision_bag'][-1]
#                 min_recall,max_recall=standard_seting['recall_bag'][0], standard_seting['recall_bag'][-1]
#                 new_a1_x = np.linspace(min_pre, max_pre, 1000)
#                 new_a2_x = np.linspace(min_recall, max_recall, 1000)
#                 new_a1_y = forward_filling(new_a1_x, standard_seting['precision_bag'], standard_seting['recall_bag'])

#                 if standard_seting['recall_bag'][0]-standard_seting['recall_bag'][-1]>0:
#                     new_a2_y =forward_filling(np.flip(new_a2_x), np.flip(standard_seting['recall_bag']),np.flip(standard_seting['precision_bag']))
#                     calculating_avg[seed]["interaction_false"]['precision']=np.flip(new_a2_y)
#                 else:
#                     new_a2_y = forward_filling(new_a2_x, standard_seting['recall_bag'], standard_seting['precision_bag'])
#                     calculating_avg[seed]["interaction_false"]['precision']=new_a2_y
                
#                 # new_a2_y = np.interp(new_a2_x, np.flip(standard_seting['recall_bag']),np.flip(standard_seting['precision_bag']))
#                 # calculating_avg[seed]["interaction_false"]['precision']=np.flip(new_a2_y)
#                 calculating_avg[seed]["interaction_false"]['recall']=new_a1_y 
#                 reference_setting[seed]["interaction_false"]['precision']=standard_seting['precision_bag']
#                 reference_setting[seed]["interaction_false"]['recall']=standard_seting['recall_bag']
#                 reference_setting[seed]["interaction_false"]['prc_avg']=standard_seting['prc_avg_bag']
                
# avg_pre_true=[]
# avg_recall_true=[]
# avg_pre_false=[]
# avg_recall_false=[]

# for seed in seeds:
#     if len(calculating_avg[seed]["interaction_true"])>0:
#         avg_pre_true.append(calculating_avg[seed]["interaction_true"]['precision'])
#         avg_recall_true.append(calculating_avg[seed]["interaction_true"]['recall'])

#     if len(calculating_avg[seed]["interaction_false"])>0:
#         avg_pre_false.append(calculating_avg[seed]["interaction_false"]['precision'])
#         avg_recall_false.append(calculating_avg[seed]["interaction_false"]['recall'])

# calculating_avg["interaction_true"]={}
# calculating_avg["interaction_false"]={}
# calculating_avg["interaction_true"]["avg_precision"]=np.mean(np.array(avg_pre_true),axis=0)
# calculating_avg["interaction_true"]["avg_recall"]=np.mean(np.array(avg_recall_true),axis=0)
# calculating_avg["interaction_false"]["avg_precision"]=np.mean(np.array(avg_pre_false),axis=0)
# calculating_avg["interaction_false"]["avg_recall"]=np.mean(np.array(avg_recall_false),axis=0)

# calculating_avg["interaction_true"]["sd_precision"]=np.std(np.array(avg_pre_true),axis=0)
# calculating_avg["interaction_true"]["sd_recall"]=np.std(np.array(avg_recall_true),axis=0)
# calculating_avg["interaction_false"]["sd_precision"]=np.std(np.array(avg_pre_false),axis=0)
# calculating_avg["interaction_false"]["sd_recall"]=np.std(np.array(avg_recall_false),axis=0)



def ploting_outputs(seeds, criteria1_base,criteria2_base, criteria3_base,criteria4_base, criteriasnp1, criteriasnp2, criteriasnp3, criteriamax1, criteriamax2, criteriamax3, criteriapre1, criteriapre2, criteriapre3, variating_parameters, path, get_avg_dic, reference_setting,calculating_avg,savingpath, read_result_byseed,attention_mec):

   
    criteriasnp3=criteriasnp3[0:14]
    criteriamax3=criteriamax3[0:14]

    figure, axis = plt.subplots(2, 2, figsize=(7, 7))
    
    # axis[0,0].plot(calculating_avg['interaction_false']["avg_recall"],calculating_avg["interaction_false"]["avg_precision"],label="w/o interaction avg",color="orange")
    # axis[0,0].plot(calculating_avg['interaction_true']["avg_recall"],calculating_avg["interaction_true"]["avg_precision"],label="w interaction avg",color="blue")
    # axis[0,0].fill_between(x=calculating_avg["interaction_true"]["avg_recall"],
    #  y1=np.add(np.array(calculating_avg["interaction_true"]["avg_precision"]),np.array(calculating_avg["interaction_true"]["sd_precision"])),
    #  y2=np.subtract(np.array(calculating_avg["interaction_true"]["avg_precision"]),np.array(calculating_avg["interaction_true"]["sd_precision"])),alpha=0.2,color="blue")

    # axis[0,0].fill_between(x=calculating_avg["interaction_false"]["avg_recall"],
    #  y1=np.add(np.array(calculating_avg["interaction_false"]["avg_precision"]),np.array(calculating_avg["interaction_false"]["sd_precision"])),
    #  y2=np.subtract(np.array(calculating_avg["interaction_false"]["avg_precision"]),np.array(calculating_avg["interaction_false"]["sd_precision"])),alpha=0.2,color="orange")

    # axis[0,0].set_title("{}_{}_{}_{}".format(criteria1_base,criteria2_base,criteria3_base,criteria4_base))
    # axis[0,0].legend(prop={'size': 5},loc='lower right')
    figure.delaxes(axis[1,1])
    for parameter in variating_parameters:
        if parameter=="csnp":
            evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteriasnp1, criteriasnp2, criteriasnp3, criteria_bag,get_avg_dic, parameter, path, attention_mec)

            split_element_true=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('p')]
            split_element_false=list(list(evaluation_scores_false_avg.keys())[0])[list(list(evaluation_scores_false_avg.keys())[0]).index('p')]
            parameters_true=sorted(list(evaluation_scores_true_avg.keys()), key=lambda x: float(x.split(split_element_true)[-1]))
            parameters_false=sorted(list(evaluation_scores_false_avg.keys()), key=lambda x: float(x.split(split_element_false)[-1]))

            values_arr_roc_true=[]
            sd_arr_roc_true=[]
            values_arr_prc_true=[]
            sd_arr_prc_true=[]
            values_arr_roc_false=[]
            sd_arr_roc_false=[]
            values_arr_prc_false=[]
            sd_arr_prc_false=[]

            for para in parameters_true:
                values_arr_roc_true.append(evaluation_scores_true_avg[para]['roc_auc_mean'])
                sd_arr_roc_true.append(evaluation_scores_true_avg[para]['roc_auc_sd'])
                values_arr_prc_true.append(evaluation_scores_true_avg[para]['prc_auc_mean'])
                sd_arr_prc_true.append(evaluation_scores_true_avg[para]['prc_auc_sd'])

            for para in parameters_false:
                values_arr_roc_false.append(evaluation_scores_false_avg[para]['roc_auc_mean'])
                sd_arr_roc_false.append(evaluation_scores_false_avg[para]['roc_auc_sd'])
                values_arr_prc_false.append(evaluation_scores_false_avg[para]['prc_auc_mean'])
                sd_arr_prc_false.append(evaluation_scores_false_avg[para]['prc_auc_sd'])


            x = np.array(range(len(parameters_true)))
            axis[0,0].set_xticks(x, parameters_true)
            axis[0,0].plot(x, values_arr_prc_true, "-^",color="blue",label="w interaction")#CHANGE
            axis[0,0].plot(x, values_arr_prc_false, "-^",color="orange",label="w/o interaction")#CHANGE

            axis[0,0].fill_between(np.array(x),y1=np.add(np.array(values_arr_prc_true),np.array(sd_arr_prc_true)),y2=np.subtract(np.array(values_arr_prc_true),np.array(sd_arr_prc_true)),alpha=0.2)
            axis[0,0].fill_between(np.array(x),y1=np.add(np.array(values_arr_prc_false),np.array(sd_arr_prc_false)),y2=np.subtract(np.array(values_arr_prc_false),np.array(sd_arr_prc_false)),alpha=0.2)
            axis[0,0].set_xlim([0,max(x)])
            axis[0,0].set_ylim([0,1])
            axis[0,0].set_ylabel("AUPRC")
            axis[0,0].set_title("{}{}_{}".format(criteriasnp1, criteriasnp2, criteriasnp3))
            axis[0,0].legend(prop={'size': 8},loc = 'lower left')
            

        if parameter=="max_present":
            evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteriamax1, criteriamax2, criteriamax3, criteria_bag, get_avg_dic, parameter,path,attention_mec)

            split_element_true=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('0')-1]
            split_element_false=list(list(evaluation_scores_false_avg.keys())[0])[list(list(evaluation_scores_false_avg.keys())[0]).index('0')-1]

            parameters_true=sorted(list(evaluation_scores_true_avg.keys()), key=lambda x: float(x.split(split_element_true)[-1]))
            parameters_false=sorted(list(evaluation_scores_false_avg.keys()), key=lambda x: float(x.split(split_element_false)[-1]))

            values_arr_roc_true=[]
            sd_arr_roc_true=[]
            values_arr_prc_true=[]
            sd_arr_prc_true=[]
            values_arr_roc_false=[]
            sd_arr_roc_false=[]
            values_arr_prc_false=[]
            sd_arr_prc_false=[]

            for para in parameters_true:
                values_arr_roc_true.append(evaluation_scores_true_avg[para]['roc_auc_mean'])
                sd_arr_roc_true.append(evaluation_scores_true_avg[para]['roc_auc_sd'])
                values_arr_prc_true.append(evaluation_scores_true_avg[para]['prc_auc_mean'])
                sd_arr_prc_true.append(evaluation_scores_true_avg[para]['prc_auc_sd'])

            for para in parameters_false:
                values_arr_roc_false.append(evaluation_scores_false_avg[para]['roc_auc_mean'])
                sd_arr_roc_false.append(evaluation_scores_false_avg[para]['roc_auc_sd'])
                values_arr_prc_false.append(evaluation_scores_false_avg[para]['prc_auc_mean'])
                sd_arr_prc_false.append(evaluation_scores_false_avg[para]['prc_auc_sd'])

            x = np.array(range(len(parameters_true)))
            axis[0,1].set_xticks(x, parameters_true)
            axis[0,1].plot(x, values_arr_prc_true, "-^",color="blue",label="w interaction")#CHANGE
            axis[0,1].plot(x, values_arr_prc_false, "-^",color="orange",label="w/o interaction")#CHANGE
            # axis[1,0].errorbar(x, values_arr_prc_false, yerr=sd_arr_prc_false,elinewidth=5)
            # axis[1,0].errorbar(x, values_arr_prc_true, yerr=sd_arr_prc_true,elinewidth=5)

            axis[0,1].fill_between(np.array(x),y1=np.add(np.array(values_arr_prc_true),np.array(sd_arr_prc_true)),y2=np.subtract(np.array(values_arr_prc_true),np.array(sd_arr_prc_true)),alpha=0.2)
            axis[0,1].fill_between(np.array(x),y1=np.add(np.array(values_arr_prc_false),np.array(sd_arr_prc_false)),y2=np.subtract(np.array(values_arr_prc_false),np.array(sd_arr_prc_false)),alpha=0.2)
            
            axis[0,1].set_xlim([0,max(x)])
            axis[0,1].set_ylim([0,1])
            axis[0,1].set_ylabel("AUPRC")
            axis[0,1].set_title("{}{}_{}".format(criteriamax1, criteriamax2, criteriamax3))
            axis[0,1].legend(prop={'size': 8},loc = 'lower left')


        if parameter=="prevalence":
            evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteriapre1, criteriapre2, criteriapre3,criteria_bag, get_avg_dic,variating_parameter=parameter, path=path, attention_mec=attention_mec)

            split_element_false=list(list(evaluation_scores_false_avg.keys())[0])[list(list(evaluation_scores_false_avg.keys())[0]).index('0')-1]
            parameters_false=sorted(list(evaluation_scores_false_avg.keys()), key=lambda x: float(x.split(split_element_false)[-1]))
            
            split_element_true=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('0')-1]
            parameters_true=sorted(list(evaluation_scores_true_avg.keys()), key=lambda x: float(x.split(split_element_true)[-1]))

            values_arr_roc_true=[]
            sd_arr_roc_true=[]
            values_arr_prc_true=[]
            sd_arr_prc_true=[]
            values_arr_roc_false=[]
            sd_arr_roc_false=[]
            values_arr_prc_false=[]
            sd_arr_prc_false=[]

            for para in parameters_true:
                values_arr_roc_true.append(evaluation_scores_true_avg[para]['roc_auc_mean'])
                sd_arr_roc_true.append(evaluation_scores_true_avg[para]['roc_auc_sd'])
                values_arr_prc_true.append(evaluation_scores_true_avg[para]['prc_auc_mean'])
                sd_arr_prc_true.append(evaluation_scores_true_avg[para]['prc_auc_sd'])

            for para in parameters_false:
                values_arr_roc_false.append(evaluation_scores_false_avg[para]['roc_auc_mean'])
                sd_arr_roc_false.append(evaluation_scores_false_avg[para]['roc_auc_sd'])
                values_arr_prc_false.append(evaluation_scores_false_avg[para]['prc_auc_mean'])
                sd_arr_prc_false.append(evaluation_scores_false_avg[para]['prc_auc_sd'])

            x = np.array(range(len(parameters_true)))
            axis[1,0].set_xticks(x, parameters_true)
            axis[1,0].plot(x, values_arr_prc_true, "-^",color="blue",label="w interaction" )
            axis[1,0].plot(x, values_arr_prc_false, "-^",color="orange",label="w/o interaction")
            # axis[1,1].errorbar(x, values_arr_prc_true, yerr=sd_arr_prc_true,elinewidth=5,label="w interaction" )
            # axis[1,1].errorbar(x, values_arr_prc_false, yerr=sd_arr_prc_false,elinewidth=5,label="w/o interaction")
            axis[1,0].fill_between(np.array(x),y1=np.add(np.array(values_arr_prc_true),np.array(sd_arr_prc_true)),y2=np.subtract(np.array(values_arr_prc_true),np.array(sd_arr_prc_true)),alpha=0.2)
            axis[1,0].fill_between(np.array(x),y1=np.add(np.array(values_arr_prc_false),np.array(sd_arr_prc_false)),y2=np.subtract(np.array(values_arr_prc_false),np.array(sd_arr_prc_false)),alpha=0.2)
            
            axis[1,0].set_xlim([0,max(x)])
            axis[1,0].set_ylim([0,5])
            axis[1,0].set_ylabel("AUPRC")
            axis[1,0].set_title("{}{}_{}".format(criteriapre1, criteriapre2, criteriapre3))
            axis[1,0].legend(prop={'size': 8})

    plt.tight_layout()
    if criteriasnp1=="nsnp200":
        plt.savefig(savingpath+"{}_variatingparameter.png".format(criteriasnp1[:-1]))
    else:
        plt.savefig(savingpath+"{}_variatingparameter.png".format(criteriasnp1))


variating_parameters=["csnp","prevalence","max_present"]
if ploting_snp==20:
    #nsnp20 setting
    criteriasnp1="nsnp20_"
    criteriasnp2="max1.0"
    criteriasnp3="prevalence0.35.pkl"
    criteriamax1="nsnp20_"
    criteriamax2="csnp3"
    criteriamax3="prevalence0.35.pkl"
    criteriapre1="nsnp20_"
    criteriapre2="csnp3" 
    criteriapre3="max1.0"

elif ploting_snp==200:
    #nsnp 200 setting
    criteriasnp1="nsnp200_"
    criteriasnp2="max0.8"
    criteriasnp3="prevalence0.35.pkl"
    criteriamax1="nsnp200_"
    criteriamax2="csnp3"
    criteriamax3="prevalence0.35.pkl"
    criteriapre1="nsnp200_"
    criteriapre2="csnp3" 
    criteriapre3="max0.8"



os.makedirs(saving_path,exist_ok=True)

ploting_outputs(seeds, criteria1_base,criteria2_base, criteria3_base,criteria4_base,criteriasnp1, criteriasnp2, criteriasnp3, criteriamax1, criteriamax2, criteriamax3, criteriapre1, criteriapre2, criteriapre3, variating_parameters, path, get_avg_dic, reference_setting,calculating_avg,saving_path,read_result_byseed, attention_mec)

        











