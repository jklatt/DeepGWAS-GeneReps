from turtle import position
from weakref import ref
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


def read_result_byseed(seeds, criteria1, criteria2, criteria3, get_avg_dic, variating_parameter, path):
    # extracting max_present dictionary
    evaluation_scores_true={}
    evaluation_scores_false={}

    for seed in seeds:
        PATH=path+str(seed)
        filenames=os.listdir(PATH)

    #variating max_present
        for file in filenames:
            
            if variating_parameter=="prevalence":
                splited_name=file.split('_')
                train_parameter=splited_name[4].split('.p')[0]

            if variating_parameter=="csnp":
                splited_name=file.split('_')
                train_parameter=splited_name[2]
               
            if variating_parameter=="max_present":
                splited_name=file.split('_')
                train_parameter=splited_name[1]



        
            if((criteria1 in file ) and (criteria2 in file) and (criteria3 in file)):
                
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


                if 'iTrue' in file:
                    evaluation_scores_true[train_parameter]['roc_auc'].append(evaluation_dict['roc_auc_bag'])
                    evaluation_scores_true[train_parameter]['prc_avg'].append(evaluation_dict['prc_avg_bag'])

                else:
                    evaluation_scores_false[train_parameter]['roc_auc'].append(evaluation_dict['roc_auc_bag'])
                    evaluation_scores_false[train_parameter]['prc_avg'].append(evaluation_dict['prc_avg_bag'])


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


seeds=range(1,3)
selected_seed=1
#snp20
# criteria1_base="nsnp20"
# criteria2_base="csnp3"
# criteria3_base="max1.0"
# criteria4_base="prevalence0.35"

#snp200
criteria1_base="nsnp200"
criteria2_base="csnp3"
criteria3_base="max0.8"
criteria4_base="prevalence0.35"


path="/home/zixshu/DeepGWAS/metrics_bedreader_leakyrelu_reduceplateu_lr0.0004_twostep_MLP_upsampling/"

reference_setting={}
reference_setting["interaction_true"]={}
reference_setting["interaction_false"]={}

# reference setting 
seedfilepath=path+str(selected_seed)
filelist=os.listdir(seedfilepath)
for file in filelist:
    if criteria1_base in file and criteria2_base in file and criteria3_base in file and criteria4_base in file:
        FILE_PATH=seedfilepath+"/"+file
        with open(FILE_PATH, "rb") as f:
                standard_seting=pickle.load(f)

        if "True" in file:
            reference_setting["interaction_true"]['precision']=standard_seting['precision_bag']
            reference_setting["interaction_true"]['recall']=standard_seting['recall_bag']
            reference_setting["interaction_true"]['prc_avg']=standard_seting['prc_avg_bag']

        else:
            reference_setting["interaction_false"]['precision']=standard_seting['precision_bag']
            reference_setting["interaction_false"]['recall']=standard_seting['recall_bag']
            reference_setting["interaction_false"]['prc_avg']=standard_seting['prc_avg_bag']



def ploting_outputs(criteria1_base,criteria2_base, criteria3_base,criteria4_base, criteriasnp1, criteriasnp2, criteriasnp3, criteriamax1, criteriamax2, criteriamax3, criteriapre1, criteriapre2, criteriapre3, variating_parameters, path, get_avg_dic, reference_setting,savingpath):

   
    criteriasnp3=criteriasnp3[0:14]
    criteriamax3=criteriamax3[0:14]

    figure, axis = plt.subplots(2, 2, figsize=(7, 7))

    for par in list(reference_setting.keys()):
        if "true" in par:
            axis[0,0].plot(reference_setting[par]['recall'],reference_setting[par]['precision'],label="w interaction AUC="+str(round(reference_setting[par]['prc_avg'],3)))
        else:
            axis[0,0].plot(reference_setting[par]['recall'],reference_setting[par]['precision'],label="w/o interaction AUC="+str(round(reference_setting[par]['prc_avg'],3)))
    axis[0,0].set_title("{}_{}_{}_{}".format(criteria1_base,criteria2_base,criteria3_base,criteria4_base))
    axis[0,0].legend(prop={'size': 8})
    
    for parameter in variating_parameters:
        if parameter=="csnp":
            evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteriasnp1, criteriasnp2, criteriasnp3, get_avg_dic, parameter, path)

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
            axis[0,1].set_xticks(x, parameters_true)
            axis[0,1].plot(x, values_arr_roc_true, "-^",color="blue")
            axis[0,1].plot(x, values_arr_roc_false, "-^",color="orange")
            axis[0,1].errorbar(x, values_arr_roc_true, yerr=sd_arr_roc_true,elinewidth=5,label="w interaction")
            axis[0,1].errorbar(x, values_arr_roc_false, yerr=sd_arr_roc_false,elinewidth=5,label="w/o interaction")
            axis[0,1].set_xlim([0,max(x)])
            axis[0,1].set_ylim([0,1])
            axis[0,1].set_ylabel("AUROC")
            axis[0,1].set_title("{}{}_{}".format(criteriasnp1, criteriasnp2, criteriasnp3))
            axis[0,1].legend(prop={'size': 8},loc = 'lower left')
            

        if parameter=="max_present":
            evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteriamax1, criteriamax2, criteriamax3, get_avg_dic, parameter,path)

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
            axis[1,0].set_xticks(x, parameters_true)
            axis[1,0].plot(x, values_arr_roc_true, "-^",color="blue")
            axis[1,0].plot(x, values_arr_roc_false, "-^",color="orange")
            axis[1,0].errorbar(x, values_arr_roc_false, yerr=sd_arr_roc_false,elinewidth=5,label="w interaction")
            axis[1,0].errorbar(x, values_arr_roc_true, yerr=sd_arr_roc_true,elinewidth=5,label="w/o interaction")
            axis[1,0].set_xlim([0,max(x)])
            axis[1,0].set_ylim([0,1])
            axis[1,0].set_ylabel("AUROC")
            axis[1,0].set_title("{}{}_{}".format(criteriamax1, criteriamax2, criteriamax3))
            axis[1,0].legend(prop={'size': 8},loc = 'lower left')


        if parameter=="prevalence":
            evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteriapre1, criteriapre2, criteriapre3,get_avg_dic,variating_parameter=parameter, path=path)

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
            axis[1,1].set_xticks(x, parameters_true)
            axis[1,1].plot(x, values_arr_prc_true, "-^",color="blue")
            axis[1,1].plot(x, values_arr_prc_false, "-^",color="orange")
            axis[1,1].errorbar(x, values_arr_prc_true, yerr=sd_arr_prc_true,elinewidth=5,label="w interaction" )
            axis[1,1].errorbar(x, values_arr_prc_false, yerr=sd_arr_prc_false,elinewidth=5,label="w/o interaction")
            axis[1,1].set_xlim([0,max(x)])
            axis[1,1].set_ylim([0,5])
            axis[1,1].set_ylabel("AUPRC")
            axis[1,1].set_title("{}{}_{}".format(criteriapre1, criteriapre2, criteriapre3))
            axis[1,1].legend(prop={'size': 8})

    plt.tight_layout()
    if criteriasnp1=="nsnp200":
        plt.savefig(savingpath+"{}_variatingparameter.png".format(criteriasnp1[:-1]))
    else:
        plt.savefig(savingpath+"{}_variatingparameter.png".format(criteriasnp1))



        



    # if variating_parameter=="prevalence":
    #     axis[1,1].set_xticks(x, parameters)
    #     axis[1].plot(x, values_arr_prc, "-^",color="orange")
    #     axis[1].errorbar(x, values_arr_prc, yerr=sd_arr_prc,elinewidth=5)
    #     axis[1].set_xlim([0,max(x)])
    #     axis[1].set_ylim([0,5])
    #     axis[1].set_ylabel("AUPRC")

    # else:
    #     axis[1].set_xticks(x, parameters)
    #     axis[1].plot(x, values_arr_prc, "-^",color="orange")
    #     axis[1].errorbar(x, values_arr_prc, yerr=sd_arr_prc,elinewidth=5)
    #     axis[1].set_xlim([0,max(x)])
    #     axis[1].set_ylim([0,1])
    #     axis[1].set_ylabel("AUPRC")


variating_parameters=["csnp","prevalence","max_present"]
#nsnp20 setting
# criteriasnp1="nsnp20_"
# criteriasnp2="max1.0"
# criteriasnp3="prevalence0.35.pkl"
# criteriamax1="nsnp20_"
# criteriamax2="csnp3"
# criteriamax3="prevalence0.35.pkl"
# criteriapre1="nsnp20_"
# criteriapre2="csnp3" 
# criteriapre3="max1.0"


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



saving_path="/home/zixshu/DeepGWAS/plot_multiprc_vsparameter_lr0.0004/"
os.makedirs(saving_path,exist_ok=True)

ploting_outputs(criteria1_base,criteria2_base, criteria3_base,criteria4_base,criteriasnp1, criteriasnp2, criteriasnp3, criteriamax1, criteriamax2, criteriamax3, criteriapre1, criteriapre2, criteriapre3, variating_parameters, path, get_avg_dic, reference_setting,saving_path)








            


