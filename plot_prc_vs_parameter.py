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





def ploting_seed_avg(criteria1, criteria2, criteria3, variating_parameter, evaluation_scores_true_avg, interaction, path):

    if 'snp' in list(evaluation_scores_true_avg.keys())[0]:
        split_element=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('p')]
        
    else:
        split_element=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('0')-1]

    parameters=sorted(list(evaluation_scores_true_avg.keys()), key=lambda x: float(x.split(split_element)[-1]))

    values_arr_roc=[]
    sd_arr_roc=[]
    values_arr_prc=[]
    sd_arr_prc=[]
    for parameter in parameters:
        values_arr_roc.append(evaluation_scores_true_avg[parameter]['roc_auc_mean'])
        sd_arr_roc.append(evaluation_scores_true_avg[parameter]['roc_auc_sd'])
        values_arr_prc.append(evaluation_scores_true_avg[parameter]['prc_auc_mean'])
        sd_arr_prc.append(evaluation_scores_true_avg[parameter]['prc_auc_sd'])



    x = np.array(range(len(parameters)))

    #remove the pkl ending for the string
    if "pkl" in criteria3:
            criteria3=criteria3[0:14]

    #ploting the parameter vs score plot
    figure, axis = plt.subplots(1, 2, figsize=(7, 7))
    axis[0].set_xticks(x, parameters)
    axis[0].plot(x, values_arr_roc, "-^",color="blue")
    axis[0].errorbar(x, values_arr_roc, yerr=sd_arr_roc,elinewidth=5)
    axis[0].set_xlim([0,max(x)])
    axis[0].set_ylim([0,1])
    axis[0].set_ylabel("AUROC")


    if variating_parameter=="prevalence":
        axis[1].set_xticks(x, parameters)
        axis[1].plot(x, values_arr_prc, "-^",color="orange")
        axis[1].errorbar(x, values_arr_prc, yerr=sd_arr_prc,elinewidth=5)
        axis[1].set_xlim([0,max(x)])
        axis[1].set_ylim([0,5])
        axis[1].set_ylabel("AUPRC")

    else:
        axis[1].set_xticks(x, parameters)
        axis[1].plot(x, values_arr_prc, "-^",color="orange")
        axis[1].errorbar(x, values_arr_prc, yerr=sd_arr_prc,elinewidth=5)
        axis[1].set_xlim([0,max(x)])
        axis[1].set_ylim([0,1])
        axis[1].set_ylabel("AUPRC")






    if interaction=="True":
        figure.suptitle("{}_{}_{}_variating{}_iTrue".format(criteria1,criteria2,criteria3, variating_parameter))
        plt.tight_layout()
        plt.savefig(path+"{}_{}_{}_variating{}_iTrue.png".format(criteria1,criteria2,criteria3, variating_parameter))

    else:
        figure.suptitle("{}_{}_{}_variating{}_iFalse".format(criteria1,criteria2,criteria3, variating_parameter))
        plt.tight_layout()
        plt.savefig(path+"{}_{}_{}_variating{}_iFalse.png".format(criteria1,criteria2,criteria3, variating_parameter))



seeds=list(range(1,3))

#max_present snp=20
# criteria1="nsnp20"
# criteria2="csnp3"
# criteria3="prevalence0.35.pkl"
# variating_parameter="max_present"

# #num_csnp snp=20
# criteria1="nsnp20"
# criteria2="max1.0"
# criteria3="prevalence0.35.pkl"
# variating_parameter="csnp"

#prevalence snp=20
# criteria1="nsnp20"
# criteria2="csnp3"
# criteria3="max1.0"
# variating_parameter="prevalence"



#max_present snp=200
# criteria1="nsnp200"
# criteria2="csnp3"
# criteria3="prevalence0.35.pkl"
# variating_parameter="max_present"

#num_csnp snp=200
# criteria1="nsnp200"
# criteria2="max0.8"
# criteria3="prevalence0.35.pkl"
# variating_parameter="csnp"

#prevalence snp=200
criteria1="nsnp200"
criteria2="csnp3"
criteria3="max0.8"
variating_parameter="prevalence"

path="/home/zixshu/DeepGWAS/metrics_bedreader_leakyrelu_reduceplateu_lr0.0009_twostep_MLP_upsampling/"
saving_path="/home/zixshu/DeepGWAS/plot_prc_vs_parameter_leakyrelu_MLP_lr0.0009_factor0.8_upsampling/"
os.makedirs(saving_path,exist_ok=True)
evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteria1, criteria2, criteria3, get_avg_dic, variating_parameter,path)

ploting_seed_avg(criteria1, criteria2, criteria3, variating_parameter,evaluation_scores_true_avg,"True",saving_path)
ploting_seed_avg(criteria1, criteria2, criteria3, variating_parameter,evaluation_scores_false_avg, "False",saving_path)




