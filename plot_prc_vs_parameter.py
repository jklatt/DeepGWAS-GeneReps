import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

    

def read_result_byseed(seeds, criteria1, criteria2, criteria3, get_avg_dic, variating_parameter):
    # extracting max_present dictionary
    evaluation_scores_true={}
    evaluation_scores_false={}

    for seed in seeds:
        PATH="/home/zixshu/DeepGWAS/metrics_bedreader_testlr/"+str(seed)
        filenames=os.listdir(PATH)

    #variating max_present
        for file in filenames:
            splited_name=file.split('_')
            train_parameter=splited_name[1]
        
            if((criteria1 in file ) and (criteria2 in file) and (criteria3 in file)):
                if seed==1:
                    if 'iTrue' in file:
                        evaluation_scores_true[train_parameter]={}
                        evaluation_scores_true[train_parameter]['roc_auc']=[]
                        evaluation_scores_true[train_parameter]['prc_avg']=[]



                    else:
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





def ploting_seed_avg(criteria1, criteria2, criteria3, variating_parameter, evaluation_scores_true_avg, interaction):

    if 'snp' in list(evaluation_scores_true_avg.keys())[0]:
                split_element=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('p')]
        
    else:
        split_element=list(list(evaluation_scores_true_avg.keys())[0])[list(list(evaluation_scores_true_avg.keys())[0]).index('0')-1]

    parameters=sorted(list(evaluation_scores_true_avg.keys()), key=lambda x: float(x.split(split_element)[-1]))

    values_arr=[]
    sd_arr=[]
    for parameter in parameters:
        values_arr.append(evaluation_scores_true_avg[parameter]['roc_auc_mean'])
        sd_arr.append(evaluation_scores_true_avg[parameter]['roc_auc_sd'])

    x = np.array(range(len(parameters)))

    #remove the pkl ending for the string
    if "pkl" in criteria3:
            criteria3=criteria3[0:14]

    #ploting the parameter vs score plot
    plt.figure()
    plt.xticks(x, parameters)
    plt.plot(x, values_arr, "-^",color="blue")
    plt.errorbar(x, values_arr, yerr=sd_arr,elinewidth=5)
    plt.xlim([0,max(x)])
    plt.ylim([0,1.1])

    if interaction=="True":
        plt.title("{}_{}_{}_variating{}_iTrue".format(criteria1,criteria2,criteria3, variating_parameter))
        plt.savefig("/home/zixshu/DeepGWAS/plots_bedreader_testlr/{}_{}_{}_variating{}_iTrue.png".format(criteria1,criteria2,criteria3, variating_parameter))

    else:
        plt.title("{}_{}_{}_variating{}_iFalse".format(criteria1,criteria2,criteria3, variating_parameter))
        plt.savefig("/home/zixshu/DeepGWAS/plots_bedreader_testlr/{}_{}_{}_variating{}_iFalse.png".format(criteria1,criteria2,criteria3, variating_parameter))



seeds=list(range(1,6))
criteria1="nsnp20"
criteria2="csnp3"
criteria3="prevalence0.35.pkl"
variating_parameter="max_present"
evaluation_scores_true_avg, evaluation_scores_false_avg= read_result_byseed(seeds, criteria1, criteria2, criteria3, get_avg_dic, variating_parameter)

ploting_seed_avg(criteria1, criteria2, criteria3, variating_parameter,evaluation_scores_true_avg,"True")
ploting_seed_avg(criteria1, criteria2, criteria3, variating_parameter,evaluation_scores_false_avg, "False")




