from unicodedata import numeric
import matplotlib.pyplot as plt
import pickle
import os
import re
from sklearn.model_selection import ParameterSampler


def extract_dicts(train_parameter:str, file,evaluation_dict, evaluation_scores_true, evaluation_scores_false):
    
    if 'iTrue' in file:
        evaluation_scores_true[train_parameter]['fpr']=evaluation_dict['fpr_bag']
        evaluation_scores_true[train_parameter]['tpr']=evaluation_dict['tpr_bag']
        evaluation_scores_true[train_parameter]['roc_auc']=evaluation_dict['roc_auc_bag']
        evaluation_scores_true[train_parameter]['precision']=evaluation_dict['precision_bag']
        evaluation_scores_true[train_parameter]['recall']=evaluation_dict['recall_bag']
        evaluation_scores_true[train_parameter]['prc_avg']=evaluation_dict['prc_avg_bag']

        evaluation_scores_true[train_parameter]['fpr_instance']=evaluation_dict['fpr_instance']
        evaluation_scores_true[train_parameter]['tpr_instance']=evaluation_dict['tpr_instance']
        evaluation_scores_true[train_parameter]['roc_auc_instance']=evaluation_dict['roc_auc_instance']
        evaluation_scores_true[train_parameter]['precision_instance']=evaluation_dict['precision_instance']
        evaluation_scores_true[train_parameter]['recall_instance']=evaluation_dict['recall_instance']
        evaluation_scores_true[train_parameter]['prc_avg_instance']=evaluation_dict['prc_avg_instance']


    else:
        evaluation_scores_false[train_parameter]['fpr']=evaluation_dict['fpr_bag']
        evaluation_scores_false[train_parameter]['tpr']=evaluation_dict['tpr_bag']
        evaluation_scores_false[train_parameter]['roc_auc']=evaluation_dict['roc_auc_bag']
        evaluation_scores_false[train_parameter]['precision']=evaluation_dict['precision_bag']
        evaluation_scores_false[train_parameter]['recall']=evaluation_dict['recall_bag']
        evaluation_scores_false[train_parameter]['prc_avg']=evaluation_dict['prc_avg_bag']

        evaluation_scores_false[train_parameter]['fpr_instance']=evaluation_dict['fpr_instance']
        evaluation_scores_false[train_parameter]['tpr_instance']=evaluation_dict['tpr_instance']
        evaluation_scores_false[train_parameter]['roc_auc_instance']=evaluation_dict['roc_auc_instance']
        evaluation_scores_false[train_parameter]['precision_instance']=evaluation_dict['precision_instance']
        evaluation_scores_false[train_parameter]['recall_instance']=evaluation_dict['recall_instance']
        evaluation_scores_false[train_parameter]['prc_avg_instance']=evaluation_dict['prc_avg_instance']



    return evaluation_scores_true, evaluation_scores_false


def plot_comparisions(evaluation_scores_true, variating_variable,interaction):
    if 'snp' in variating_variable:
        split_element=list(list(evaluation_scores_true.keys())[0])[list(list(evaluation_scores_true.keys())[0]).index('p')]
    else:
        split_element=list(list(evaluation_scores_true.keys())[0])[list(list(evaluation_scores_true.keys())[0]).index('0')-1]

    parameters=sorted(list(evaluation_scores_true.keys()), key=lambda x: float(x.split(split_element)[-1]))
## bag level roc
    figure, axis = plt.subplots(2, 2, figsize=(7, 7))
    figure.suptitle('nsnp20 csnp3 prevalence0.35 i{} and variating {}'.format(interaction,variating_variable))

    for parameter in parameters:
        axis[0, 0].plot(evaluation_scores_true[parameter]['fpr'],evaluation_scores_true[parameter]['tpr'],label="{} AUC=".format(parameter)+str(round(evaluation_scores_true[parameter]['roc_auc'],3)))
    axis[0, 0].legend()
    axis[0, 0].set_title('Bag level ROC')
    axis[0, 0].set_ylabel('True Positive Rate')
    axis[0, 0].set_xlabel('False Positive Rate')
    axis[0, 0].legend(loc = 'lower right')
    axis[0, 0].set_xlim([0, 1])
    axis[0, 0].set_ylim([0, 1])
    axis[0, 0].plot([0, 1], [0, 1],'r--')

## bag level prc
    for parameter in parameters:
        axis[0, 1].plot(evaluation_scores_true[parameter]['recall'], evaluation_scores_true[parameter]['precision'],label="{} AUC=".format(parameter)+str(round(evaluation_scores_true[parameter]['prc_avg'],3)))
    axis[0, 1].legend()
    axis[0, 1].set_title('Bag level PRC')
    axis[0, 1].set_xlabel('Recall')
    axis[0, 1].set_ylabel('Precision')
    axis[0, 1].legend(loc = 'lower left')
    axis[0, 1].set_xlim([0, 1])
    axis[0, 1].set_ylim([0, 1])


## instance level roc
    for parameter in parameters:
        axis[1, 0].plot(evaluation_scores_true[parameter]['fpr_instance'],evaluation_scores_true[parameter]['tpr_instance'],label="{} AUC=".format(parameter)+str(round(evaluation_scores_true[parameter]['roc_auc_instance'],3)))
    axis[1, 0].legend()
    axis[1, 0].set_title('Instance level ROC')
    axis[1, 0].set_ylabel('True Positive Rate')
    axis[1, 0].set_xlabel('False Positive Rate')
    axis[1, 0].legend(loc = 'lower right')
    axis[1, 0].set_xlim([0, 1])
    axis[1, 0].set_ylim([0, 1])
    axis[1, 0].plot([0, 1], [0, 1],'r--')

# instance level prc
    for parameter in parameters:
        axis[1, 1].plot(evaluation_scores_true[parameter]['recall_instance'], evaluation_scores_true[parameter]['precision_instance'],label="{} AUC=".format(parameter)+str(round((evaluation_scores_true[parameter]['prc_avg_instance']),3)))
    axis[1, 1].legend()
    axis[1, 1].set_title('Instance level PRC')
    axis[1, 1].set_xlabel('Recall')
    axis[1, 1].set_ylabel('Precision')
    axis[1, 1].legend(loc = 'upper right')
    axis[1, 1].set_xlim([0, 1])
    axis[1, 1].set_ylim([0, 1])


    plt.tight_layout()
    os.makedirs('/home/zixshu/DeepGWAS/plots_version2_evaluation', exist_ok=True)
    plt.savefig('/home/zixshu/DeepGWAS/plots_version2_evaluation/var_{}_interaction{}.png'.format(variating_variable, interaction))



PATH='/home/zixshu/DeepGWAS/metrics_version2'
filenames=os.listdir('/home/zixshu/DeepGWAS/metrics_version2')

# extracting max_present dictionary
evaluation_scores_true={}
evaluation_scores_false={}

for file in filenames:
   
    splited_name=file.split('_')
    train_parameter=splited_name[1]
    
    if( ('nsnp20' in file ) and ('csnp3' in file) and ('prevalence0.35.pkl'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'max_present','True')
plot_comparisions(evaluation_scores_false,'max_present','False')




# extracting prevalence dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    
    splited_name=file.split('_')
    train_parameter=splited_name[4].split('.p')[0]

    
    if( ('nsnp20' in file ) and ('csnp3' in file) and ('max0.45'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'prevalence','True')
plot_comparisions(evaluation_scores_false,'prevalence','False')



# extracting number of casual SNP dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[2]

    
    if( ('nsnp20' in file ) and ('prevalence0.35.pkl' in file) and ('max0.45'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'csnp','True')
plot_comparisions(evaluation_scores_false,'csnp','False')



# extracting number of number of SNP dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[0]
    
    if('prevalence0.35.pkl' in file) and ('csnp3' in file) and ('0.3' not in file) and ('0.15' not in file) and ('0.5' not in file) and ('nsnp20_max0.45' not in file):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

# plot_comparisions(evaluation_scores_true,'nsnp','True')
# plot_comparisions(evaluation_scores_false,'nsnp','False')




