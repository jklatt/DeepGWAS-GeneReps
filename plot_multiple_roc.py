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


def plot_comparisions(evaluation_scores_true, variating_variable,interaction, num_snp):
    if 'snp' in variating_variable:
        split_element=list(list(evaluation_scores_true.keys())[0])[list(list(evaluation_scores_true.keys())[0]).index('p')]
    else:
        split_element=list(list(evaluation_scores_true.keys())[0])[list(list(evaluation_scores_true.keys())[0]).index('0')-1]

    parameters=sorted(list(evaluation_scores_true.keys()), key=lambda x: float(x.split(split_element)[-1]))
    ## bag level roc
    figure, axis = plt.subplots(2, 2, figsize=(7, 7))
    figure.suptitle('nsnp{} csnp3 prevalence0.35 i{} and variating {}'.format(num_snp, interaction,variating_variable))

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
    axis[0, 1].axhline(y=0.35, color='grey', linestyle='dotted')


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
    axis[1, 1].axhline(y=0.35, color='grey', linestyle='dotted')


    plt.tight_layout()
    os.makedirs('/home/zixshu/DeepGWAS/plots_bedreader_relu_lr0.0005/1', exist_ok=True)
    plt.savefig('/home/zixshu/DeepGWAS/plots_bedreader_relu_lr0.0005/1/numsnp{}_var_{}_interaction{}.png'.format(num_snp, variating_variable, interaction))
    # os.makedirs('/home/zixshu/DeepGWAS/plots_leakyrelu/1', exist_ok=True)
    # plt.savefig('/home/zixshu/DeepGWAS/plots_leakyrelu/1/numsnp{}_var_{}_interaction{}.png'.format(num_snp, variating_variable, interaction))


PATH='/home/zixshu/DeepGWAS/metrics_bedreader_relu_lr0.0005/1'
filenames=os.listdir('/home/zixshu/DeepGWAS/metrics_bedreader_relu_lr0.0005/1')

#######################################################################################
# this is the plot for experiment for num_snp=20
#
# extracting max_present dictionary
evaluation_scores_true={}
evaluation_scores_false={}

#variating max_present
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

plot_comparisions(evaluation_scores_true,'max_present','True', "20")
plot_comparisions(evaluation_scores_false,'max_present','False', "20")



# extracting prevalence dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    
    splited_name=file.split('_')
    train_parameter=splited_name[4].split('.p')[0]

    
    if( ('nsnp20' in file ) and ('csnp3' in file) and ('max1.0'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'prevalence','True',"20")
plot_comparisions(evaluation_scores_false,'prevalence','False',"20")




# extracting number of casual SNP dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[2]

    
    if( ('nsnp20' in file ) and ('prevalence0.35.pkl' in file) and ('max1.0'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'csnp','True',"20")
plot_comparisions(evaluation_scores_false,'csnp','False',"20")




#######################################################################################
# this is the plot for experiment for num_snp=200
#
# extracting max_present dictionary
evaluation_scores_true={}
evaluation_scores_false={}

#variating max_present
for file in filenames:
   
    splited_name=file.split('_')
    train_parameter=splited_name[1]
    
    if( ('nsnp200' in file ) and ('csnp3' in file) and ('prevalence0.35.pkl'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

# plot_comparisions(evaluation_scores_true,'max_present','True', "200")
# plot_comparisions(evaluation_scores_false,'max_present','False', "200")



# extracting prevalence dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[4].split('.p')[0]

    if( ('nsnp200' in file ) and ('csnp3' in file) and ('max0.8'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'prevalence','True',"200")
plot_comparisions(evaluation_scores_false,'prevalence','False',"200")




# extracting number of casual SNP dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[2]

    
    if( ('nsnp200' in file ) and ('prevalence0.35.pkl' in file) and ('max0.8'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}
 
        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

plot_comparisions(evaluation_scores_true,'csnp','True',"200")
plot_comparisions(evaluation_scores_false,'csnp','False',"200")





#######################################################################################
# this is the plot for experiment for num_snp=2000
#
# extracting max_present dictionary
evaluation_scores_true={}
evaluation_scores_false={}

#variating max_present
for file in filenames:
   
    splited_name=file.split('_')
    train_parameter=splited_name[1]
    
    if( ('nsnp2000' in file ) and ('csnp3' in file) and ('prevalence0.35.pkl'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

# plot_comparisions(evaluation_scores_true,'max_present','True', "2000")
# plot_comparisions(evaluation_scores_false,'max_present','False', "2000")



# extracting prevalence dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[4].split('.p')[0]

    if( ('nsnp2000' in file ) and ('csnp3' in file) and ('max0.5'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

# plot_comparisions(evaluation_scores_true,'prevalence','True',"2000")
# plot_comparisions(evaluation_scores_false,'prevalence','False',"2000")




# extracting number of casual SNP dictionary
evaluation_scores_true={}
evaluation_scores_false={}
for file in filenames:
    splited_name=file.split('_')
    train_parameter=splited_name[2]

    
    if( ('nsnp2000' in file ) and ('prevalence0.35.pkl' in file) and ('max0.5'in file) ):
        if 'iTrue' in file:
            evaluation_scores_true[train_parameter]={}

        else:
            evaluation_scores_false[train_parameter]={}

        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        evaluation_scores_true, evaluation_scores_false=extract_dicts(train_parameter, file , evaluation_dict, evaluation_scores_true, evaluation_scores_false)

# plot_comparisions(evaluation_scores_true,'csnp','True',"2000")
# plot_comparisions(evaluation_scores_false,'csnp','False',"2000")

































