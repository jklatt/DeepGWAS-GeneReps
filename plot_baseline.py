import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


ploting_length=500
setting='semi'


if setting=='control':
    seeds=range(1,6)
    METRIC_PATH="/home/zixshu/DeepGWAS/baseline/metrics_fourmoments"
elif setting=="semi":
    seeds=range(5)
    METRIC_PATH="/home/zixshu/DeepGWAS/baseline/metrics_fourmoments_semi"

evaluation_scores_true={}
evaluation_scores_false={}
for seed in seeds:
    PATH=METRIC_PATH+"/"+str(seed)
    filenames=os.listdir(PATH)
    for file in filenames:
        if ploting_length==200:
            if "snp200_" or 'selectedlength20_' in file:
                FILE_PATH=PATH+"/"+file
                with open(FILE_PATH, "rb") as f:
                    evaluation_dict=pickle.load(f)
                if "iTrue" in file:
                    evaluation_scores_true[seed]={}
                    evaluation_scores_true[seed]['precision']=evaluation_dict['precision']
                    evaluation_scores_true[seed]['recall']=evaluation_dict['recall']
                    evaluation_scores_true[seed]['fpr']=evaluation_dict['fpr']
                    evaluation_scores_true[seed]['tpr']=evaluation_dict['tpr']
                    evaluation_scores_true[seed]['roc_auc']=evaluation_dict['roc_auc']
                    evaluation_scores_true[seed]['prc_avg']=evaluation_dict['prc_avg']

                else:
                    evaluation_scores_false[seed]={}
                    evaluation_scores_false[seed]['precision']=evaluation_dict['precision']
                    evaluation_scores_false[seed]['recall']=evaluation_dict['recall']
                    evaluation_scores_false[seed]['fpr']=evaluation_dict['fpr']
                    evaluation_scores_false[seed]['tpr']=evaluation_dict['tpr']
                    evaluation_scores_false[seed]['roc_auc']=evaluation_dict['roc_auc']
                    evaluation_scores_false[seed]['prc_avg']=evaluation_dict['prc_avg']
            
        elif ploting_length==20:
            if "snp20_" or 'selectedlength200_' in file:
                FILE_PATH=PATH+"/"+file
                with open(FILE_PATH, "rb") as f:
                    evaluation_dict=pickle.load(f)
                if "iTrue" in file:
                    evaluation_scores_true[seed]={}
                    evaluation_scores_true[seed]['precision']=evaluation_dict['precision']
                    evaluation_scores_true[seed]['recall']=evaluation_dict['recall']
                    evaluation_scores_true[seed]['fpr']=evaluation_dict['fpr']
                    evaluation_scores_true[seed]['tpr']=evaluation_dict['tpr']
                    evaluation_scores_true[seed]['roc_auc']=evaluation_dict['roc_auc']
                    evaluation_scores_true[seed]['prc_avg']=evaluation_dict['prc_avg']
                else:
                    evaluation_scores_false[seed]={}
                    evaluation_scores_false[seed]['precision']=evaluation_dict['precision']
                    evaluation_scores_false[seed]['recall']=evaluation_dict['recall']
                    evaluation_scores_false[seed]['fpr']=evaluation_dict['fpr']
                    evaluation_scores_false[seed]['tpr']=evaluation_dict['tpr']
                    evaluation_scores_false[seed]['roc_auc']=evaluation_dict['roc_auc']
                    evaluation_scores_false[seed]['prc_avg']=evaluation_dict['prc_avg']

        elif ploting_length==500:
            if 'selectedlength500_' in file:
                FILE_PATH=PATH+"/"+file
                with open(FILE_PATH, "rb") as f:
                    evaluation_dict=pickle.load(f)
                if "iTrue" in file:
                    evaluation_scores_true[seed]={}
                    evaluation_scores_true[seed]['precision']=evaluation_dict['precision']
                    evaluation_scores_true[seed]['recall']=evaluation_dict['recall']
                    evaluation_scores_true[seed]['fpr']=evaluation_dict['fpr']
                    evaluation_scores_true[seed]['tpr']=evaluation_dict['tpr']
                    evaluation_scores_true[seed]['roc_auc']=evaluation_dict['roc_auc']
                    evaluation_scores_true[seed]['prc_avg']=evaluation_dict['prc_avg']
                else:
                    evaluation_scores_false[seed]={}
                    evaluation_scores_false[seed]['precision']=evaluation_dict['precision']
                    evaluation_scores_false[seed]['recall']=evaluation_dict['recall']
                    evaluation_scores_false[seed]['fpr']=evaluation_dict['fpr']
                    evaluation_scores_false[seed]['tpr']=evaluation_dict['tpr']
                    evaluation_scores_false[seed]['roc_auc']=evaluation_dict['roc_auc']
                    evaluation_scores_false[seed]['prc_avg']=evaluation_dict['prc_avg']



figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle("control setting w interation baseline")
for seed in seeds:
    axis[0].plot(evaluation_scores_true[seed]['fpr'],evaluation_scores_true[seed]['tpr'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_true[seed]['roc_auc'],3)))
    axis[1].plot(evaluation_scores_true[seed]['recall'], evaluation_scores_true[seed]['precision'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_true[seed]['prc_avg'],3)))
    
axis[0].legend()
axis[0].legend(loc = 'lower right')
axis[0].set_title('Bag level ROC')
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1])
axis[0].plot([0, 1], [0, 1],'r--')

axis[1].legend()
axis[1].legend(loc = 'lower left')
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])             

# SAVING_PATH="/home/zixshu/DeepGWAS/baseline_semi_setting_logisticreg_plots/"
SAVING_PATH="/home/zixshu/DeepGWAS/baseline_{}_setting_plots/".format(setting)
os.makedirs(SAVING_PATH,exist_ok=True)
plt.savefig(SAVING_PATH+"plot_snplength{}_iTrue_test.png".format(str(ploting_length)))

####false interaction

figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle("control setting w/o interation baseline")
for seed in seeds:
    axis[0].plot(evaluation_scores_false[seed]['fpr'],evaluation_scores_false[seed]['tpr'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_false[seed]['roc_auc'],3)))
    axis[1].plot(evaluation_scores_false[seed]['recall'], evaluation_scores_false[seed]['precision'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_false[seed]['prc_avg'],3)))
    
axis[0].legend()
axis[0].legend(loc = 'lower right')
axis[0].set_title('Bag level ROC')
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1])
axis[0].plot([0, 1], [0, 1],'r--')

axis[1].legend()
axis[1].legend(loc = 'lower left')
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])             

# SAVING_PATH="/home/zixshu/DeepGWAS/baseline_semi_setting_logisticreg_plots/"
SAVING_PATH="/home/zixshu/DeepGWAS/baseline_{}_setting_plots/".format(setting)
os.makedirs(SAVING_PATH,exist_ok=True)
plt.savefig(SAVING_PATH+"plot_snplength{}_iFalse_test.png".format(str(ploting_length)))












