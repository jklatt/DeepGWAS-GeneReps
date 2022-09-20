import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

path="/home/zixshu/DeepGWAS/semi_simulation_setting/metrics_bedreader_leakyrelu_reduceplateu_lr0.0005_twostep_MLP_upsampling_attweight_attention_alogpick/"
gene_name_list=['AT2G14030', 'AT3G26240', 'AT3G26260', 'AT3G31005', 'AT5G45060']
genes=range(5)
evaluation_scores_true={}
evaluation_scores_false={}

for gene in genes:
    PATH=path+str(gene)
    filenames=os.listdir(PATH)
    gene_name=gene_name_list[gene]

    
    for file in filenames:
        FILE_PATH=PATH+'/'+file
        with open(FILE_PATH, "rb") as f:
            evaluation_dict=pickle.load(f)

        if 'iTrue' in file:
            evaluation_scores_true[gene_name]={}
            evaluation_scores_true[gene_name]['roc_auc']=evaluation_dict['roc_auc_bag']
            evaluation_scores_true[gene_name]['prc_avg']=evaluation_dict['prc_avg_bag']
            evaluation_scores_true[gene_name]['fpr']=evaluation_dict['fpr_bag']
            evaluation_scores_true[gene_name]['tpr']=evaluation_dict['tpr_bag']
            evaluation_scores_true[gene_name]['precision']=evaluation_dict['precision_bag']
            evaluation_scores_true[gene_name]['recall']=evaluation_dict['recall_bag']


        else:
            evaluation_scores_false[gene_name]={}
            evaluation_scores_false[gene_name]['roc_auc']=evaluation_dict['roc_auc_bag']
            evaluation_scores_false[gene_name]['prc_avg']=evaluation_dict['prc_avg_bag']
            evaluation_scores_false[gene_name]['fpr']=evaluation_dict['fpr_bag']
            evaluation_scores_false[gene_name]['tpr']=evaluation_dict['tpr_bag']
            evaluation_scores_false[gene_name]['precision']=evaluation_dict['precision_bag']
            evaluation_scores_false[gene_name]['recall']=evaluation_dict['recall_bag']



figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle("semi simulation setting w interation")
for gene in gene_name_list:
    axis[0].plot(evaluation_scores_true[gene]['fpr'],evaluation_scores_true[gene]['tpr'],label="{} AUC=".format(gene)+str(round(evaluation_scores_true[gene]['roc_auc'],3)))
    axis[1].plot(evaluation_scores_true[gene]['recall'], evaluation_scores_true[gene]['precision'],label="{} AUC=".format(gene)+str(round(evaluation_scores_true[gene]['prc_avg'],3)))
axis[0].legend()
axis[0].set_title('Bag level ROC')
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')
axis[0].legend(loc = 'lower right')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1])
axis[0].plot([0, 1], [0, 1],'r--')

axis[1].legend()
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].legend(loc = 'lower left')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])

SAVING_PATH="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_snplength20_iTrue.png"
plt.savefig(SAVING_PATH)

figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle("semi simulation setting w/o interation")
for gene in gene_name_list:
    axis[0].plot(evaluation_scores_false[gene]['fpr'],evaluation_scores_false[gene]['tpr'],label="{} AUC=".format(gene)+str(round(evaluation_scores_false[gene]['roc_auc'],3)))
    axis[1].plot(evaluation_scores_false[gene]['recall'], evaluation_scores_false[gene]['precision'],label="{} AUC=".format(gene)+str(round(evaluation_scores_false[gene]['prc_avg'],3)))
axis[0].legend()
axis[0].set_title('Bag level ROC')
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')
axis[0].legend(loc = 'lower right')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1])
axis[0].plot([0, 1], [0, 1],'r--')

axis[1].legend()
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].legend(loc = 'lower left')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])


SAVING_PATH="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_snplength20_iFalse.png"
plt.savefig(SAVING_PATH)








