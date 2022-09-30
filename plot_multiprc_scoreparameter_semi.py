import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

ploting_length=200
if ploting_length==20:
    #snp20
    path="/home/zixshu/DeepGWAS/semi_simulation_setting/metrics_bedreader_leakyrelu_reduceplateu_lr0.0005_twostep_MLP_upsampling_attweight_attention_alogpick/"
    gene_name_list=['AT5G48440','AT3G52970', 'AT2G36570','AT4G10350', 'AT2G16676']
elif ploting_length==200:
    #snp200
    path="/home/zixshu/DeepGWAS/semi_simulation_setting/metrics_bedreader_leakyrelu_reduceplateu_lr0.0005_twostep_MLP_upsampling_attweight_attention_snplength200_alogpick/"
    gene_name_list=['AT2G14030', 'AT3G26240', 'AT3G26260', 'AT3G31005', 'AT5G45060']
elif ploting_length==500:
    #snp greater than 500
    path="/home/zixshu/DeepGWAS/semi_simulation_setting/metrics_bedreader_leakyrelu_reduceplateu_lr0.0005_twostep_MLP_upsampling_attweight_attention_snplength500_alogpick/"
    gene_name_list=['AT1G43060', 'AT1G58602', 'AT4G19490', 'AT5G24740', 'AT5G32690']


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


def interporating(evaluation_scores_true, gene,variable1, variable2):
    min_pre,max_pre=min(evaluation_scores_true[gene][variable1]), max(evaluation_scores_true[gene][variable1])
    min_recall,max_recall=min(evaluation_scores_true[gene][variable2]), max(evaluation_scores_true[gene][variable2])
    new_a1_x = np.linspace(min_pre, max_pre, 50)
    new_a2_x = np.linspace(min_recall, max_recall,50)
    new_a1_y = np.interp(new_a1_x, evaluation_scores_true[gene][variable1], evaluation_scores_true[gene][variable2])
    new_a2_y = np.interp(new_a2_x, np.flip(evaluation_scores_true[gene][variable2]),np.flip(evaluation_scores_true[gene][variable1]))
    evaluation_scores_true[gene]['{}_inter'.format(variable1)]=np.flip(new_a2_y)
    evaluation_scores_true[gene]['{}_inter'.format(variable2)]=new_a1_y
    return evaluation_scores_true

#interporating the lines
for gene in gene_name_list:
    evaluation_scores_true=interporating(evaluation_scores_true, gene,'precision','recall') 
    evaluation_scores_true=interporating(evaluation_scores_true, gene,'fpr', 'tpr')
    evaluation_scores_false=interporating(evaluation_scores_false, gene,'precision','recall') 
    evaluation_scores_false=interporating(evaluation_scores_false, gene,'fpr', 'tpr')


precision_true=[]
precision_false=[]
recall_true=[]
recall_false=[]
fpr_true=[]
tpr_true=[]
fpr_false=[]
tpr_false=[]
for gene in gene_name_list:
    precision_true.append(evaluation_scores_true[gene]['precision_inter'])
    recall_true.append(evaluation_scores_true[gene]['recall_inter'])
    precision_false.append(evaluation_scores_false[gene]['precision_inter'])
    recall_false.append(evaluation_scores_false[gene]['recall_inter'])

    fpr_true.append(evaluation_scores_true[gene]['fpr_inter'])
    tpr_true.append(evaluation_scores_true[gene]['tpr_inter'])
    fpr_false.append(evaluation_scores_false[gene]['fpr_inter'])
    tpr_false.append(evaluation_scores_false[gene]['tpr_inter'])


interporated_score={}
interporated_score['true']={}
interporated_score['false']={}
interporated_score['true']['avg_precision']=np.mean(np.array(precision_true),axis=0)
interporated_score['true']['avg_recall']=np.mean(np.array(recall_true),axis=0)
interporated_score['false']['avg_precision']=np.mean(np.array(precision_false),axis=0)
interporated_score['false']['avg_recall']=np.mean(np.array(recall_false),axis=0)

interporated_score['true']['std_precision']=np.std(np.array(precision_true),axis=0)
interporated_score['true']['std_recall']=np.std(np.array(recall_true),axis=0)
interporated_score['false']['std_precision']=np.std(np.array(precision_false),axis=0)
interporated_score['false']['std_recall']=np.std(np.array(recall_false),axis=0)

interporated_score['true']['avg_fpr']=np.mean(np.array(fpr_true),axis=0)
interporated_score['true']['avg_tpr']=np.mean(np.array(tpr_true),axis=0)
interporated_score['false']['avg_fpr']=np.mean(np.array(fpr_false),axis=0)
interporated_score['false']['avg_tpr']=np.mean(np.array(tpr_false),axis=0)

interporated_score['true']['std_fpr']=np.std(np.array(fpr_true),axis=0)
interporated_score['true']['std_tpr']=np.std(np.array(tpr_true),axis=0)
interporated_score['false']['std_fpr']=np.std(np.array(fpr_false),axis=0)
interporated_score['false']['std_tpr']=np.std(np.array(tpr_false),axis=0)



figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle("semi simulation setting w interation")
# for gene in gene_name_list:
    # axis[0].plot(evaluation_scores_true[gene]['fpr_inter'],evaluation_scores_true[gene]['tpr_inter'],label="{} AUC=".format(gene)+str(round(evaluation_scores_true[gene]['roc_auc'],3)))
    # axis[1].plot(evaluation_scores_true[gene]['recall_inter'], evaluation_scores_true[gene]['precision_inter'],label="{} AUC=".format(gene)+str(round(evaluation_scores_true[gene]['prc_avg'],3)))
    
axis[0].plot(interporated_score['true']['avg_fpr'], interporated_score['true']['avg_tpr'])
axis[0].fill_between(x=interporated_score['true']['avg_fpr'],y1=np.add(np.array(interporated_score['true']['avg_tpr']), np.array(interporated_score['true']['std_tpr'])), y2=np.subtract(np.array(interporated_score['true']['avg_tpr']),np.array(interporated_score['true']['std_tpr'])),alpha=0.2)
axis[1].plot(interporated_score['true']['avg_recall'], interporated_score['true']['avg_precision'])
axis[1].fill_between(x=interporated_score['true']['avg_recall'],y1=np.add(np.array(interporated_score['true']['avg_precision']), np.array(interporated_score['true']['std_precision'])), y2=np.subtract(np.array(interporated_score['true']['avg_precision']),np.array(interporated_score['true']['std_precision'])),alpha=0.2)

# axis[0].legend()
# axis[0].legend(loc = 'lower right')
axis[0].set_title('Bag level ROC')
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1.1])
axis[0].plot([0, 1], [0, 1],'r--')

# axis[1].legend()
# axis[1].legend(loc = 'lower left')
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])

SAVING_PATH="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_snplength{}_iTrue_test.png".format(str(ploting_length))
plt.savefig(SAVING_PATH)

figure, axis = plt.subplots(1, 2, figsize=(7, 7))
figure.suptitle("semi simulation setting w/o interation")
# for gene in gene_name_list:
#     axis[0].plot(evaluation_scores_false[gene]['fpr_inter'],evaluation_scores_false[gene]['tpr_inter'],label="{} AUC=".format(gene)+str(round(evaluation_scores_false[gene]['roc_auc'],3)))
#     axis[1].plot(evaluation_scores_false[gene]['recall_inter'], evaluation_scores_false[gene]['precision_inter'],label="{} AUC=".format(gene)+str(round(evaluation_scores_false[gene]['prc_avg'],3)))
axis[0].plot(interporated_score['false']['avg_fpr'],interporated_score['false']['avg_tpr'])
axis[0].fill_between(x=interporated_score['false']['avg_fpr'],y1=np.add(np.array(interporated_score['false']['avg_tpr']), np.array(interporated_score['false']['std_tpr'])), y2=np.subtract(np.array(interporated_score['false']['avg_tpr']),np.array(interporated_score['false']['std_tpr'])),alpha=0.2)
axis[1].plot(interporated_score['false']['avg_recall'], interporated_score['false']['avg_precision'])
axis[1].fill_between(x=interporated_score['false']['avg_recall'],y1=np.add(np.array(interporated_score['false']['avg_precision']), np.array(interporated_score['false']['std_precision'])), y2=np.subtract(np.array(interporated_score['false']['avg_precision']),np.array(interporated_score['false']['std_precision'])),alpha=0.2)

# axis[0].legend()
# axis[0].legend(loc = 'lower right')
axis[0].set_title('Bag level ROC')
axis[0].set_ylabel('True Positive Rate')
axis[0].set_xlabel('False Positive Rate')
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1.1])
axis[0].plot([0, 1], [0, 1],'r--')

# axis[1].legend()
# axis[1].legend(loc = 'lower left')
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])


SAVING_PATH="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_snplength{}_iFalse_test.png".format(str(ploting_length))
plt.savefig(SAVING_PATH)








