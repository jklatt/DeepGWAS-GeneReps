import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


ploting_length=200
setting='control'


if setting=='control':
    seeds=range(1,6)
    METRIC_PATH="/home/zixshu/DeepGWAS/baseline/metrics_correctbaseline"
elif setting=="semi":
    seeds=range(5)
    METRIC_PATH="/home/zixshu/DeepGWAS/baseline/metrics_fourmoments_semi_logistic"

evaluation_scores_true={}
evaluation_scores_false={}
for seed in seeds:
    PATH=METRIC_PATH+"/"+str(seed)
    filenames=os.listdir(PATH)
    for file in filenames:
        if ploting_length==200:
            if "snp200_" in file or 'selectedlength200_' in file:
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
            if "snp20_" in file or 'selectedlength20_' in file:
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

def interporating(evaluation_scores_true, seed,variable1, variable2):
    new_a1_x = np.linspace(evaluation_scores_true[seed][variable1][0], evaluation_scores_true[seed][variable1][-1], 1000)
    new_a2_x = np.linspace(evaluation_scores_true[seed][variable2][0], evaluation_scores_true[seed][variable2][-1],1000)
 
    new_a1_y = forward_filling(new_a1_x,evaluation_scores_true[seed][variable1], evaluation_scores_true[seed][variable2])
    if evaluation_scores_true[seed][variable2][0]-evaluation_scores_true[seed][variable2][-1]>0:
        new_a2_y = forward_filling(np.flip(new_a2_x), np.flip(evaluation_scores_true[seed][variable2]),np.flip(evaluation_scores_true[seed][variable1]))
        evaluation_scores_true[seed]['{}_inter'.format(variable1)]=np.flip(new_a2_y)
        
    else:
        new_a2_y = forward_filling(new_a2_x, evaluation_scores_true[seed][variable2], evaluation_scores_true[seed][variable1])
        evaluation_scores_true[seed]['{}_inter'.format(variable1)]=new_a2_y

    evaluation_scores_true[seed]['{}_inter'.format(variable2)]=new_a1_y
    return evaluation_scores_true

#interporating the lines
for seed in seeds:
    evaluation_scores_true=interporating(evaluation_scores_true, seed,'precision','recall') 
    evaluation_scores_true=interporating(evaluation_scores_true, seed,'fpr', 'tpr')
    evaluation_scores_false=interporating(evaluation_scores_false, seed,'precision','recall') 
    evaluation_scores_false=interporating(evaluation_scores_false, seed,'fpr', 'tpr')


precision_true=[]
precision_false=[]
recall_true=[]
recall_false=[]
fpr_true=[]
tpr_true=[]
fpr_false=[]
tpr_false=[]
for seed in seeds:
    precision_true.append(evaluation_scores_true[seed]['precision_inter'])
    recall_true.append(evaluation_scores_true[seed]['recall_inter'])
    precision_false.append(evaluation_scores_false[seed]['precision_inter'])
    recall_false.append(evaluation_scores_false[seed]['recall_inter'])

    fpr_true.append(evaluation_scores_true[seed]['fpr_inter'])
    tpr_true.append(evaluation_scores_true[seed]['tpr_inter'])
    fpr_false.append(evaluation_scores_false[seed]['fpr_inter'])
    tpr_false.append(evaluation_scores_false[seed]['tpr_inter'])


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
figure.suptitle("control setting w interation baseline")
# for seed in seeds:
#     axis[0].plot(evaluation_scores_true[seed]['fpr'],evaluation_scores_true[seed]['tpr'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_true[seed]['roc_auc'],3)))
#     axis[1].plot(evaluation_scores_true[seed]['recall'], evaluation_scores_true[seed]['precision'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_true[seed]['prc_avg'],3)))
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
axis[0].set_ylim([0, 1])
axis[0].plot([0, 1], [0, 1],'r--')

# axis[1].legend()
# axis[1].legend(loc = 'lower left')
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
# for seed in seeds:
#     axis[0].plot(evaluation_scores_false[seed]['fpr'],evaluation_scores_false[seed]['tpr'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_false[seed]['roc_auc'],3)))
#     axis[1].plot(evaluation_scores_false[seed]['recall'], evaluation_scores_false[seed]['precision'],label="{} AUC=".format(str(seed))+str(round(evaluation_scores_false[seed]['prc_avg'],3)))
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
axis[0].set_ylim([0, 1])
axis[0].plot([0, 1], [0, 1],'r--')

# axis[1].legend()
# axis[1].legend(loc = 'lower left')
axis[1].set_title('Bag level PRC')
axis[1].set_xlabel('Recall')
axis[1].set_ylabel('Precision')
axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])             

# SAVING_PATH="/home/zixshu/DeepGWAS/baseline_semi_setting_logisticreg_plots/"
SAVING_PATH="/home/zixshu/DeepGWAS/baseline_{}_setting_plots/".format(setting)
os.makedirs(SAVING_PATH,exist_ok=True)
plt.savefig(SAVING_PATH+"plot_snplength{}_iFalse_test.png".format(str(ploting_length)))












