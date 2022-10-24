from fileinput import filename
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

seeds=range(1,6)
ploting_snp=20
criteria_bag="allbags_"
# criteria_bag="truebags"

# models=["attention","gated attention"]
models=["attention present","gated attention present", "attention","gated attention"]


if ploting_snp==20:
    gene_name_list=['AT5G48440','AT3G52970', 'AT2G36570','AT4G10350', 'AT2G16676']
elif ploting_snp==200:
    gene_name_list=['AT2G14030', 'AT3G26240', 'AT3G26260', 'AT3G31005', 'AT5G45060']
elif ploting_snp==500:
    gene_name_list=['AT1G43060', 'AT1G58602', 'AT4G19490', 'AT5G24740', 'AT5G32690']

interporated_score={}
interporated_score['true']={}
interporated_score['false']={}

# def interporating(evaluation_scores_true, gene,variable1, variable2, criteria_bag,interporated_score):
#     new_a1_x = np.linspace(evaluation_scores_true[gene][variable1][0], evaluation_scores_true[gene][variable1][-1], 1000)
#     new_a2_x = np.linspace(evaluation_scores_true[gene][variable2][0], evaluation_scores_true[gene][variable2][-1],1000)
    
#     if evaluation_scores_true[gene][variable2][0]-evaluation_scores_true[gene][variable2][-1]>0:
#         new_a2_y = forward_filling(np.flip(new_a2_x), np.flip(evaluation_scores_true[gene][variable2]),np.flip(evaluation_scores_true[gene][variable1]))
#         evaluation_scores_true[gene]['{}_inter'.format(variable1)]=np.flip(new_a2_y)
        
#     else:
#         new_a2_y = forward_filling(new_a2_x, evaluation_scores_true[gene][variable2], evaluation_scores_true[gene][variable1])
#         evaluation_scores_true[gene]['{}_inter'.format(variable1)]=new_a2_y

#     # evaluation_scores_true[gene]['{}_inter'.format(variable2)]=new_a1_y
#     evaluation_scores_true[gene]['{}_inter'.format(variable2)]=new_a2_x
#     return evaluation_scores_true


def evaluation_dic_generation(model,ploting_length):
    if model=="attention":
        # #attention
        path="/home/zixshu/DeepGWAS/semi_simulation_setting/instance_level_results_lr0.0005_attention_onlypresentFalse_withattention_manual_all/"
        saving_path="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_instances_level_attention_onlypresentFalse/"

    elif model=="gated attention":
        path="/home/zixshu/DeepGWAS/semi_simulation_setting/instance_level_results_lr0.0005_gated_attention_onlypresentFalse_withattention_manual_all/"
        saving_path="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_instances_level_gated_attention_onlypresentFalse/"

    elif model=="attention present":
        path="/home/zixshu/DeepGWAS/semi_simulation_setting/instance_level_results_lr0.0005_attention_onlypresent_onlypresentTrue_withattention_manual_all/"
        saving_path="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_instances_level_attention_onlypresentTrue/"

    elif model=="gated attention present":
        path="/home/zixshu/DeepGWAS/semi_simulation_setting/instance_level_results_lr0.0005_gated_attention_onlypresent_onlypresentTrue_withattention_manual_all/"
        saving_path="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/plot_instances_level_gated_attention_onlypresentTrue/"


    genes=range(5)
    evaluation_scores_true={}
    evaluation_scores_false={}

    for gene in genes:
        PATH=path
        filenames=os.listdir(PATH)
        gene_name=gene_name_list[gene]

    
        for file in filenames:
            if ploting_length==20:
                criteria_length="20_"               
            elif ploting_length==200:
                criteria_length="200_"
            elif ploting_length==500:
                criteria_length=str(500)
                        
            
            if 'iTrue' in file and (criteria_bag in file)and ("geneind"+str(gene) in file) and (criteria_length in file):
                FILE_PATH=PATH+'/'+file
                with open(FILE_PATH, "rb") as f:
                    evaluation_dict=pickle.load(f)

                if ("present" in model) and (criteria_bag=="allbags_"):
                    labels=evaluation_dict['labels']
                else:
                    labels=evaluation_dict['label']
                attention_weights=evaluation_dict[0]

                precision, recall, thresholds_prc = precision_recall_curve(labels, attention_weights)
                prc_avg = average_precision_score(labels,attention_weights)
                evaluation_scores_true[gene_name]={}    
                evaluation_scores_true[gene_name]['prc_avg']=prc_avg 
                evaluation_scores_true[gene_name]['precision']=precision
                evaluation_scores_true[gene_name]['recall']=recall

                # evaluation_scores_true[gene_name]['fpr']=evaluation_dict['fpr_bag']
                # evaluation_scores_true[gene_name]['tpr']=evaluation_dict['tpr_bag']
                # evaluation_scores_true[gene_name]['roc_auc']=evaluation_dict['roc_auc_bag']
            elif 'iFalse' in file and (criteria_bag in file)and ("geneind"+str(gene) in file) and (criteria_length in file):
                FILE_PATH=PATH+'/'+file
                with open(FILE_PATH, "rb") as f:
                    evaluation_dict=pickle.load(f)
                if ("present" in model) and (criteria_bag=="allbags_"):
                    labels=evaluation_dict['labels']
                else:
                    labels=evaluation_dict['label']

                attention_weights=evaluation_dict[0]
                precision, recall, thresholds_prc = precision_recall_curve(labels, attention_weights)
                # print(evaluation_dict)
                prc_avg = average_precision_score(labels,attention_weights)
                print(evaluation_dict)
                evaluation_scores_false[gene_name]={}
                evaluation_scores_false[gene_name]['prc_avg']=prc_avg
                evaluation_scores_false[gene_name]['precision']=precision
                evaluation_scores_false[gene_name]['recall']=recall

                # evaluation_scores_false[gene_name]['fpr']=evaluation_dict['fpr_bag']
                # evaluation_scores_false[gene_name]['tpr']=evaluation_dict['tpr_bag']
                # evaluation_scores_false[gene_name]['roc_auc']=evaluation_dict['roc_auc_bag']


    
    prc_avg_true_list=[]
    prc_avg_false_list=[]
    
    for gene in gene_name_list:
        prc_avg_true_list.append(evaluation_scores_true[gene]['prc_avg'])
        prc_avg_false_list.append(evaluation_scores_false[gene]['prc_avg'])

    interporated_score['true'][model]={}
    interporated_score['false'][model]={}
    interporated_score['true'][model]["prc_mean"]=np.mean(prc_avg_true_list)
    interporated_score['true'][model]["prc_std"]=np.std(prc_avg_true_list)
    interporated_score['false'][model]["prc_mean"]=np.mean(prc_avg_false_list)
    interporated_score['false'][model]["prc_std"]=np.std(prc_avg_false_list)
    

    #interporating the lines
    # for gene in gene_name_list:
        # evaluation_scores_true=interporating(evaluation_scores_true, gene,'precision','recall') 
        # evaluation_scores_true=interporating(evaluation_scores_true, gene,'fpr', 'tpr')
        # evaluation_scores_false=interporating(evaluation_scores_false, gene,'precision','recall') 
        # evaluation_scores_false=interporating(evaluation_scores_false, gene,'fpr', 'tpr')

    # precision_true=[]
    # precision_false=[]
    # recall_true=[]
    # recall_false=[]
    # fpr_true=[]
    # tpr_true=[]
    # fpr_false=[]
    # tpr_false=[]
    # for gene in gene_name_list:
    #     precision_true.append(evaluation_scores_true[gene]['precision_inter'])
    #     recall_true.append(evaluation_scores_true[gene]['recall_inter'])
    #     precision_false.append(evaluation_scores_false[gene]['precision_inter'])
    #     recall_false.append(evaluation_scores_false[gene]['recall_inter'])

    #     fpr_true.append(evaluation_scores_true[gene]['fpr_inter'])
    #     tpr_true.append(evaluation_scores_true[gene]['tpr_inter'])
    #     fpr_false.append(evaluation_scores_false[gene]['fpr_inter'])
    #     tpr_false.append(evaluation_scores_false[gene]['tpr_inter'])


    # interporated_score={}
    # interporated_score['true']={}
    # interporated_score['false']={}
    # interporated_score['true']['avg_precision']=np.mean(np.array(precision_true),axis=0)
    # interporated_score['true']['avg_recall']=np.mean(np.array(recall_true),axis=0)
    # interporated_score['false']['avg_precision']=np.mean(np.array(precision_false),axis=0)
    # interporated_score['false']['avg_recall']=np.mean(np.array(recall_false),axis=0)

    # interporated_score['true']['std_precision']=np.std(np.array(precision_true),axis=0)
    # interporated_score['true']['std_recall']=np.std(np.array(recall_true),axis=0)
    # interporated_score['false']['std_precision']=np.std(np.array(precision_false),axis=0)
    # interporated_score['false']['std_recall']=np.std(np.array(recall_false),axis=0)

    # interporated_score['true']['avg_fpr']=np.mean(np.array(fpr_true),axis=0)
    # interporated_score['true']['avg_tpr']=np.mean(np.array(tpr_true),axis=0)
    # interporated_score['false']['avg_fpr']=np.mean(np.array(fpr_false),axis=0)
    # interporated_score['false']['avg_tpr']=np.mean(np.array(tpr_false),axis=0)

    # interporated_score['true']['std_fpr']=np.std(np.array(fpr_true),axis=0)
    # interporated_score['true']['std_tpr']=np.std(np.array(tpr_true),axis=0)
    # interporated_score['false']['std_fpr']=np.std(np.array(fpr_false),axis=0)
    # interporated_score['false']['std_tpr']=np.std(np.array(tpr_false),axis=0)

    return interporated_score



# def forward_filling(new_list1, list1, list2):
#     filled_list1=[[]]*len(new_list1)

#     for i in range(0,len(new_list1)):
#         if i==0:
#             filled_list1[i]=list2[0]

#         if i==(len(new_list1)-1):
#             filled_list1[i]=list2[-1]
#         else:
#             greater_list=list1>new_list1[i]
#             greater_ind=[i for i, x in enumerate(greater_list) if x]
#             if len(greater_ind)>0:
#                 filled_list1[i]=list2[min(greater_ind)-1]

#     return filled_list1







for model in models:
    interporated_score=evaluation_dic_generation(model,ploting_snp)   
    # axis[0].plot(interporated_score['true']['avg_fpr'], interporated_score['true']['avg_tpr'],label="w.i {}".format(model))
    # axis[0].fill_between(x=interporated_score['true']['avg_fpr'],y1=np.add(np.array(interporated_score['true']['avg_tpr']), np.array(interporated_score['true']['std_tpr'])), y2=np.subtract(np.array(interporated_score['true']['avg_tpr']),np.array(interporated_score['true']['std_tpr'])),alpha=0.2)
prc_true_line=[]
prc_false_line=[]
prc_true_line_sd=[]
prc_false_line_sd=[]
for model in models:
    prc_true_line.append(interporated_score['true'][model]['prc_mean'])
    prc_true_line_sd.append(interporated_score['true'][model]['prc_mean'])
    prc_false_line.append(interporated_score['false'][model]['prc_std'])
    prc_false_line_sd.append(interporated_score['false'][model]['prc_std'])

#TODO: PLOTING FOR PRC LINE AND CHECK IF THE PERFORMANCE IS THIS BAD COMPARED TO THE TOY EXAMPLE.

figure, axis = plt.subplots(1, 1, figsize=(7, 7))
figure.suptitle("semi simulation setting with selected gene length{}".format(str(ploting_snp)))

x = np.array(range(len(models)))
axis.set_xticks(x, models)
axis.plot(x, prc_true_line,label="w interaction")
axis.fill_between(x=x,y1=np.add(prc_true_line, prc_true_line_sd), y2=np.subtract(prc_true_line,prc_true_line_sd),alpha=0.2)

# axis[0].plot(interporated_score['false']['avg_fpr'],interporated_score['false']['avg_tpr'], label="w/o.i {}".format(model))
# axis[0].fill_between(x=interporated_score['false']['avg_fpr'],y1=np.add(np.array(interporated_score['false']['avg_tpr']), np.array(interporated_score['false']['std_tpr'])), y2=np.subtract(np.array(interporated_score['false']['avg_tpr']),np.array(interporated_score['false']['std_tpr'])),alpha=0.2)
axis.plot(x, prc_false_line,label="w/o interaction")
axis.fill_between(x=x,y1=np.add(prc_false_line, prc_false_line_sd), y2=np.subtract(prc_false_line,prc_false_line_sd),alpha=0.2)


axis.legend()
axis.legend(loc = 'lower right')
axis.set_title('Instance level Evaluation')
axis.set_xlabel('Models')
axis.set_ylabel('Average Precision')
axis.set_xlim([0, 3])
axis.set_ylim([0, 1.02])


# axis[0].legend()
# axis[0].legend(loc = 'lower right')
# axis[0].set_title('Bag level ROC')
# axis[0].set_ylabel('True Positive Rate')
# axis[0].set_xlabel('False Positive Rate')
# axis[0].set_xlim([0, 1])
# axis[0].set_ylim([0, 1.1])
# axis[0].plot([0, 1], [0, 1],'r--')

# axis[1].legend()
# axis[1].legend(loc = 'lower left')
# axis[1].set_title('Bag level PRC')
# axis[1].set_xlabel('Recall')
# axis[1].set_ylabel('Precision')
# axis[1].set_xlim([0, 1])
# axis[1].set_ylim([0, 1])


os.makedirs("/home/zixshu/DeepGWAS/semi_simulation_setting_plots",exist_ok=True)
SAVING_PATH="/home/zixshu/DeepGWAS/semi_simulation_setting_plots/instancelevel_plot_snplength{}_manual_presentsnp_together_comparingmodel.png".format(str(ploting_snp))
plt.savefig(SAVING_PATH)
