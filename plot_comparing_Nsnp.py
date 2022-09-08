from genericpath import exists
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


PATH="/home/zixshu/DeepGWAS/metrics_bedreader_leakyrelu_reduceplateu_lr0.0005_twostep_MLP_upsampling_attweight_attention_weightedfix_variousNsnp"
PATH_20200="/home/zixshu/DeepGWAS/metrics_bedreader_leakyrelu_reduceplateu_lr0.0005_twostep_MLP_upsampling_attweight_attention_weightedfix"
seeds=list(range(1,6))

Nsnp=['nsnp40',"nsnp50","nsnp100"]
evaluation_scores_true={}
evaluation_scores_false={}
for nsnp in Nsnp:
    prc_avg_true=[]
    prc_avg_false=[]
    prc_avg_true_20=[]
    prc_avg_false_20=[]
    prc_avg_true_200=[]
    prc_avg_false_200=[]

    for seed in seeds:
        SEEDFILE_PATH=PATH+"/"+str(seed)
        filenames=os.listdir(SEEDFILE_PATH)
        for file in filenames:
            if( (nsnp in file) and ('csnp3' in file) and ('prevalence0.35.pkl'in file) and ("max1.0" in file or "max0.95_" in file or "max0.9_" in file)):
                FILE_PATH=SEEDFILE_PATH+"/"+file
                with open(FILE_PATH, "rb") as f:
                    evaluation_dict=pickle.load(f)
                if "True" in file:
                    prc_avg_true.append(evaluation_dict['prc_avg_bag'])
                else:
                    prc_avg_false.append(evaluation_dict['prc_avg_bag'])

    evaluation_scores_true[nsnp]={}
    evaluation_scores_false[nsnp]={}
    evaluation_scores_true[nsnp]['prc_avg_mean']=np.mean(prc_avg_true)
    evaluation_scores_false[nsnp]['prc_avg_mean']=np.mean(prc_avg_false)
    evaluation_scores_true[nsnp]['prc_avg_std']=np.std(prc_avg_true)
    evaluation_scores_false[nsnp]['prc_avg_std']=np.std(prc_avg_false)


for seed in seeds:
    SEEDFILE_20200=PATH_20200+"/"+str(seed)
    filenames=os.listdir(SEEDFILE_20200)
    for file in filenames:
        if( ("nsnp20" in file) and ('csnp3' in file) and ('prevalence0.35.pkl'in file) and ("max1.0" in file)):
            FILE_PATH_20=SEEDFILE_20200+"/"+file
            with open(FILE_PATH_20, "rb") as f:
                evaluation_dict_20=pickle.load(f)
            if "True" in file:
                prc_avg_true_20.append(evaluation_dict_20['prc_avg_bag'])
            else:
                prc_avg_false_20.append(evaluation_dict_20['prc_avg_bag'])

        if( ("nsnp200" in file) and ('csnp3' in file) and ('prevalence0.35.pkl'in file) and ("max0.8" in file)):
            FILE_PATH_200=SEEDFILE_20200+"/"+file
            with open(FILE_PATH_200, "rb") as f:
                evaluation_dict_200=pickle.load(f)
            if "True" in file:
                prc_avg_true_200.append(evaluation_dict_200['prc_avg_bag'])
            else:
                prc_avg_false_200.append(evaluation_dict_200['prc_avg_bag'])


evaluation_scores_true["nsnp20"]={}
evaluation_scores_true["nsnp200"]={}
evaluation_scores_false["nsnp20"]={}
evaluation_scores_false["nsnp200"]={}

evaluation_scores_true["nsnp20"]['prc_avg_mean']=np.mean(prc_avg_true_20)
evaluation_scores_true["nsnp20"]['prc_avg_std']=np.std(prc_avg_true_20)
evaluation_scores_true["nsnp200"]['prc_avg_mean']=np.mean(prc_avg_true_200)
evaluation_scores_true["nsnp200"]['prc_avg_std']=np.std(prc_avg_true_200)

evaluation_scores_false["nsnp20"]['prc_avg_mean']=np.mean(prc_avg_false_20)
evaluation_scores_false["nsnp20"]['prc_avg_std']=np.std(prc_avg_false_20)
evaluation_scores_false["nsnp200"]['prc_avg_mean']=np.mean(prc_avg_false_200)
evaluation_scores_false["nsnp200"]['prc_avg_std']=np.std(prc_avg_false_200)




split_element=list(list(evaluation_scores_true.keys())[0])[list(list(evaluation_scores_true.keys())[0]).index('p')]
parameters=sorted(list(evaluation_scores_true.keys()), key=lambda x: float(x.split(split_element)[-1]))
# figure, axis = plt.subplots(1, 2, figsize=(7, 7))

x = np.array(range(len(parameters)))
prc_avg_mean_true_sorted=[]
prc_avg_mean_false_sorted=[]
prc_avg_sd_true_sorted=[]
prc_avg_sd_false_sorted=[]
for parameter in parameters:
    prc_avg_mean_true_sorted.append(evaluation_scores_true[parameter]['prc_avg_mean'])
    prc_avg_mean_false_sorted.append(evaluation_scores_false[parameter]['prc_avg_mean'])
    prc_avg_sd_true_sorted.append(evaluation_scores_true[parameter]['prc_avg_std'])
    prc_avg_sd_false_sorted.append(evaluation_scores_false[parameter]['prc_avg_std'])

plt.plot(x,prc_avg_mean_true_sorted,"-^", label="Interaction True")
plt.plot(x,prc_avg_mean_false_sorted,"-^", label="Interaction False")
plt.fill_between(np.array(x),y1=np.add(np.array(prc_avg_mean_true_sorted),np.array(prc_avg_sd_true_sorted)),y2=np.subtract(np.array(prc_avg_mean_true_sorted),np.array(prc_avg_sd_true_sorted)),alpha=0.2)
plt.fill_between(np.array(x),y1=np.add(np.array(prc_avg_mean_false_sorted),np.array(prc_avg_sd_false_sorted)),y2=np.subtract(np.array(prc_avg_mean_false_sorted),np.array(prc_avg_sd_false_sorted)),alpha=0.2)
plt.xticks(x, parameters)
plt.title('Various Nsnp with AUPRC')  
plt.ylabel('AUPRC')
plt.xlabel('Nsnp')
plt.legend(loc = 'lower right')
plt.xlim([0, max(x)])
plt.ylim([0, 1])

SAVE_PATH="/home/zixshu/DeepGWAS/plot_comparing_Nsnp_weightedfix"
os.makedirs(SAVE_PATH, exist_ok=True)
SAVING_PATH=SAVE_PATH+"/plot_comparing_diff_Nsnp.png"
plt.savefig(SAVING_PATH)






                

        

























