import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

seeds=range(5)
METRIC_PATH="/home/zixshu/DeepGWAS/baseline/metrics_fourmoments"
for seed in seeds:
    PATH=METRIC_PATH+"/"+int(seed)
    filenames=os.listdir(PATH)
    for file in filenames:
        if "snp200_" in file:
            FILE_PATH=PATH+"/"+file
            with open(FILE_PATH, "rb") as f:
                evaluation_dict=pickle.load(f)
















