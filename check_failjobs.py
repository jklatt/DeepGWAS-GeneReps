import os



PATH="/home/zixshu/DeepGWAS/output_twostep_leakyrelu_mlp_lr0.00085_upsampling"
filenames=os.listdir(PATH)
count=0
for name in filenames:
    filepath=PATH+"/"+name
    with open(filepath) as f:
        file_text=f.readlines()
    
    # if "ValueError" in file_text:
    for line in file_text:
        if "ValueError" in line:
            print(name)
            count+=1

print("the number of failed job is", count)
