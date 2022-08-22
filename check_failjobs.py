import os



PATH="/home/zixshu/DeepGWAS/output_lr0.0004"
filenames=os.listdir(PATH)
count=0
for name in filenames:
    if "2000" not in name:
    # if "lr0.003" in name:
        filepath=PATH+"/"+name
        with open(filepath) as f:
            file_text=f.readlines()
        
        for line in file_text:
            if "ValueError" in line:
                print(name)
                count+=1

print("the number of failed job is", count)


#24fail 2000 lr 0.02
#22fail 2000 lr 0.0002
#22fail 2000 lr 0.00002
#0.0003 or 0.0004 lr 2000 fail 40 in two seeds
# 43 fail with 0.00015