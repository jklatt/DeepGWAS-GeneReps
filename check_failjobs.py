import os



PATH="/home/zixshu/DeepGWAS/out_try_biggermlpfor2000"
filenames=os.listdir(PATH)
count=0
for name in filenames:
    # if "2000" in name:
    if "lr0.003" in name:
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