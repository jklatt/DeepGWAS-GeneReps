import os



PATH="/home/zixshu/DeepGWAS/output_twostep_leakyreulu_mlp_lr0.0008_upsampling_withthreesetting_threefails20200"
filenames=os.listdir(PATH)
count=0
for name in filenames:
    # if "2000" not in name:
    filepath=PATH+"/"+name
    with open(filepath) as f:
        file_text=f.readlines()
    
    for line in file_text:
        if "ValueError" in line:
            print(name)
            count+=1

print("the number of failed job is", count)
