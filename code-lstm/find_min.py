import numpy as np
import copy
import os
import math


#####    This python routine is a typical example used for searching the suitable LSTM NN structures with small loss.  


def find_index(l1):
    l2 = copy.deepcopy(l1) 
    result = list() 
    d = dict()  
    l1.sort() 
    for index, value in enumerate(l2):
        d[value] = index 
    for i in l1:
        result.append(d[i])
    return result,l1

def find_word():
    with open("output.log","r") as op:
        ops=op.readlines()
        for i,l in enumerate(ops):
            if "test_model_loss" in l:
                test_word=ops[i].split(':')[1]
    return test_word


def string_switch(x,y,z,s=1):
    with open(x, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(x, "w", encoding="utf-8") as f_w:
        n = 0
        if s == 1:
            for line in lines:
                if y in line:
                    line = line.replace(y,z)
                    f_w.write(line)
                    n += 1
                    break
                f_w.write(line)
                n += 1
            for i in range(n,len(lines)):
                f_w.write(lines[i])
        elif s == 'g':
            for line in lines:
                if y in line:
                    line = line.replace(y,z)
                f_w.write(line)

#--------------------------------------------------
path=os.getcwd()
for file_i in os.listdir(path):
    if (file_i.split('_')[0]) == 'train': 
        path_1=os.path.join(path,file_i)
        os.chdir(path_1)
        test_word=find_word()
        os.chdir("..")
        with open("test_model_loss.txt", "a+") as tf:
            tf.write("{a}:{b}".format(a=file_i,b=test_word)) 
#------------------------------------------------------------
with open ("test_model_loss.txt", "r") as tf:
    tfs=tf.readlines()
all_loss=[]
for line in tfs:
    all_loss.append(float((line.split(":")[1]).strip()))
new_index,new_loss=find_index(all_loss)
#--------------------------------------------------------

mean_first=np.mean([math.log10(x) for x in new_loss[0:20]])
choose_index=[]
for k in range(20):
    if new_loss[k] < 10**mean_first:
        choose_index.append(k)
        print (tfs[int(new_index[k])])
print ("number of param:",len(choose_index))
if len(choose_index) == 0:
    print ("There is no good model!")
elif len(choose_index) > 0:
    if_input=input("The reasonable sets of parameters have been output!")
#-------------------------------------------------------------------------------------------


    

    



    
    
    
    
