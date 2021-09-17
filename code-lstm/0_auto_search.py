import numpy as np
import os
import shutil
from LSTM_tmp.from_input_to_json import read_data_with_label
from LSTM_tmp.from_input_to_json import a
import bisect
import re
from itertools import product


#  This python routine generates the input files for the grid search of the suitable network structure.
#  This routine reads the input file "para_input.inp"


#-----------------------------------------------------------------------------
def cover_input(key_label,para_select,n_para):
    with open("input-2.inp","a+") as op:
        for i in range(n_para):
            op.write("{label}={a}\n".format(label=key_label[i],a=para_select[i]))
    op.close()

#----------------------------------------------------------------------------
def is_float(istr):
    if istr.count('.') == 1:
        ileft=istr.split('.')[0]
        if ileft.isdigit() and int(ileft) == 0:
            return float(istr)
        else:
            return int(istr)
    if istr.count('.') == 0:
        return int(istr)
			
#------------------------------------------------------------------------------

def step_range(begin,last,n_step,if_int,list_data):
    if if_int == 0:
        n_list=np.arange(begin,last+n_step,n_step)
        n_list=[float(i) for i in n_list]
        list_data.append(n_list)
        return list_data
    if if_int == 1:
        n_list=np.arange(begin,last+n_step,n_step)
        n_list_1=[int(i) for i in n_list]
        list_data.append(n_list_1)
        return list_data
    if if_int == 2:
        list_data.append(begin)
        return list_data

#-------------------------------------------------------------------------------
def b(key):
    key=para_all[str(key)]
    if len(re.findall('[a-zA-Z]+',key)) == 0:
        key=eval(key)
    return key
#-------------------------------------------------------------------------------

def list_combination(d_data):
    all_d_data=[]
    for i in product(*d_data):
        all_d_data.append(list(i))
    return all_d_data

#-------------------------------------------------------------------------------        
def create_all_file(all_data,parm_file,rnn_predict):
    file_number=len(all_data)
    print (file_number)
    path=os.getcwd()
    old_path=os.path.join(path,"LSTM_tmp/input-1.inp")
    input_path=os.path.join(path,parm_file)
    if os.path.exists(input_path) == False:
        os.system("mkdir {a}".format(a=parm_file))
    elif os.path.exists(input_path) == True:
        print ("file:{a} already exists!".format(a=parm_file))
        exit()
    
    os.chdir(input_path)
    for k in range(file_number):
        replace_label=['train','back','layer','neuron']
        key_label=['train_step','look_back','n_layer','n_neuron','rnn_predict']
        file_list=[]
        n_para_1=len(replace_label)
        n_para_2=len(key_label)
        for m in range(n_para_1):
            file_list.append(replace_label[m])
            file_list.append(str(all_data[k][m]))
        all_data_3=all_data[k]
        all_data_3.append(rnn_predict)
        #--------------------------------------
        new_file_name="_".join(file_list)
        os.system("mkdir {a}".format(a=new_file_name))
        new_path=os.path.join(input_path,new_file_name)
        os.chdir(new_path)
        shutil.copyfile(old_path,"input-1.inp")
        os.system("cp -r ../../LSTM_tmp/*.py .")
        os.system("cp -r ../../LSTM_tmp/pop_t .")
        cover_input(key_label,all_data_3,n_para_2)
        os.chdir('..')
        print ("file:{a} has been created!".format(a=new_path))


#----------------------------------------------------------------------------------------
def select_elements(seq, perc):
    return seq[::int(round(1.0/perc))]

#-----------------------------------------------------------------------------------------
para_all=read_data_with_label("para_input.inp")
list_data=[]
train_step=step_range(b('train_step_begin'),b('train_step_last'),b('n_step_of_train'),1,list_data)
look_back=step_range(b('look_back_begin'),b('look_back_last'),b('n_step_of_look_back'),1,train_step)
n_layer=step_range(b('n_layer_begin'),b('n_layer_last'),b('n_step_of_layer'),1,look_back)
n_neuron=step_range(b('n_neuron_begin'),b('n_neuron_last'),b('n_step_of_neuron'),1,n_layer)
random_search=eval(para_all['random_search'])
search_rate=eval(para_all['search_rate'])
rnn_predict=eval(para_all['rnn_predict'])
parm_file=para_all['parm_file']
#------------------------------------------------------------------------------------------
all_parm_list=list_combination(list_data)
if random_search == 0:
    create_all_file(all_parm_list,parm_file,rnn_predict)
if random_search == 1:
    all_parm_list=select_elements(all_parm_list,search_rate)
    create_all_file(all_parm_list,parm_file,rnn_predict)
#-------------------------------------------------------------------------------------------




















