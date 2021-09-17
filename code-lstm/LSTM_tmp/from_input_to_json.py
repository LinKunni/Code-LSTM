import numpy as np
import copy
import os
import shutil
import json
import re


def create_dir(path):
    try:
        shutil.rmtree(path)
        os.makedirs(path)
    except:
        os.makedirs(path)


def read_data_with_label(filename):
    inp = {}
    fp = open(filename, 'r')
    for line in fp:
        if line.strip() != '':
            if "=" in line.strip():
                string_read = line.strip().split('=', 1)
                string_zero = string_read[0].replace(' ', '')
                strong_one = string_read[1].replace(' ', '')
                inp[string_zero] = strong_one
    return inp

def dump_json(filename, obj):
    """
    dump an object in json format.
    """
    json.encoder.FLOAT_REPR = lambda f: format("%.18g" % f)
    fp = open(filename, mode='w')
    my_str = json.dumps(obj, indent=2)
    fp.write(my_str)
    fp.close()
    return

def load_json(filename):
    """
    load an object in json format.
    """
    json.encoder.FLOAT_REPR = lambda f: format("%.18g" % f)
    fp = open(filename, mode='r')
    obj = json.load(fp)
    return obj


def input_to_json_file (input_file_1, input_file_2, json_file) :
    xxx_1 = read_data_with_label (input_file_1)
    xxx_2 = read_data_with_label (input_file_2)
    xxx=dict(xxx_1, ** xxx_2)
    dump_json(json_file, xxx) 

def read_from_json (json_file) :
    para_all = load_json(json_file)
    return para_all

def pro_wf() : 
    input_file_1 = 'input-1.inp'
    input_file_2 = 'input-2.inp'
    json_file = 'input.json'
    input_to_json_file (input_file_1, input_file_2, json_file)
    para_all = read_from_json (json_file)
    return para_all

def a(key):
    para_all=pro_wf()
    key=para_all[str(key)]
    if len(re.findall('[a-zA-Z]+',key)) == 0:
        key=eval(key)
    return key

def output_parm():
    para_all=pro_wf()
    print ("\n\n---------------    Parameter   --------------")
    for kk,vv in para_all.items():
        print ("{a} = {b}".format(a=kk,b=vv))
    #print ("---------------------------------------------")

	

if __name__ == '__main__':
	para_all=pro_wf()
	print (para_all)
	



