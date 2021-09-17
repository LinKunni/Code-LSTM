import numpy as np
import from_input_to_json
import os 
import pandas as pd
#from collections import Counter
import random


def get_s0_time_series(first_step,train_step,last_step,look_back,validation_ratio,n_features):
    ss=np.loadtxt("pop_t")
    s0_pop=ss[:,1:(n_features+1)]
    dt=ss[1,0]-ss[0,0]
    all_time_series_x=[]
    all_time_series_y=[]
    all_time_series_number=last_step-look_back-first_step+2
    for i in range(all_time_series_number):
        all_time_series_x.append(list(s0_pop[first_step+i:first_step+look_back+i-1]))
        all_time_series_y.append(list(s0_pop[first_step+look_back+i-1:first_step+look_back+i]))
	
    train_and_validation_all=train_step-look_back-first_step+2
    n_validation_test_model=round(train_and_validation_all*(1/4))
    train_and_validation=train_and_validation_all-n_validation_test_model
    train_number=round(train_and_validation*(1-validation_ratio))
    validation_number=train_and_validation-train_number
    test_number=len(all_time_series_y)-train_and_validation_all

    print ("\n\n\n--------------------   Series  -------------------------")
    print ("train_and_validation_and_model_test:",train_and_validation_all)
    print ("train_and_validation:",train_and_validation)
    print ("n_validation_test_model:",n_validation_test_model)
    print ("train_number:",train_number)
    print ("validation_number:",validation_number)
    print ("test_number:",test_number)
    #---------------------------------------------------------------------------------------------- 
    
    n_choice=[k for k in range(train_and_validation)]
    random.shuffle(n_choice)

    all_time_series_x_train=[]
    all_time_series_y_train=[]
    all_time_series_x_validation=[]
    all_time_series_y_validation=[]

    for j in range(train_number):
            all_time_series_x_train.append(all_time_series_x[n_choice[j]])
            all_time_series_y_train.append(all_time_series_y[n_choice[j]])
    for k in range(train_number,train_and_validation):
            all_time_series_x_validation.append(all_time_series_x[n_choice[k]])
            all_time_series_y_validation.append(all_time_series_y[n_choice[k]])

    #------------------------------------------------------------------------------------------------
    s0_train_series_x=np.reshape(all_time_series_x_train,(train_number,look_back-1,n_features))
    s0_train_series_y=np.reshape(all_time_series_y_train,(train_number,1,n_features))
    s0_validation_series_x=np.reshape(all_time_series_x_validation,(validation_number,look_back-1,n_features))
    s0_validation_series_y=np.reshape(all_time_series_y_validation,(validation_number,1,n_features))
    s0_test_series_x=np.reshape(all_time_series_x[train_and_validation_all:],(test_number,look_back-1,n_features))
    s0_test_series_y=np.reshape(all_time_series_y[train_and_validation_all:],(test_number,1,n_features))
    tv_test_series_x=np.reshape(all_time_series_x[0:train_and_validation],(train_and_validation,look_back-1,n_features))
    tv_test_series_y=np.reshape(all_time_series_y[0:train_and_validation],(train_and_validation,1,n_features))
    validation_test_x=np.reshape(all_time_series_x[train_and_validation:train_and_validation_all],(n_validation_test_model,look_back-1,n_features))
    validation_test_y=np.reshape(all_time_series_y[train_and_validation:train_and_validation_all],(n_validation_test_model,1,n_features))
    print ("Shape of train-set-x and validation-set-x:",np.shape(tv_test_series_x))
    print ("Shape of train-set-y and validation-set-y:",np.shape(tv_test_series_y))
    return s0_train_series_x,s0_train_series_y,s0_validation_series_x,s0_validation_series_y,s0_test_series_x,s0_test_series_y,test_number,tv_test_series_x,tv_test_series_y,validation_test_x,validation_test_y



def save_time_series(dataset_x,dataset_y,file_name,series_file,n_features):
    data_number_x = np.shape(dataset_x)[0]
    data_length_x = np.shape(dataset_x)[1]
	
    data_number_y = np.shape(dataset_y)[0]
    data_length_y = np.shape(dataset_y)[1]
	

    if not os.path.exists(series_file):
        os.makedirs(series_file)
    

    order=np.reshape([int(i) for i in range(1,data_number_x+1)],(data_number_x,1))
    for i in range(n_features):
        filename_txt=file_name+"_input{a}.txt".format(a=(i+1))
        filename_csv=file_name+"_input{a}.csv".format(a=(i+1))
        csv_path = os.path.join(series_file,filename_csv)
	
        dataset_x_a=np.reshape(dataset_x[:,:,i],(-1,data_length_x))
        dataset_y_a=np.reshape(dataset_y[:,:,i],(-1,data_length_y))
	
        x_y=np.append(dataset_x_a,dataset_y_a,axis=1)
        order_x_y=np.append(order,x_y,axis=1)


        df = pd.DataFrame(x_y)
        df.to_csv(csv_path, index=False)

        os.chdir(series_file)	
        np.savetxt(filename_txt,order_x_y,fmt='%.14f')
        with open (filename_txt,"r+") as f:
            content=f.read()
            f.seek(0,0)
            f.write('#  n_{a}   {a}_x1_to_{a}_x{b}-features{c}    {a}_y\n'.format(a=file_name,b=data_length_x,c=(i+1))+content)
            f.close()
        os.chdir('..')
    print (file_name,"Time-series have been saved in txt and csv files.")







