import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense,Dropout
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics as sm



mpl.use('Agg')

#------------------------------------------------------------------
def cal_var(tv_predict,tv_y):
    train_var_1=[]
    n_long=np.shape(tv_predict)[0]
    n_width=np.shape(tv_predict)[1]
    for i in range(n_long):
        for j in range(n_width):
            train_var_1.append(tv_predict[i][j]-tv_y[i][j])
    train_var=(np.array(train_var_1)).var()
    return train_var
    
#----------------------------------------------------------------
def build(new_file_name,n_layer,n_neuron,n_features,n_step_out,train_x,train_y,validation_x,validation_y,epoches,train_file,activation,loss,optimizer,tv_x,tv_y):
    
    train_y=np.reshape(train_y,(-1,n_features))
    validation_y=np.reshape(validation_y,(-1,n_features))
    tv_y=np.reshape(tv_y,(-1,n_features))
    
    model = Sequential()
    print ("\n\n\n-------------------   Model  -----------------")
    if n_layer == 1:
        model.add(LSTM(n_neuron,activation=activation,return_sequences=False,input_shape=(None,n_features)))
    elif n_layer > 1:
        for i in range(n_layer):  
            if i==0:
                model.add(LSTM(n_neuron,activation=activation,return_sequences=True,input_shape=(None,n_features)))
            elif i==n_layer-1:
                model.add(LSTM(n_neuron,activation=activation,return_sequences=False))
            else:
                model.add(LSTM(n_neuron,activation=activation,return_sequences=True))
    model.add(Dense(n_step_out))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer,metrics='accuracy')
    history = model.fit(train_x,train_y,validation_data=(validation_x,validation_y),epochs=epoches,batch_size=50,verbose=0, shuffle=True)
    train_loss=history.history['loss']
    validation_loss=history.history['val_loss']
    #------------------------------------------------------------------------------------------------
    if not os.path.exists(train_file):
	    os.makedirs(train_file)
    os.chdir(train_file)
    #-----------------------------------------------------------------------------------------------
    show_loss(train_loss,validation_loss,epoches)	
    #----------------------------------------------------------------------------------------------
    train_loss=np.reshape(train_loss,(epoches,1))
    validation_loss=np.reshape(validation_loss,(epoches,1))
    n_epoches=np.reshape([int(i) for i in range(1,epoches+1)],(epoches,1))
    append_1=np.append(n_epoches,train_loss,axis=1)
    append_2=np.append(append_1,validation_loss,axis=1)
	
    np.savetxt("train_validation_loss.txt",append_2)
    with open ("train_validation_loss.txt","r+") as f:
        content=f.read()
        f.seek(0,0)
        f.write('#   epoches    train_loss      validation_loss\n'+content)
        f.close()
    print ("Train_loss and validation_loss have been saved in txt file.")
    #---------------------------------------------------------------------------------------------
    
    tv_predict=model.predict(tv_x)
    tv_number=np.shape(tv_x)[0]
    pred_train_loss,pred_train_accuracy = model.evaluate(tv_x,tv_y,verbose=0)
    print ("\n\n\n-----------------   Training Results -------------------------")
    print ("train_loss:",pred_train_loss)
    print ("train_accuracy:",pred_train_accuracy)
    train_var=cal_var(tv_predict,tv_y)
    show_jpg(tv_predict,tv_y,tv_number,'tv',n_features,new_file_name)
    os.chdir('..')
	
    model.save("lstm.h5")


def show_loss(train_loss,validation_loss,epoches):
    epoches=[int(i) for i in range(1,epoches+1)]
    fig = plt.figure(figsize=(4,3))
    fig.set_tight_layout(True)
    plt.plot(epoches,train_loss,"r",label="train_loss")
    plt.plot(epoches,validation_loss,"g",label="validation_loss")
    plt.title("Trian_loss and Validation_loss",fontsize=14)
    plt.xlabel("Epoches",fontsize=14)
    plt.ylabel("Loss",fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig("train_validation_loss.jpg")
    print ("The train_loss and validation_loss have been saved in jpg file.")



def show_jpg(predict,label,number,data_type,n_features,new_file_name):
    n=[int(i) for i in range(1,number+1)]
    fig = plt.figure(figsize=(7,5))
    fig.set_tight_layout(True)
    label=np.reshape(label,(-1,n_features))
    print ("Shape of {a}_y_label:".format(a=data_type),np.shape(label))
    print ("Shape of {a}_y_predict:".format(a=data_type),np.shape(predict))
    for i in range(n_features):
        plt.plot(n,np.reshape(label[:,i],(-1,1)),"r",label="{a}_label{b}".format(a=data_type,b=(i+1)))
        plt.plot(n,np.reshape(predict[:,i],(-1,1)),"g",label="{a}_predict{b}".format(a=data_type,b=(i+1)))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("{a}_data".format(a=data_type),fontsize=14)
    plt.xlabel("Time",fontsize=14)
    plt.ylabel("Population",fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig("{a}_data_predict_{b}.jpg".format(a=data_type,b=new_file_name))
    #--------------------------------------------------------------------------------------------
    number=np.reshape(n,(number,1))
    for j in range(n_features):
        all_y_data=np.append(number,np.append(np.reshape(label[:,j],(-1,1)),np.reshape(predict[:,j],(-1,1)),axis=1),axis=1)
        np.savetxt("{a}_label_predict-input{b}.txt".format(a=data_type,b=(j+1)),all_y_data)

        with open ("{a}_label_predict-input{b}.txt".format(a=data_type,b=(j+1)),"r+") as f:
            content=f.read()
            f.seek(0,0)
            f.write('#   n_{a}   {a}_label-features{c}    {a}_y_predict-features{c}\n'.format(a=data_type,c=(j+i))+content)
            f.close()
    print ("{a}_y_label and {a}_y_predict have been saved in txt and jpg files.".format(a=data_type))
#----------------------------------------------------------------------------------------------------------------


def ordinary_jpg(n_features,data,data_type):
    fig = plt.figure(figsize=(4,3))
    fig.set_tight_layout(True)
    for i in range(n_features):
        data_i=[x[i] for x in data]
        plt.plot(data_i,label="variance_{a}".format(a=i))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("{a}_variance".format(a=data_type),fontsize=14)
    plt.xlabel("Time",fontsize=14)
    plt.ylabel("Variance",fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig("{a}_variance.jpg".format(a=data_type),dpi=300)

#--------------------------------------------------------------------------------------------------------------










