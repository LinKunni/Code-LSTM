import numpy as np
from create_model_and_train import show_jpg,ordinary_jpg
import tensorflow as tf
import os
import copy
from tensorflow.keras.models import load_model



def rnn_test(test_file,n_fea,look_back,test_x,test_y,test_number,model,new_file_name):
    test_y=np.reshape(test_y,(-1,n_fea))
    pop_predict=np.zeros(np.shape(test_y))
    test_tmp = test_x[0]
    test_loss,test_accuracy = model.evaluate(test_x,test_y,verbose=0)
    print ("\n\n\n----------------   Testing Results  -----------------")
    print ("test_loss:",test_loss)
    print ("test_accuracy:",test_accuracy)

    for i in range(test_number):
        new_test_x=np.reshape(test_tmp,(1,look_back-1,n_fea))
        y_predict=model.predict(new_test_x)
        pop_predict[i]=y_predict
        test_tmp=np.append(test_tmp[1:,:],y_predict,axis=0)

    if not os.path.exists(test_file):
        os.makedirs(test_file)

    os.chdir(test_file)
    show_jpg(pop_predict,test_y,test_number,'test',n_fea,new_file_name)
    os.chdir('..')
    print ("\n\n\n--------------------  Finish   -----------------")
    print ("Successfully!")
#--------------------------------------------------------------------------------------------

