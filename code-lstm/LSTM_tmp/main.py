import numpy as np
from read_pop import get_s0_time_series
from read_pop import save_time_series
from create_model_and_train import build
from create_model_and_train import show_loss
from create_model_and_train import show_jpg
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from from_input_to_json import a,output_parm
import os
from prediction import rnn_test

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
with tf.device('/CPU:0'):

	output_parm()
	file_list=[]
	file_list.append(str(a('train_step')))
	file_list.append(str(a('look_back')))
	file_list.append(str(a('n_layer')))
	file_list.append(str(a('n_neuron')))
	print (file_list)
	new_file_name="_".join(file_list)

	train_x,train_y,validation_x,validation_y,test_x,test_y,test_number,tv_x,tv_y,model_x,model_y=get_s0_time_series(a('first_step'),a('train_step'),a('last_step'),a('look_back'),a('validation_ratio'),a('n_features'))
	save_time_series(dataset_x=train_x,dataset_y=train_y,file_name='train',series_file=a('series_file'),n_features=a('n_features'))
	save_time_series(dataset_x=validation_x,dataset_y=validation_y,file_name='validation',series_file=a('series_file'),n_features=a('n_features'))
	save_time_series(dataset_x=test_x,dataset_y=test_y,file_name='test',series_file=a('series_file'),n_features=a('n_features'))

	#----------------------------------------------------------------------------------------

	build(new_file_name,a('n_layer'),a('n_neuron'),a('n_features'),a('n_step_out'),train_x,train_y,validation_x,validation_y,a('epoches'),a('train_file'),a('activation'),a('loss'),a('optimizer'),tv_x,tv_y)

	model=load_model("lstm.h5")
	model_y=np.reshape(model_y,(-1,a('n_features')))
	test_model_loss,test_model_accuracy = model.evaluate(model_x,model_y,verbose=0)
	print ("\n\n\n----------------   Model Testing  -----------------")
	print ("test_model_loss:",test_model_loss)
	print ("test_model_accuracy:",test_model_accuracy)


	#-------------------------------------------------------------------------------------------
	if a('rnn_predict') == 1:
	    rnn_test(a('test_file'),a('n_features'),a('look_back'),test_x,test_y,test_number,model,new_file_name)
	#--------------------------------------------------------------------------------------------
