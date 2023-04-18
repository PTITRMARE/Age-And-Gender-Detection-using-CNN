
from Config import *

from Dataloader import *

from model import *



#train the data according to the age and gender 

train_ds_age,test_ds_age,class_name_age=dataloader(data_dir)  # for the data of age

train_ds_gender,test_ds_gender,class_name_gender=dataloader(data_dir2) # for the data of gender

resnet_model_age=model(Input_Shape,class_name_age,train_ds_age,test_ds_age) # definding the age model

resnet_model_gender=model(Input_Shape,class_name_gender,train_ds_gender,test_ds_gender) # defing the gender model

compilemodel(resnet_model_age,train_ds_age,test_ds_age,csv_file_age) # compile the age model according to the given value 

compilemodel(resnet_model_gender,train_ds_gender,test_ds_gender,csv_file_gender) # compile the gender model according to the given value

resnet_model_age.save('resnet_model_age.h5') # saving the age_model

resnet_model_gender.save('resnet_model_gender.h5') #saving the gender model 
