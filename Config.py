
import os


import pandas as pd


import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


import tensorflow as tf

    
#%matplotlib inline

from tensorflow.keras import layers

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import CSVLogger

###---------------------------------------
### Global  parameters and Hyperparamaters
###---------------------------------------

inpDir = '/content/drive/MyDrive/datasets' # location where input data is stored
outDir = '/kaggle/working/' # location to store outputs where save the model and weights
AgeDir = '/kaggle/input/adience-dataset-preprocessed/datasets/age' # location of the images of data related file
GenderDir = '/kaggle/input/adience-dataset-preprocessed/datasets/gender' # location to data related file
inferenceDir = '/home/dai/Downloads/project-20230223T105805Z-001/project/inference' # location related to this dataset
csv_file_age = "/kaggle/working/training_history_age.csv" #location for log file for Age
csv_file_gender="/kaggle/working/training_history_gender.csv" # location for log file for gender
age_weights="/home/dai/Downloads/project-20230223T105805Z-001/project/saved model/resnet_model_age(1).h5"
gender_weights="/home/dai/Downloads/project-20230223T105805Z-001/project/saved model/resnet_model_gender(1).h5"
#altName = 'dropout_bn_no_bias_leaky_flowers' # Model file name for this alternative

RANDOM_STATE = 24 # for initialization ----- REMEMBER: to remove at the time of promotion to production

tf.random.set_seed(RANDOM_STATE)


EPOCHS = 10   # number of cycles to run

ALPHA = 0.01

BATCH_SIZE = 16

TEST_SIZE = 0.2

IMG_HEIGHT = 224

IMG_WIDTH = 224

FLIP_MODE = "horizontal_and_vertical"

# for rotation transformation 
ROTATION_FACTOR = (-0.1, 0.1) 

FILL_MODE = 'nearest'

ES_PATIENCE = 20 # if performance does not improve stop

LR_PATIENCE = 10 # if performace is not improving reduce alpha

LR_FACTOR = 0.5 # rate of reduction of alpha# Train the model
num_epochs = 10

# Set parameters for decoration of plots
params = {'legend.fontsize' : 'large',
          'figure.figsize'  : (15,6),
          'axes.labelsize'  : 'x-large',
          'axes.titlesize'  :'x-large',
          'xtick.labelsize' :'large',
          'ytick.labelsize' :'large',
         }

CMAP = plt.cm.brg

plt.rcParams.update(params) # update rcParams

Input_Shape = (IMG_HEIGHT, IMG_WIDTH, 3)

data_dir = os.path.join(inpDir, AgeDir)

data_dir2 = os.path.join(inpDir, GenderDir)
