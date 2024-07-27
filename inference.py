
from Config import *

from Dataloader import *

from model import *

from train import *

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.models import load_model

import os

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

model_age = load_model(age_weights)

model_gender=load_model(gender_weights)

model_age.load_weights(age_weights)

model_gender.load_weights(gender_weights)

images = []
for filename in os.listdir(inferenceDir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(inferenceDir, filename)
        image = Image.open(image_path)
        # preprocess the image
        image = image.resize((224, 224)) # example resizing to 224x224
        image = np.array(image)
        images.append(image)
images = np.array(images)

# Make predictions on the images using the model
A=model_age.predict(images)

G=model_gender.predict(images)
Age=[]
for i in range(0,7):
    predicted_class = np.argmax(A[i])
    age= ['0-3', '15-23', '25-36', '38-48', '4-6', '48-58', '60-100', '8-13']
    z=age[predicted_class]
    Age.append(z)
    
Gender=[]
for i in range(0,7):
    predicted_class = np.argmax(G[i])
    gender= ['Female', 'Male']
    x=gender[predicted_class]
    Gender.append(x)
    


# Display the predicted age and gender for each image
for i in range(len(images)):
    plt.imshow(images[i])
    plt.title("Predicted Age: {} | Predicted Gender: {}".format(Age[i], Gender[i]))
    plt.show()
