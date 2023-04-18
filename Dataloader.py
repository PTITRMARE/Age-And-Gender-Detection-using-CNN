#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:05:35 2023

@author: dai
"""
from Config import *

def dataloader(data_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=TEST_SIZE,
        subset='training',
        seed= RANDOM_STATE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=TEST_SIZE,
        subset='validation',
        seed= RANDOM_STATE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
    
    class_names = train_ds.class_names
    print('total ', len(class_names), class_names)
    class_name_age=len(class_names)
    
    #Autoyunning for ## Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    
    
    return train_ds,test_ds,class_name_age




