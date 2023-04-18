#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:36:36 2023

@author: dai
"""

from Config import *

from Dataloader import *


def model(Input_Shape,n_classes,train_ds,test_ds):
    ## Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    #from keras.engine import sequential
    resnet_model= tf.keras.Sequential()

    pretraind_model= tf.keras.applications.ResNet50(include_top=False,
                                                    input_shape=Input_Shape,
                                                    pooling='avg',
                                                    classes=n_classes,
                                                    weights="imagenet")

    for layer in pretraind_model.layers:
        layer.trainable=False

    resnet_model.add(pretraind_model)
    resnet_model.add(tf.keras.layers.Flatten())
    resnet_model.add(tf.keras.layers.Dense(512,activation="relu"))
    resnet_model.add(tf.keras.layers.Dense(n_classes,activation="softmax"))
    
    
    resnet_model.summary()
    
    return resnet_model

def compilemodel(model,train_ds,test_ds,csv_file):
    
    csv_logger = CSVLogger(filename=csv_file, separator=",", append=False)
    
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    history=model.fit(train_ds,
                     epochs=EPOCHS,
                     validation_data=test_ds,
                     callbacks=[csv_logger],
                     batch_size=BATCH_SIZE)
