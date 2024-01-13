#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:25:31 2024

code to troubleshoot issues with the model code.

@author: mfitzpatrick
"""

import tensorflow as tf
import os
import cv2
import numpy as np
import tqdm
from sklearn.preprocessing import LabelBinarizer

#location for the skate clips
BASE_PATH = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/data'
#glob pattern to locate all the .MP4 files within the subdirectories of the BASE_PATH
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.MP4')
# desired length of the clips, aim to do this to normalise the size of the .npy arrays
SEQUENCE_LENGTH = 80

input_directory = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/Lists/'

#creates a binary vector for each label
LABELS = ['Bail','Make'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

with open(input_directory + 'test_list.txt') as f:
    test_list = [row.strip() for row in list(f)]

with open(input_directory + 'train_list.txt') as f:
    train_list = [row.strip() for row in list(f)]
    


def make_generator(file_list):
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            # constructs the full path to the feature file corresponding to the current path
            full_path = os.path.join(BASE_PATH + '/', path).replace('.MP4', '.npy')
            # extracts the label from the directory name of the current path
            label = os.path.basename(os.path.dirname(path))
            # loads the features from the full path
            features = np.load(full_path)
            # creates a zero-filled array of the dimension (Sequence_length, 2048)
            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            # copies the features into the padded_sequence and pads them with zeros if its length is less than SEQUENCE_LENGTH
            padded_sequence[0:len(features)] = np.array(features)
                
            # transforms the extracted label into a binary vector using the previously defined encoder
            transformed_label = encoder.transform([label])

            # Print the shapes for debugging
            print(f'Original sequence shape: {features.shape}, Padded sequence shape: {padded_sequence.shape}, Transformed label shape: {transformed_label[0].shape}')
            
            # yields a batch containing the padded sequence of features and the transformed label
            yield padded_sequence, transformed_label[0]

    return generator

train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS)-1)))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS)-1)))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
model.fit(train_dataset, epochs=17, callbacks=[tensorboard_callback], validation_data=valid_dataset)