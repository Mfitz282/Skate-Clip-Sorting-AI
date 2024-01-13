#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:56:22 2023

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

def frame_generator():
    #use glob function to return a list of all files within VIEDOS_PATH
    video_paths = tf.io.gfile.glob(VIDEOS_PATH)
    np.random.shuffle(video_paths)

    #iterate through all files in the video_paths list 
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        #open the video in cv2
        cap = cv2.VideoCapture(video_path)
        #obtain the frame count
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {num_frames}")
        #calculate the frame rate required to meet the desired sequence length
        sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
        current_frame = 0
        #initialise a varibale with the max number of frames 
        max_images = SEQUENCE_LENGTH
        
        #starts an infinite loop through the frames until conditions are met
        while True:
            #reads the frame from the videe. If 'success' returns false then no more frames are available
            #and the loop breaks
            #cap.red() returns a tuple of a boolean and a frame count, this line assigns these values to 'success'
            #and 'frame'
            success, frame = cap.read()
            if not success:
                break
                
            #check to see if the current frame is divisible by the required frame rate
            if current_frame % sample_every_frame == 0:
                # OPENCV reads in BGR, tensorflow expects RGB so we invert the order
                frame = frame[:, :, ::-1]
                #image must be resized to 299, 299 resolution as a requirement for the inceptions_v3 model
                img = tf.image.resize(frame, (299, 299))
                #preprocessing of the image to meet the requirements for incepction_v3 (e.g scaling pixel values)
                img = tf.keras.applications.inception_v3.preprocess_input(img)
                max_images -= 1
                #produces a tuple of img and video path, with image containing the frame number
                #this way each is unique and it doesn't get lost on the next iteration
                yield img, video_path

            #stop once max_images hits 0 as we now have the 80 images from the video
            if max_images == 0:
                break
            current_frame += 1


#function in TF that creates a dataset using the Python generator function, input is the frame genrator function
#which is producing preprocessed images along with their unique video paths
dataset = tf.data.Dataset.from_generator(frame_generator,
            #output types speicifes the data types of the elements that the generator yields
             output_types=(tf.float32, tf.string),
             output_shapes=((299, 299, 3), ()))

#batches the elements of the dataset into groups of 16 to allow for processing multiple samples simultaneously
#prefetching allows for data loading and model training to overlap which should speed things up
#autotune is a dynamic way of throttling prefetch to ensure system limitations aren't hit 
dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

#initiate the incpetionV3 model, classification head is left of, weights from pretrained data taken from imagenet
inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

#retrieves the ouput tensor from the last layer. this contains the extracted features from the input image
x = inception_v3.output

#adds global average pooling layer to the ouput tensor x, this is a downsampling operation that aims to simplify
#the information while retaining the essence of what makes a feature important 
pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)

#creates a new model that takes the input of the original inceptionV3 model and produces the ouput of the global
#average pooling layer. This new model can be used to obtain feature vectors of a fixed size for input images without
#trying to classify them
feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)

#initialise a variable to track the current video
current_path = None
#initialise an empty list to store all features extracted from a frame
all_features = []

#loop through the dataset, iterating over each batch of images
for img, batch_paths in tqdm.tqdm(dataset):
    #use the feature extraction model on the img
    batch_features = feature_extraction_model(img)
    #reshape the batch of features into a 2D tensor in preparation for saving
    batch_features = tf.reshape(batch_features, 
                              (batch_features.shape[0], -1))
    
    #iterate over the features and video paths in the current batch
    for features, path in zip(batch_features.numpy(), batch_paths.numpy()):
        #check to see if the video path has changed and that current path is not empty (i.e not the first iteration)
        #his means we've completed processing frames for one video
        if path != current_path and current_path is not None:
            # creates an output path for saving the feature to a NumPy file
            output_path = current_path.decode().replace('.MP4', '.npy')
            np.save(output_path, all_features)
            print(f"Saved features to {output_path}")
            #reset the list of features ready for the next video
            all_features = []
        
        #updates the current video path
        current_path = path
        #appebds the features from the current frame to the list
        all_features.append(features)

#creates a binary vector for each label
LABELS = ['Bail','Make','Blank'] 
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

input_directory = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/Lists/'

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
            # yields a batch containing the padded sequence of features and the transformed label
            yield padded_sequence, transformed_label[0]

    return generator

train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
model.fit(train_dataset, epochs=17, callbacks=[tensorboard_callback], validation_data=valid_dataset)

