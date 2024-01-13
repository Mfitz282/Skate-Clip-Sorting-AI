#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:22:16 2023

@author: mfitzpatrick
"""

import os
import random

video_directory = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/data'
output_directory = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/data'

# Function to get a list of .MP4 files in a directory with subfolder information
def get_mp4_files_with_subfolder(directory, subfolder):
    return [os.path.join(subfolder, f) for f in os.listdir(os.path.join(directory, subfolder)) if f.lower().endswith('.mp4')]

# Function to split videos into training and testing sets
def split_videos(directory, train_ratio=0.8):
    bail_videos = get_mp4_files_with_subfolder(directory, 'Bail')
    make_videos = get_mp4_files_with_subfolder(directory, 'Make')

    # Calculate the number of videos for training from each category
    num_bail_train = int(len(bail_videos) * train_ratio)
    num_make_train = int(len(make_videos) * train_ratio)

    # Randomly select videos for training
    train_bail = random.sample(bail_videos, num_bail_train)
    train_make = random.sample(make_videos, num_make_train)

    # The remaining videos are for testing
    test_bail = list(set(bail_videos) - set(train_bail))
    test_make = list(set(make_videos) - set(train_make))

    return train_bail, train_make, test_bail, test_make

# Write lists to text files
def write_list_to_txt(file_path, video_list):
    with open(file_path, 'w') as file:
        for video in video_list:
            file.write(video + '\n')

# Create lists
train_bail, train_make, test_bail, test_make = split_videos(video_directory)

# Write training and testing lists to text files
train_txt_path = os.path.join(output_directory, 'train_list.txt')
test_txt_path = os.path.join(output_directory, 'test_list.txt')

write_list_to_txt(train_txt_path, train_bail + train_make)
write_list_to_txt(test_txt_path, test_bail + test_make)

