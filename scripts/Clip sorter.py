#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:04:02 2023

script for moving video files into sub folders based on their class

@author: mfitzpatrick
"""

import pandas as pd
import os
import shutil

# Load the spreadsheet into a Pandas DataFrame
spreadsheet_path = '/Users/mfitzpatrick/Documents/Notes/SkateFootageDataSet.xlsx'  # Change this to the path of your spreadsheet
df = pd.read_excel(spreadsheet_path)

# Specify the folder containing your video clips
main_video_folder = '/Users/mfitzpatrick/Pictures/GoPro/Test 1'  # Change this to the path of your video folder

# Create a dictionary to store video file labels
video_labels = dict(zip(df['Video_File'], df['Label']))

# Create "make" and "bail" directories within the main video folder
make_directory = os.path.join(main_video_folder, 'Make')
bail_directory = os.path.join(main_video_folder, 'Bail')

os.makedirs(make_directory, exist_ok=True)
os.makedirs(bail_directory, exist_ok=True)

# Iterate through subfolders and video files
for root, dirs, files in os.walk(main_video_folder):
    for video_file in files:
        video_path = os.path.join(root, video_file)

        # Check if the video file is in the dictionary
        if video_file in video_labels:
            label = video_labels[video_file]

            # Move the video file to the corresponding directory
            if label == 'Make':
                destination_folder = make_directory
            elif label == 'Bail':
                destination_folder = bail_directory
            else:
                print(f"Warning: Unknown label '{label}' for video {video_path}")
                continue

            destination_path = os.path.join(destination_folder, video_file)
            shutil.move(video_path, destination_path)
            print(f"Moved video: {video_path} to {destination_path}")
        else:
            print(f"Warning: No label found for video {video_path}")
