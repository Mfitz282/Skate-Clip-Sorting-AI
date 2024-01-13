#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:01:25 2023

script for creating video file labels from a spreadsheet

@author: mfitzpatrick
"""

import pandas as pd
import os

# Load the spreadsheet into a Pandas DataFrame
# use the two columns: 'Video_File' and 'Label'
spreadsheet_path = '/Users/mfitzpatrick/Documents/Notes/SkateFootageDataSet.xlsx'  # Change this to the path of your spreadsheet
df = pd.read_excel(spreadsheet_path)


# Specify the folder containing your video clips
main_video_folder = '/Users/mfitzpatrick/Pictures/GoPro'  # Change this to the path of your video folder

# Create a dictionary to store video file labels
video_labels = dict(zip(df['Video_File'], df['Label']))

# Iterate through subfolders and video files
for root, dirs, files in os.walk(main_video_folder):
    for video_file in files:
        video_path = os.path.join(root, video_file)

        # Check if the video file is in the dictionary
        if video_file in video_labels:
            label = video_labels[video_file]
            print(f"Video: {video_path}, Label: {label}")
        else:
            print(f"Warning: No label found for video {video_path}")

