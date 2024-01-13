#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:10:55 2023

code for pre processing video clips

didn't run the first time because I didn't have cv2 installed

ran but failed the second time, think its becasue .mp4 vs .MP4

installed cv2 and then realised that the ouput folder was named incorrectly  

OLD CODE: Just produced JPEGS of the frames i.e no good for actually feeding into a CNN
"""

import cv2
import os
import numpy as np
import pandas as pd

# Define input and output directories
output_dir = '/Users/mfitzpatrick/Pictures/GoPro/Test 1/Test Output'
input_dir = '/Users/mfitzpatrick/Pictures/GoPro/Test 1'

# Define the size of the output frames
output_size = (172, 172)

# Define the frame skipping interval (e.g., 2 for every 2nd frame, 3 for every 3rd frame)
frame_skip_interval = 3

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
        if video_file.lower().endswith(('.mp4', '.avi')) and video_file in video_labels:
            label = video_labels[video_file]

            # Open the video file
            video = cv2.VideoCapture(video_path)

            # Check if the video file is successfully opened
            if not video.isOpened():
                print(f"Error opening video file: {video_path}")
                continue

            # Initialize an empty list to store frames
            frames = []

            # Loop over all frames in the video
            frame_count = 0
            while True:
                # Read the next frame from the video
                ret, frame = video.read()
                if not ret:
                    break

                # Check if the current frame should be skipped
                if frame_count % frame_skip_interval != 0:
                    frame_count += 1
                    continue

                # Resize the frame to the output size
                resized_frame = cv2.resize(frame, output_size)

                # Append the resized frame to the list
                frames.append(resized_frame)

                # Increment the frame count
                frame_count += 1

            # Release the video file
            video.release()

            # Convert the list of frames to a NumPy array
            video_cube = np.stack(frames, axis=0)  # Stack frames along the first axis

            # Save the 3D cube to a NumPy file with the label
            output_filename = os.path.splitext(video_file)[0] + f'_{label}_3d_cube.npy'
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, {'data': video_cube, 'label': label})

            print(f"3D cube saved to: {output_path} with label: {label}")

print("Processing completed.")

