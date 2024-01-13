#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:22:04 2023

clip compress script, takes the input directory, iterates through it and takes any
.mp4 files, loads them 1 by 1 and saves them as a new reduced sized version

@author: mfitzpatrick
"""

import os
from moviepy.editor import VideoFileClip

def downscale_video(input_path, output_path, scale_factor):
    # Load the video clip
    video_clip = VideoFileClip(input_path)

    # Downscale the video clip
    downscaled_clip = video_clip.resize(scale_factor)

    # Write the downscaled video to the output file
    downscaled_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Close the video clip objects
    video_clip.close()
    downscaled_clip.close()

def process_videos(input_folder, output_folder, scale_factor):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all .mp4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".MP4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"compressed_{filename}")
            
            # Downscale the video
            downscale_video(input_path, output_path, scale_factor)

# Example usage: Process all .mp4 videos in the "input_folder" and save the results in "input_folder/compressed"
input_folder = "/Users/mfitzpatrick/Pictures/GoPro/Test 1/Make"
output_folder = os.path.join(input_folder, "compressed")
scale_factor = 0.25

process_videos(input_folder, output_folder, scale_factor)

        
    
        