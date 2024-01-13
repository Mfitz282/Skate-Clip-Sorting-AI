#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:17:17 2023

@author: mfitzpatrick
"""

import numpy as np
import matplotlib.pyplot as plt

# Assuming GH012587_3d_cube.npy is the file containing your 3D cube
file_path = '/Users/mfitzpatrick/Pictures/GoPro/Test 1/Test Output/GH012587_3d_cube.npy'

# Load the 3D cube from the file
video_cube = np.load(file_path)

# Print the shape of the 3D cube
print(video_cube.shape)



# Select frames to visualize
frames_to_visualize = [0, 10, 20]  # Adjust the frame indices as needed

# Plot the selected frames
for i in frames_to_visualize:
    plt.imshow(video_cube[i, :, :, :])
    plt.title(f"Frame {i}")
    plt.show()
