#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:51:38 2024

@author: mfitzpatrick
"""

import os
from sklearn.preprocessing import LabelBinarizer

LABELS = ['Bail', 'Make'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

full_path = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/Lists/Bail/GH012489.MP4'
path = 'Bail/GH012489.MP4'

label = os.path.basename(os.path.dirname(path))

transformed_label = encoder.transform([label])

transformed_label = transformed_label.reshape((2,))

print('label is :' + label)
print(transformed_label)
print(transformed_label.shape)
