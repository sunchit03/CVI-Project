# --- preprocess.py ---

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_SIZE = (200, 66)
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

def preprocess_data(csv_path):
    data_list = []
    df = pd.read_csv(csv_path, names=column_names)

    for i, address in enumerate(df['center'].values):
        img = cv2.imread(address)
        img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        data_list.append(img)

        if i % 100 == 0:
            print(f'[INFO]: {i} images processed')

    data_list = np.array(data_list)
    label_list = np.array(df['steering'].values)
    return train_test_split(data_list, label_list, test_size=0.2, shuffle=True, random_state=42)
