# --- augment.py ---

import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def flip_image_and_angle(image, angle):
    flipped_image = cv2.flip(image, 1)
    flipped_angle = -angle
    return flipped_image, flipped_angle

def adjust_brightness_cv2(image):
    img_uint8 = (image * 255).astype(np.uint8)
    scale = np.random.uniform(0.4, 1.6)
    bright = cv2.multiply(img_uint8, np.ones(img_uint8.shape, dtype="uint8"), scale=scale)
    bright = bright.astype(np.float32) / 255.0
    return bright

def data_augmentation(X_train, y_train):
    augmented_images = []
    augmented_angles = []

    for i in range(len(X_train)):
        image = X_train[i]
        angle = y_train[i]
        augmented_images.append(image)
        augmented_angles.append(angle)

        if random.random() < 0.5:
            flipped_image, flipped_angle = flip_image_and_angle(image, angle)
            augmented_images.append(flipped_image)
            augmented_angles.append(flipped_angle)

        if random.random() < 0.5:
            bright_image = adjust_brightness_cv2(image)
            augmented_images.append(bright_image)
            augmented_angles.append(angle)

    X_train_aug = np.array(augmented_images)
    y_train_aug = np.array(augmented_angles)

    aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )
    aug.fit(X_train_aug)

    return X_train_aug, y_train_aug, aug