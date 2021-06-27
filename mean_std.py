# Calculate Mean and Std of the images

import os
import cv2
import numpy as np

def calc_avg_mean_std(img_names, img_root, size):
    mean_sum = np.array([0., 0., 0.])
    std_sum = np.array([0., 0., 0.])
    n_images = len(img_names)
    for img_name in img_names:
        img = cv2.imread(img_root + img_name)
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean, std = cv2.meanStdDev(img)
        mean_sum += np.squeeze(mean)
        std_sum += np.squeeze(std)
    return (mean_sum / n_images, std_sum / n_images)

filepath = os.getcwd () + '/images/'
img_names = os.listdir(filepath)
mean, std = calc_avg_mean_std(img_names, filepath, (225,300))
print(f'Train Mean: {mean}')
print(f'Train Std: {std}')
mean = mean / 255.
std = std / 255.
print(f'Train Mean: {mean}')
print(f'Train Std: {std}')