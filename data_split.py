import os 
import csv 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from PIL import Image

all_data = []
# input_folder = '/content/drive/MyDrive/Flix Stock/classification-assignment/'
input_folder = os.getcwd()

with open('final_attributes.csv') as csv_file:
    # parse it as CSV
    reader = csv.DictReader(csv_file)
    # tqdm shows pretty progress bar
    # each row in the CSV file corresponds to the image
    for row in tqdm(reader, total=reader.line_num):
        # we need image ID to build the path to the image file
        img_id = row['filename']
        # Using all three attributes
        neck = row['neck']
        sleeve_length = row['sleeve_length']
        pattern = row['pattern']
        img_name = os.path.join(input_folder, 'images', str(img_id))
        # check if file is in place
        if os.path.exists(img_name):
            # check if the image has 225*300 pixels with 3 channels
            # print("Hi")
            img = Image.open(img_name)
            if img.size == (225, 300) and img.mode == "RGB":
                all_data.append([img_name, neck, sleeve_length, pattern])
        else:
            print("Something went wrong: there is no file ", img_name)

# Set the seed to reproduce the findings later
np.random.seed(99)

# construct a Numpy array from the list
all_data = np.asarray(all_data)

# Take 1782 samples in random order
inds = np.random.choice(1782, 1782, replace=False)

# split the data into train/val and save them as csv files
train = all_data[inds][:1500]
val = all_data[inds][1500:]
train_df = pd.DataFrame(train)
train_df.columns=["filename", "neck", "sleeve_length", "pattern"]
val_df = pd.DataFrame(val)
val_df.columns=["filename", "neck", "sleeve_length", "pattern"]

# save the dataframe as a csv file
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)