# Creates a clean_attributes.csv file corresponding to the images available
import os
import csv
import numpy as np 
import pandas as pd 
from natsort import natsorted 

# Remove the Thumbs.db file 
thumbs_path = './images/Thumbs.db'
if os.path.exists(thumbs_path):
  os.system(f'rm {thumbs_path}')
else:
  pass
# Number of files in the images folder
images_path = os.getcwd() + '/images'
img_names = os.listdir(images_path)
img_names = natsorted(img_names)
print(f'Total Images in the folder: {len(img_names)}')

images_matched = 0 
# fields = []
new_rows = []
with open("attributes.csv", "r") as f:
    header_reader = csv.reader(f)
    # fields.append(next(header_reader))
    new_rows.append(next(header_reader))

filename = "attributes.csv"
for i in img_names: 
  with open(filename,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader: 
      if row[0] == i:
        # print(row)
        images_matched +=1
        new_rows.append(row)
        break 

# print(f'Total Images Matched : {images_matched}')
print(f"New Rows' Length: {len(new_rows)}")

with open('clean_attributes.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    # writing the data rows 
    csvwriter.writerows(new_rows)

# load and summarize the dataset

# load the dataset
dataset = pd.read_csv('clean_attributes.csv', header='infer')

# summarize the dataset
print(dataset.describe())

# Count the number of Nans 
nans = dataset.isnull().sum(axis = 0)
print('_'*50)

print('Number of NaNs:')
print(nans)

# Replace the NaNs with the median values 

dataset['neck'] = dataset['neck'].replace(np.NaN, dataset['neck'].median()).astype(int)
dataset['sleeve_length'] = dataset['sleeve_length'].replace(np.NaN, dataset['sleeve_length'].median()).astype(int)
dataset['pattern'] = dataset['pattern'].replace(np.NaN, dataset['pattern'].median()).astype(int)
dataset.to_csv('final_attributes.csv', index=False)
os.system('rm clean_attributes.csv')