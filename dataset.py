import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.6921719, 0.66398524, 0.66453462]
std = [0.2412245, 0.24936252, 0.24726714]

class DatasetAttributes():
    def __init__(self, annotation_path):
        neck_labels = []
        sleeve_labels = []
        pattern_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                neck_labels.append(row['neck'])
                sleeve_labels.append(row['sleeve_length'])
                pattern_labels.append(row['pattern'])

        self.neck_labels = np.unique(neck_labels)
        self.sleeve_labels = np.unique(sleeve_labels)
        self.pattern_labels = np.unique(pattern_labels)

        self.num_necks = len(self.neck_labels)
        self.num_sleeves = len(self.sleeve_labels)
        self.num_patterns = len(self.pattern_labels)

        self.neck_id_to_name = dict(zip(range(len(self.neck_labels)), self.neck_labels))
        self.neck_name_to_id = dict(zip(self.neck_labels, range(len(self.neck_labels))))

        self.sleeve_id_to_name = dict(zip(range(len(self.sleeve_labels)), self.sleeve_labels))
        self.sleeve_name_to_id = dict(zip(self.sleeve_labels, range(len(self.sleeve_labels))))

        self.pattern_id_to_name = dict(zip(range(len(self.pattern_labels)), self.pattern_labels))
        self.pattern_name_to_id = dict(zip(self.pattern_labels, range(len(self.pattern_labels))))


class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.neck_labels = []
        self.sleeve_labels = []
        self.pattern_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['filename'])
                self.neck_labels.append(self.attr.neck_name_to_id[row['neck']])
                self.sleeve_labels.append(self.attr.sleeve_name_to_id[row['sleeve_length']])
                self.pattern_labels.append(self.attr.pattern_name_to_id[row['pattern']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'neck_labels': self.neck_labels[idx],
                'sleeve_labels': self.sleeve_labels[idx],
                'pattern_labels': self.pattern_labels[idx]
            }
        }
        return dict_data