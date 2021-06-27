# Flixstock Deep Learning Assignment
### Problem Statement: Predicting Deep Fashion Attributes
##### Folder structure
Directory: ``` cd /Users/abhijeetdhupia/Documents/Flix Stock/classification-assignment```
```

classification-assignment
.
├── images
│── attributes.csv
│── final_attributes.csv
│── train.csv
│── test.csv
│── data_preprocess.py
│── data_split.py
├── dataset.py
├── model.py
├── train.py
└── test.py
```

##### Dataset Observations and Preprocessing: 
1. #Images in the ```attributes.csv``` > #Images in the ```images``` folder. 
2. One row was listed twice in ```attributes.csv```
3. Here, median = mode and the most common attributes were, ```neck = 6, sleeve_length = 3``` and ```pattern = 9```
4. #N/A values were replaced by the mode values of each column. 

#####  Steps: 
1. To clean the attributes.csv file run and take only the relevant information. Run the below code, it will create the ```final_attributes.csv``` file.
```python data_preprocess.py ```
2. Now, the dataset is randomly split into ```train.csv``` and ```val.csv```. Run the below code ``` python data_split.py```
3. Next, we will calculate the mean and standard deviation of images for normalize the dataset by running ```python mean_std.py```
3. ```dataset.py``` contains functions to prepare the dataset for training by getting the unique number of labels along with the ground truths of each images. 
4. ```model.py``` contains the MobileNetV2 architecture with modified last layer to give three output for Neck Type, Sleeve Length and Pattern. 
5. ```python train.py``` will train the model. It will also create ```checkpoints``` and ```logs``` directories where it saves the weightfiles and logs. 
6. ```test.py``` will test the trained model on ```val.csv```. The weightfile address is required and it can be given as, ```python test.py --checkpoint ./checkpoints/2021-06-27_20-45/checkpoint-000100.pth```