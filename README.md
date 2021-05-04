# Abuse Detection System
---
Provide a Deep Learning-Based real-time solution for nursing homes and hospitals for detecting cases of abuse in the elderly population by analyzing security camera frames and performing real-time forecasting using three machine learning models YOLO, DeepSort, ADS

## Table of contents
---
* [Project highlights](#Project-highlights)
* [Introduction ](#Introduction)
* [Data collection](#Data-collection)
* [Model training && Evaluation](#Model-training-&&-Evaluation)
* [Model architecture](#Model-architecture)
* [ADS PIPELINE Demo](#ADS-PIPELINE-Demo)
* [System overview](#System overview)
* [Setup](#setup)
* [Input-Output examples](#Input-Output-examples)
* [Reference](#Reference)


## Project highlights
---
-	Creating a new novel dataset ADS-dataset that contain worldwide videos
clips of abuse capture by surveillance cameras in the real-world scenes 

-	Design and implemented ML pipeline for video raw data to generate features, data augmentation techniques, and resampled datasets for model training.

-	Build and train a machine learning model[ADS] based on MobileNet-SSD architecture with 3D-CNN and two-stream method [RGB, OPT].
Training and evaluation of the model using AWS-SageMaker and TensorFlow frameworks. Achieved 87% in F1-score on DCSASS Abuse Dataset and 84%  on ADS Dataset.

-	Combine all models to an overall system and deploying the system in the Streamlit web app that enables the user to get real-time notification alerts
when an abuse event capture by the security camera.

## Introduction 
---
This project is defined as research(60%)\development(40%).

- Research 
Build and train deep learning models(according to a standard ML approach) to automatically identify abuse event capture by security camera

- Development
Build prototype system ADS(Abuse detection system) for deploying models and test them in a real-time environment

### Our Main goal - provide an automated solution for detecting cases of abuse of the elderly.

## Data collection
---
Data collection
In order to train deep learning models, the first step is data collection
We build data collection pipe and gather abuse video from the web
we collect 842 video clips after the filtering process


We work according to a machine learning methodology
1. search abuse video links online
2. download the links
3. convert the video to AVI format with FBS=30sec
4. cut the video into 5-sec clips
5. manual extracting from each video 5sec clips [3,4 clips for each video]
6. create more videos by using 5 data argumentation techniques
7. split the data to Train, Val, Test as shown in table2

- Method and DB expleind -[method and DB expleind.pdf](https://github.com/1amitos1/AbuseDetectionSystem_demo/files/6423235/method.and.db.expleind.pdf)

 <img src="https://user-images.githubusercontent.com/34807427/117050368-f15d1c00-ad1d-11eb-85eb-d21343f74e55.png" width="300" height="300">


## Model architecture
---
The model architecture is based on mobileNet SSD.
And the highlight of this model is utilizing
a branch of the optical flow channel to 
help build a pooling mechanism.

- Conv3D split into two channels -  RGB frame and Optical flows as shown in the figure below.
- Relu activation is adopted at the end of the RGB channel. 
- Sigmoid activation is adopted at the end of the Optical flow channel.
- RGB and Optical Flow channels outputs are multiplied together and processed by a temporal max-pooling.
- Merging Block is composed of basic 3D CNNs, used to process information after self learned temporal pooling. 
- Fully-connected layers generate output.

 <img src="https://user-images.githubusercontent.com/34807427/117047169-3c753000-ad1a-11eb-93a5-7825120596ca.png" width="550" height="400">


## Model training && Evaluation 
---
### Model training
We conduct tow experiment with different optimization algorithm
Stochastic gradient descent (SGD) and Adam   

- Experiment-1 - [Experiment 1 SGD.pdf](https://github.com/1amitos1/AbuseDetectionSystem_demo/files/6423205/Experiment.1.SGD.pdf)

- Experiment-2 - [Experiment 2 Adam.pdf](https://github.com/1amitos1/AbuseDetectionSystem_demo/files/6423210/Experiment.2.Adam.pdf)
### Evaluation
- Full Evaluation report [Model evaluation report.pdf](https://github.com/1amitos1/AbuseDetectionSystem_demo/files/6423215/Model.evaluation.report.pdf)

- We evaluate our proposed model on five Dataset.

  1-ADS data set

  2-DCSASS Abuse Dataset
 
  3-Automatic violence detection data set
 
  4- DCSASS Abuse Dataset + Automatic violence detection data set 

  5-RWF real world fight dataset

- F1 score result 
 
![res_f1_summary](https://user-images.githubusercontent.com/34807427/117053146-4a7a7f00-ad21-11eb-8c9d-74858d67efea.png)


## ADS PIPELINE Demo
---
We implemented the following models in ours pipeline Yolo DeepSort ADS
Processing steps:

### First step:
- We sample from the IP camera 2 frame set each set containing
149 frame set, and then we saves the sampling video in a folder

![First step](https://user-images.githubusercontent.com/34807427/117056502-0b4e2d00-ad25-11eb-9f83-a9bc9148d680.gif)

### Second step:
- Open sampling video folder and run yolo and DeepSort models to predict  tracking    bounding boxes in the video
- Bluer each frame set by bounding boxes then save those video clips in a folder

![Second step](https://user-images.githubusercontent.com/34807427/117056508-0d17f080-ad25-11eb-8119-d921dd36d9bd.gif)


### Third step:
- We open the video folder implement ads model preprocessing [resize shape, extract optical flow , uniform sampling to  64 frame for predication]


![Third step](https://user-images.githubusercontent.com/34807427/117056490-05f0e280-ad25-11eb-86b3-8706517114c2.gif)

- if the ads model identified the frame set as violence, we save the video clip and send it to the user email address     

![email](https://user-images.githubusercontent.com/34807427/117056968-93cccd80-ad25-11eb-9c8d-bd402e0fe378.png)


## Input-Output examples
---
input

![input](https://user-images.githubusercontent.com/34807427/117035434-fcf41700-ad0c-11eb-9e6f-c0c6d542f3ef.gif)              


output

![output](https://user-images.githubusercontent.com/34807427/117035426-fa91bd00-ad0c-11eb-93ff-6504835bee3e.gif)


## System overview
![ads overview2](https://user-images.githubusercontent.com/34807427/117057636-5ddc1900-ad26-11eb-9b71-d6344bd0dc78.png)



## Setup  
---
- first step downlaod yolo_v3 model
  [Yolo_v3 model](https://drive.google.com/file/d/1IbR2LtlqQxOr5w9u8yIeFYWtLJHksguF/view?usp=sharing)
- Add the model to model data file



## Reference
---
- Yolo_v3
- DeepSort
- RWF -


