# Abuse Detection System
---
Provide a Deep Learning-Based real-time solution for nursing homes and hospitals for detecting cases of abuse in the elderly population by analyzing security camera frames and performing real-time forecasting using three machine learning models YOLO, DeepSort, ADS

## Table of contents
---
* [Project highlights](#Project-highlights)
* [Introduction ](#Introduction)
* [Model training && Evaluation](#Model-training-&&-Evaluation)
* [Model architecture](#Model-architecture)
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


## Setup  
---
- first step downlaod yolo_v3 model
  [Yolo_v3 model](https://drive.google.com/file/d/1IbR2LtlqQxOr5w9u8yIeFYWtLJHksguF/view?usp=sharing)
- Add the model to model data file


## Input-Output examples
input

![input](https://user-images.githubusercontent.com/34807427/117035434-fcf41700-ad0c-11eb-9e6f-c0c6d542f3ef.gif)              


output

![output](https://user-images.githubusercontent.com/34807427/117035426-fa91bd00-ad0c-11eb-93ff-6504835bee3e.gif)

## Reference


