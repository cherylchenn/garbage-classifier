# Garbage Classifier

## Overview
This project is a real-time garbage classifcation system that detects hand-held objects and recommends the appropriate disposal bin. It uses the ResNet-18 CNN, and MediaPipe + OpenCV for live webcam input.

## Features
- Multi-class classification: paper, cardboard, plastic, glass, metal, trash
- Trained on the [TrashNet](https://huggingface.co/datasets/garythung/trashnet) dataset (96% validation accuracy)
- Real-time hand-held object detection with bounding box and confidence score
- Provides disposal bin recommendations

## Installation + Usage
1. Install dependencies: ```pip install -r requirements.txt```
2. Train the model and create labels: ```python train.py```  
This will create ```model.pth``` and ```labels.txt``` (which are needed for the next step).
3. Run the real-time webcam demo: ```python webcam.py```
4. To exit, click the 'q' key.

<img width="500" height="397" alt="image" src="https://github.com/user-attachments/assets/8bf11fbe-4fba-4e97-9c61-5a0a50dda2a4" />
