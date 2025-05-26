# ASL Gesture Detection using CNN, Transfer Learning, LBP, Flask, and MLflow

This project presents a gesture recognition system for American Sign Language (ASL) using machine learning. It includes image preprocessing, feature extraction using Local Binary Patterns (LBP), model training using CNN and transfer learning, performance evaluation, MLflow experiment tracking, and deployment with Flask.

## Project Description

The system classifies ASL gestures into 29 classes (A–Z, space, nothing, delete). The model was trained on grayscale images and deployed through a web interface built using Flask. MLflow was used for tracking training metrics and artifacts.

## Dataset

The dataset used is from the following GitHub repository:  
[https://github.com/kumarvivek9088/aslsigndataset/tree/main/splitdataset48x48](https://github.com/kumarvivek9088/aslsigndataset/tree/main/splitdataset48x48)

It contains 48x48 grayscale images organized into training and validation folders.

## Methods Used

- **CNN**: Built with TensorFlow and Keras for gesture classification  
- **Transfer Learning**: Used pre-trained models to enhance accuracy  
- **LBP (Local Binary Patterns)**: Applied for feature extraction  
- **Training**:
  - Epochs: 93  
  - Batch Size: 128  
  - Optimizer: Adam  
  - Loss Function: Categorical Crossentropy

## MLflow Integration

- Used to log parameters, metrics, model structure, and results  
- Confusion matrix and classification report saved as artifacts  
- MLflow tracking server was hosted locally

## Deployment with Flask

- A web application was created for gesture prediction  
- Users can upload an image and view the predicted ASL character  
- The app uses the trained Keras model for inference

## Acknowledgements

The dataset and some base code references were taken from:  
[Vivek Kumar’s GitHub repository](https://github.com/kumarvivek9088/SignLanguageDetectionUsingCNN)

All enhancements including LBP integration, transfer learning, MLflow tracking, and Flask deployment were independently implemented.

This project is intended for academic use only.

## Author

Mousumi Sarkar
M.Tech Automotive Electronics & Software (AES)  
KPIT, KIIT
