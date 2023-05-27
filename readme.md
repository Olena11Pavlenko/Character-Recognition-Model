# EMNIST Balanced Dataset - Character Recognition Documentation

## Introduction
This documentation provides an overview of the data, methods, and ideas used for character recognition using the EMNIST Balanced dataset. The goal of the project is to train a convolutional neural network (CNN) model to accurately classify handwritten characters into their respective classes.

## Data
The dataset used for this project is the EMNIST Balanced dataset. It is a subset of the larger EMNIST dataset and contains a balanced distribution of handwritten characters from 47 different classes. Each class represents a unique character, including uppercase letters, lowercase letters, and digits. The dataset consists of 131,600 grayscale images of size 28x28 pixels.

The dataset was split into training and testing sets. The training set was used to train the model, while the testing set was used to evaluate the performance of the trained model. The labels in the dataset were one-hot encoded to represent the target classes.

## Model Architecture
The model used for character recognition is a convolutional neural network (CNN). The architecture of the CNN model is as follows:

1. Conv2D layer with 32 filters, a kernel size of 3x3, ReLU activation, and an input shape of (28, 28, 1).
2. MaxPooling2D layer with a pool size of 2x2.
3. Dropout layer with a dropout rate of 0.2.
4. Conv2D layer with 128 filters, a kernel size of 3x3, and ReLU activation.
5. MaxPooling2D layer with a pool size of 2x2.
6. Dropout layer with a dropout rate of 0.2.
7. Flatten layer to convert the 2D feature maps into a 1D feature vector.
8. Dense layer with 512 units and ReLU activation.
9. Dropout layer with a dropout rate of 0.2.
10. Dense layer with 128 units and ReLU activation.
11. Dense layer with 47 units (equal to the number of classes) and softmax activation for multi-class classification.

The total number of trainable parameters in the model is 1,747,951.

## Model Training
The model was trained using the Adam optimizer and the categorical cross-entropy loss function. The accuracy metric was used to monitor the performance of the model during training.

The training process was executed for 20 epochs with a batch size of 256. Early stopping was implemented with a patience of 5 epochs to prevent overfitting. Model checkpointing was used to save the best model based on the validation accuracy.

## Accuracy
After training the model, the following accuracies were achieved:

- Training Accuracy: 91.37%
- Validation Accuracy: 88.63%

## Usage Instructions

`docker build -t handwriting .`

`docker run -v <path_to_images>:/app/images handwriting ./images`

## Author Information
This character recognition project and its accompanying documentation were created by Olena Pavlenko.