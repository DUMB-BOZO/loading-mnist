MNIST Digit Classifier with TensorFlow Keras
This repository contains a Google Colab notebook demonstrating a simple yet effective Neural Network for classifying handwritten digits from the MNIST dataset. The project aims to build, train, and evaluate a feed-forward neural network using TensorFlow Keras.

Project Description
The primary goal of this project is to implement a basic neural network to recognize handwritten digits (0-9) from the widely-used MNIST dataset. The network architecture is designed to flatten the input images and then process them through dense layers with ReLU activation, culminating in a Softmax output layer for multi-class classification. The model's performance is evaluated using a comprehensive classification report.

Model Architecture
The neural network is a sequential model composed of the following layers:

Input Layer: Flatten layer to transform the 28x28 pixel grayscale images into a 1D array of 784 features.

Hidden Dense Layer 1: A Dense layer with 400 neurons and ReLU (Rectified Linear Unit) activation function.

Hidden Dense Layer 2: A Dense layer with 128 neurons and ReLU activation function.

Output Dense Layer: A Dense layer with 10 neurons (one for each digit class) and a Softmax activation function to output probability distributions over the classes.

The model is compiled using the Adam optimizer and sparse_categorical_crossentropy as the loss function, with accuracy as the evaluation metric.

Dataset
The MNIST (Modified National Institute of Standards and Technology) dataset is used for this project. It consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is a 28x28 pixel grayscale image.

How to Run
This project is designed to be run on Google Colaboratory (Colab), a free cloud-based Jupyter notebook environment that requires no setup and runs entirely in the cloud.

Open in Google Colab:

Go to Google Colab.

Click on "File" > "Upload notebook" and select the .ipynb file from this repository.

Alternatively, if you're viewing this README on GitHub, you can usually click the "Open in Colab" badge (if available) or copy the notebook's URL and paste it into Colab's "File" > "Open notebook" > "GitHub" tab.

Run All Cells:

Once the notebook is open in Colab, go to "Runtime" > "Run all" to execute all the code cells.

The notebook will automatically download the MNIST dataset, train the model, evaluate it, and display the classification report.

Dependencies
The core libraries required are:

TensorFlow: For building and training the neural network.

Keras: TensorFlow's high-level API for neural networks.

NumPy: For numerical operations.

Scikit-learn: Specifically for classification_report to evaluate model performance.

These libraries are typically pre-installed in Google Colab environments.

Results
After training for 10 epochs, the model achieves impressive performance on the test set.

Test Loss: Approximately 0.0905

Test Accuracy: Approximately 0.9815

The classification report further details the precision, recall, and f1-score for each digit class, showing consistently high scores across all categories, indicating robust classification performance.
