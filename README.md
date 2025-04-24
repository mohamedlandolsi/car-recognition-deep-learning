# Car Make and Model Recognition

A deep learning project that classifies car makes and models using Convolutional Neural Networks (CNNs).

## Project Overview

This project implements and compares three different CNN-based approaches for car make and model recognition:

1. **Basic CNN from scratch**: A simple convolutional neural network built and trained from scratch
2. **CNN with Data Augmentation**: The same CNN architecture with data augmentation techniques applied
3. **Transfer Learning with ResNet50V2**: Using a pre-trained ResNet50V2 model with custom classification layers

## Dataset

The dataset consists of car images organized by make and model. The images are divided into training, validation, and test sets.

## Requirements

The project requires the following dependencies:
- TensorFlow 2.x
- Keras
- NumPy
- SciPy
- Matplotlib
- Pandas

You can install all requirements using:
```
pip install -r requirements.txt
```

## Project Structure

- `Car_Recognition_CNN.ipynb`: Main Jupyter notebook containing the model creation, training, and evaluation
- `cars_annos.mat`: Annotations file with class information
- `test_car_image.py`: Script for testing the model with custom car images
- `cars_train/`: Directory containing training images
- `cars_test/`: Directory containing test images
- `organized_cars_dataset/`: Directory for processed dataset split into train/validation/test sets
- `training_checkpoints/`: Directory for saved model checkpoints

## How to Use

1. Clone this repository
2. Install the dependencies
3. Run the Jupyter notebook to train and evaluate the models
4. Use the test functionality to classify new car images

## Model Performance

The project compares the performance of three different approaches:
- Basic CNN accuracy: ~70%
- CNN with data augmentation: ~75%
- Transfer learning with ResNet50V2: ~90%

## License

[MIT License](LICENSE)