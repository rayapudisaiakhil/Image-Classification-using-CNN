# Image Classification using CNN

This project implements image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNN) and Support Vector Machines (SVM).

## Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset contains 60,000 32x32 color images belonging to 10 classes. The classes are:

- Airplane
- Automobile
- Bird
- Cat 
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is divided into 50,000 train images and 10,000 test images.

## Models

The following models are implemented and evaluated:

### Convolutional Neural Network

- Input layer to accept 32x32x3 image 
- Convolutional layers with ReLU activation and max pooling
- Fully connected layers 
- Softmax output layer
- Adam optimizer
- Sparse categorical crossentropy loss

Data augmentation is used to expand the training data. 

### Support Vector Machine

- Polynomial kernel
- RBF kernel
- Grid search for hyperparameter tuning

## Usage

The project is implemented in Python using TensorFlow and scikit-learn.

To run the code:

1. Clone the repository
2. Install dependencies from requirements.txt
3. Update ROOT_PATH variable with CIFAR-10 dataset location
4. Run the IPython notebooks

## Results

The CNN model achieves 87.86% accuracy after 50 epochs of training with data augmentation.

The SVM models yield lower accuracy:

- Polynomial kernel - 44.68% 
- RBF kernel - 49.95%

## Conclusion

The CNN model outperforms SVM on this image classification task. The results demonstrate the capability of CNNs to extract hierarchical features from images.
