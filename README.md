## Emotion Detection Model
This repository contains a deep learning model for emotion detection based on the Einterface Image Dataset from Kaggle. The model is built using TensorFlow and Keras and uses a convolutional neural network (CNN) architecture.

### Dataset
The [Einterface Image Dataset consists](https://www.kaggle.com/datasets/ameyamote030/einterface-image-dataset) of 6,000 grayscale images of faces labeled with one of 7 different emotions: angry, disgusted, fearful, happy, neutral, sad, and surprised. The dataset is split into a training set of 4,500 images and a validation set of 1,500 images.

### Model Architecture
The emotion detection model uses a CNN architecture with the following layers:

Conv2D layer with 16 filters and a 3x3 kernel, with ReLU activation and input shape (128, 128, 3)
MaxPooling2D layer with a 2x2 pool size
Conv2D layer with 32 filters and a 2x2 kernel, with ReLU activation
MaxPooling2D layer with a 2x2 pool size
Conv2D layer with 64 filters and a 2x2 kernel, with ReLU activation
MaxPooling2D layer with a 2x2 pool size
Conv2D layer with 128 filters and a 2x2 kernel, with ReLU activation
MaxPooling2D layer with a 2x2 pool size
Flatten layer
Dense layer with 6 units (one for each emotion category) and softmax activation
### Evaluation Metrics
The model is evaluated on the validation set using categorical cross-entropy loss and accuracy metrics.

### Requirements
The following packages are required to run the code:

TensorFlow 2.x
NumPy
Matplotlib
OpenCV
Scikit-learn
### How to Use
Clone the repository to your local machine
Install the required packages listed above
Run 'train model notebook.ipynb' to train the model
To evaluate the model use only validation set
### Future Work
Explore additional preprocessing techniques to improve model performance
Experiment with different CNN architectures to achieve higher accuracy
Collect additional data to train the model on a more diverse set of emotions and facial expressions.
