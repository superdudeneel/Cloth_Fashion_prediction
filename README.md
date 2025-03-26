# Fashion MNIST Classification

This repository contains a simple neural network implementation using TensorFlow and Keras to classify images from the Fashion MNIST dataset.

## Dataset

The dataset used is the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), which consists of 70,000 grayscale images of 10 different categories of clothing items. Each image has a resolution of 28x28 pixels.

## Model Architecture

The model consists of the following layers:

- **Flatten Layer**: Converts the 28x28 input images into a 1D array.
- **Dense Layer (128 neurons, ReLU activation)**: A fully connected hidden layer with 128 neurons.
- **Dense Output Layer (10 neurons, Softmax activation)**: A fully connected layer with 10 neurons corresponding to the 10 classes.

## Installation

To run this project, install the necessary dependencies:

```sh
pip install tensorflow matplotlib numpy pandas
```

## Training and Evaluation

The model is trained using the Adam optimizer and sparse categorical crossentropy loss function. It runs for 3 epochs.

To train the model, run:

```sh
python train.py
```

After training, the model evaluates its performance on the test dataset and prints the accuracy.

## Making Predictions

The script includes functionality to make predictions on test images. It displays an image from the test dataset along with the predicted label.

## Usage

Modify the index in the script to visualize different images:

```python
print(class_names[np.argmax(predictions[56])])  # Change index here
plt.imshow(test_images[56])  # Change index here
```

## Results

After training for 3 epochs, the model achieves an accuracy of around 87-90% on the test dataset.

## License

This project is licensed under the MIT License.

