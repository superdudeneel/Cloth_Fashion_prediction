import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # loading the dataset
# The dataset consists of 60000 training images and 10000 testing images



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # Clothing


train_images = train_images/255.0
test_images = test_images / 255.0


model = keras.Sequential([


    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')

]) # model initialization
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy'],
) # compilation of out neural network


model.fit(train_images, train_labels, epochs = 3) # Training the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1) # Evaluation

print(test_acc)
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[56])]) # you change the index here
plt.imshow(test_images[56]) # you can change the index here
plt.colorbar()

plt.show()


