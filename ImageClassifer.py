import tensorflow as tf
import numpy as np
from tensorflow import keras

#for printing
import matplotlib.pyplot as plt

#Load a predifed dataset
fashion_mist = keras.datasets.fashion_mnist

#Pulling Data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mist.load_data()

#Show Data
#print(train_labels[0])
#print(train_images[0])
plt.imshow(train_images[1], cmap='gray', vmin=0, vmax=255)
plt.show()

print("hasvin")

#Define Neraul Net
model = keras.Sequential([
    #input image 28*28(Flatten out to 784*1 imput layer)
    keras.layers.Flatten(input_shape=(28,28)),

    #it's 128 deep, relu is a return
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    #Output layer 0-10 (depending on which cloth type), return maximum
    keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

#Compile our model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Traing model using data
model.fit(train_images, train_labels, epochs=5)

#Testing the model
test_lose = model.evaluate(test_images, test_labels)

#Make predictions
predictions = model.predict(test_images)

#Print out the prediction
print(list(predictions[1]).index(max(predictions[1])))

#Print correct answer
print(test_labels[1])