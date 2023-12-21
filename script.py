# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:10:54 2023

@author: theli
"""
#need to import all of these for most projects
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
#getting from built in dataset, divided by 255 to standardize the pixels
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['plane',"car",'bird','cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#makes 4x4 grid with 16 elements
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
#shows the og graph
plt.show()
#we limit it to 20000 just to make it faster, higher limit means more accurate predictions
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]
#model training stuff, you can comment it out after youve built and saved the model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation = 'relu' , input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation = "relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation = "relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation = 'relu'))
# model.add(layers.Dense(10, activation = 'softmax'))

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss} ")
# print(f'Accuracy: {accuracy}')

# model.save('image_classifer.model')

model = models.load_model("image_classifer.model")
from keras.preprocessing import image
#for loading images
# Load the image using OpenCV
img = cv.imread("Dog_Breeds.jpg")
# Convert BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Display the original image
plt.imshow(img)
plt.title("Original Image")
plt.show()

# Preprocess the image for prediction
img = image.load_img('Dog_Breeds.jpg', target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize the pixel values

# Reshape to match the expected input shape
img_array = np.expand_dims(img_array, axis=0)

# Make predictions
prediction = model.predict(img_array)

# Display the image with its predicted label
predicted_label = class_names[np.argmax(prediction)]
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.show()

#10 neurons activated, we want the index of the msot accurate neuron
#with argmax we get the index of neuron
index = np.argmax(prediction)
print(f'prediction is {class_names[index]}')
plt.show()
      