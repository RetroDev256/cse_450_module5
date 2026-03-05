# ------------------------------------------------------------ Import libraries

import os
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from numpy import expand_dims
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageEnhance, ImageStat
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------- Load Training Data

training_dir = "data/training/"
image_size = (100, 100)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    # Attempted augmentations of the training set:
    # zoom_range=0.3,
    # shear_range=0.2,
    # rotation_range=20,
    # fill_mode='nearest',
    # horizontal_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=.2
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size = image_size,
    class_mode="sparse",
    subset="training",
    batch_size=32,
    shuffle=True,
    seed=42,
)

validation_generator = validation_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    class_mode="sparse",
    subset="validation",
    batch_size=32,
    seed=42
)

# ---------------------------------------------------------------- Target Names

target_names = [
    'Speed_20', 'Speed_30', 'Speed_50', 'Speed_60', 'Speed_70', 'Speed_80',
    'Speed_Limit_Ends', 'Speed_100', 'Speed_120', 'Overtaking_Prohibited',
    'Overtakeing_Prohibited_Trucks', 'Priority', 'Priority_Road_Ahead',
    'Yield', 'STOP', 'Entry_Forbidden', 'Trucks_Forbidden',
    'No_Entry(one-way traffic)', 'General Danger(!)', 'Left_Curve_Ahead',
    'Right_Curve_Ahead', 'Double_Curve', 'Poor_Surface_Ahead',
    'Slippery_Surface_Ahead', 'Road_Narrows_On_Right',
    'Roadwork_Ahead', 'Traffic_Light_Ahead', 'Warning_Pedestrians',
    'Warning_Children', 'Warning_Bikes', 'Ice_Snow', 'Deer_Crossing',
    'End_Previous_Limitation', 'Turning_Right_Compulsory',
    'Turning_Left_Compulsory', 'Ahead_Only', 'Straight_Or_Right_Mandatory',
    'Straight_Or_Left_Mandatory', 'Passing_Right_Compulsory',
    'Passing_Left_Compulsory', 'Roundabout', 'End_Overtaking_Prohibition',
    'End_Overtaking_Prohibition_Trucks'
]

# ---------------------------------------------------------- MODEL ARCHITECTURE

model = models.Sequential()
model.add(layers.Conv2D(2, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (7, 7), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))

# modify the hidden layers
# model.add(layers.Conv2D(4, (5,5), strides=2,padding='same', activation='leaky_relu'))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(43))
model.summary()

# -----------------------------------------------------  COMPILATION & TRAINING

opt = keras.optimizers.Adam()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    optimizer=opt,
)

checkpoint_filepath = '/tmp/checkpoint7.weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
)

model.fit(
    train_generator,
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback],
    steps_per_epoch=120,
    initial_epoch=0,
    epochs=15,
)

# Load best weights
model.load_weights(checkpoint_filepath)

# --------------------------------------------------- FEATURE MAP VISUALIZATION

# redefine model to output right after the first hidden layer
model_temp = Model(inputs=model.inputs, outputs=model.layers[1].output) 
model_temp.summary()

# load the image with the required shape
img = load_img("data/holdout/00000.jpg", target_size=image_size)
img = img_to_array(img)
img = expand_dims(img, axis=0)

# get feature map for first hidden layer
feature_maps = model_temp.predict(img)

# ------------------------------------------------------- TESTING & PREDICTIONS

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "data/",
    classes=["mini_holdout"], 
    target_size=image_size,
    class_mode='sparse', 
    shuffle=False,
)

predictions = model.predict(test_generator)
y_pred = [np.argmax(probas) for probas in predictions]
print(y_pred)

# Save results
output = pd.DataFrame(y_pred)
output.to_csv("predictions.csv", index=False)

# ------------------------------------------------------------------ EVALUATION

real_answers = pd.read_csv("data/mini_holdout_answers.csv")
print(real_answers.head(100))

print(classification_report(real_answers['ClassId'], y_pred))

# Comparison Dataframe
overall = pd.concat([
    pd.DataFrame(y_pred, columns=['class']),
    real_answers['ClassId'].reset_index(drop=True)
], axis=1)

accuracy_calc = 1 - len(overall[overall['class'] != overall['ClassId']]) / len(overall)
print(f"Calculated Accuracy: {accuracy_calc}")
print(overall[overall['class'] != overall['ClassId']])
print("Missed: ", len(overall[overall['class'] != overall['ClassId']]))
print(overall.head(15))

# ------------------------------------------------------------ CONFUSION MATRIX

cm = confusion_matrix(real_answers['ClassId'], y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(cm.shape[1]))
plt.yticks(np.arange(cm.shape[0]))

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()