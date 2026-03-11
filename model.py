# ------------------------------------------------------------ Import libraries

import os
import math
import random
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

# ------------------------------------------------- Make Training Deterministic

SEED = 42
tf.config.experimental.enable_op_determinism()
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------ Check If We Have GPU Support

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# ---------------------------------------------------------- Load Training Data

training_dir = "data/training/"
image_size = (100, 100)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    # Attempted augmentations of the training set:
    zoom_range=0.1,
    shear_range=0.1,
    rotation_range=10,
    fill_mode='nearest',
    horizontal_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # TODO: channel shift range - try it
    # TODO: brightness shift range - try it
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
    seed=SEED,
)

validation_generator = validation_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    class_mode="sparse",
    subset="validation",
    batch_size=32,
    seed=SEED
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

# For a batch size of 64, I had Calculated Accuracy: 0.9104477611940298
# For a batch size of 32, I had Calculated Accuracy: 0.9203980099502488
# After adding in batch normalization: 0.9751243781094527 (WOW!)

# ---------------------------------------------------------- MODEL ARCHITECTURE

# TESTING MAX POOLING 2x2 INSTEAD OF 3x3:
#
#

model = models.Sequential()

# TODO: padding="same" - it pads the image so conv filters don't shrink it
# TODO: try changing to 32, and make 2x2 maxpooling
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
# TODO: add a dropout layer???
model.add(layers.Dense(256, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(43, activation="softmax"))
model.summary()

# -----------------------------------------------------  COMPILATION & TRAINING

opt = keras.optimizers.Adam()
model.compile(
    loss="sparse_categorical_crossentropy",
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

early_stop = tf.keras.callbacks.EarlyStopping(
    restore_best_weights=True,
    monitor='val_accuracy',
    patience=15,
    mode='max',
)

model.fit(
    train_generator,
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback, early_stop],
    initial_epoch=0,
    epochs=500,
)

# Load best weights
model.load_weights(checkpoint_filepath)

# ------------------------------------------------------------- SAVE FULL MODEL

model.save("model.keras")
print("Model saved to model.keras")

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

# ---------------------------------------- TESTING & PREDICTIONS - MINI HOLDOUT

mini_test_datagen = ImageDataGenerator(rescale=1./255)
mini_test_generator = mini_test_datagen.flow_from_directory(
    "data/",
    classes=["mini_holdout"], 
    target_size=image_size,
    class_mode='sparse', 
    shuffle=False,
)

mini_predictions = model.predict(mini_test_generator)
mini_y_pred = np.argmax(mini_predictions, axis=1)

# Save results
output = pd.DataFrame(mini_y_pred)
output.to_csv("mini_holdout_predictions.csv", index=False)

# ------------------------------------------------------- PREDICTIONS - HOLDOUT

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "data/",
    classes=["holdout"], 
    target_size=image_size,
    class_mode='sparse', 
    shuffle=False,
)

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# Save results
output = pd.DataFrame(y_pred)
output.to_csv("holdout_predictions.csv", index=False)

# ------------------------------------------------------------------ EVALUATION

real_answers = pd.read_csv("data/mini_holdout_answers.csv")
print(real_answers.head(100))

print(classification_report(real_answers['ClassId'], mini_y_pred))

# Comparison Dataframe
overall = pd.concat([
    pd.DataFrame(mini_y_pred, columns=['class']),
    real_answers['ClassId'].reset_index(drop=True)
], axis=1)

accuracy_calc = 1 - len(overall[overall['class'] != overall['ClassId']]) / len(overall)
print(f"Calculated Accuracy: {accuracy_calc}")
print(overall[overall['class'] != overall['ClassId']])
print("Missed: ", len(overall[overall['class'] != overall['ClassId']]))
print(overall.head(15))

# ------------------------------------------------------------ CONFUSION MATRIX

cm = confusion_matrix(real_answers['ClassId'], mini_y_pred)

plt.figure(figsize=(14, 12))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Use target names
plt.xticks(np.arange(len(target_names)), target_names, rotation=90)
plt.yticks(np.arange(len(target_names)), target_names)

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)

plt.colorbar()
plt.tight_layout()
plt.show()