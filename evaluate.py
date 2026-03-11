# ------------------------------------------------------------ Import libraries

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------ LOAD MODEL

model = tf.keras.models.load_model("model.keras")
print("Model loaded successfully")

image_size = (100,100)

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

# ------------------------------------------------ MINI HOLDOUT EVALUATION

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
# Load real answers
real_answers = pd.read_csv("data/mini_holdout_answers.csv")

print(classification_report(
    real_answers['ClassId'],
    mini_y_pred,
    labels=range(43),
    target_names=target_names
))

# Accuracy
overall = pd.concat([
    pd.DataFrame(mini_y_pred, columns=['class']),
    real_answers['ClassId'].reset_index(drop=True)
], axis=1)

accuracy_calc = 1 - len(overall[overall['class'] != overall['ClassId']]) / len(overall)
print("Mini Holdout Accuracy:", accuracy_calc)

# ------------------------------------------------------------ CONFUSION MATRIX

# Only include classes that appear in the mini holdout
labels_present = sorted(real_answers['ClassId'].unique())

cm = confusion_matrix(
    real_answers['ClassId'],
    mini_y_pred,
    labels=labels_present
)

# Corresponding target names
target_subset = [target_names[i] for i in labels_present]

plt.figure(figsize=(14, 12))
plt.imshow(cm)
plt.title("Mini Holdout Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(np.arange(len(target_subset)), target_subset, rotation=90)
plt.yticks(np.arange(len(target_subset)), target_subset)

# Add numbers in each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

plt.colorbar()
plt.tight_layout()
plt.show()