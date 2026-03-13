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

# ============================================================
# TRAINING DATA EVALUATION
# ============================================================

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/training",
    target_size=image_size,
    class_mode='sparse',
    shuffle=False
)

train_predictions = model.predict(train_generator)
train_y_pred = np.argmax(train_predictions, axis=1)

train_y_true = train_generator.classes

print("\n================ TRAINING DATA REPORT ================")

print(classification_report(
    train_y_true,
    train_y_pred,
    labels=range(43),
    target_names=target_names
))

train_accuracy = np.mean(train_y_pred == train_y_true)
print("Training Accuracy:", train_accuracy)

# ---------------- Confusion Matrix (Training)

labels_present_train = sorted(np.unique(train_y_true))
target_subset_train = [target_names[i] for i in labels_present_train]

cm_train = confusion_matrix(
    train_y_true,
    train_y_pred,
    labels=labels_present_train
)

plt.figure(figsize=(14,12))
plt.imshow(cm_train, cmap="gray_r")
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(np.arange(len(target_subset_train)), target_subset_train, rotation=90)
plt.yticks(np.arange(len(target_subset_train)), target_subset_train)

for i in range(cm_train.shape[0]):
    for j in range(cm_train.shape[1]):
        plt.text(j, i, cm_train[i, j], ha="center", va="center", fontsize=8)

plt.colorbar()
plt.tight_layout()
plt.show()

# ============================================================
# MINI HOLDOUT EVALUATION
# ============================================================

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

print("\n================ MINI HOLDOUT REPORT ================")

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

# ---------------- Confusion Matrix (Mini Holdout)

labels_present = sorted(real_answers['ClassId'].unique())
target_subset = [target_names[i] for i in labels_present]

cm = confusion_matrix(
    real_answers['ClassId'],
    mini_y_pred,
    labels=labels_present
)

plt.figure(figsize=(14,12))
plt.imshow(cm, cmap="gray_r")
plt.title("Mini Holdout Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(np.arange(len(target_subset)), target_subset, rotation=90)
plt.yticks(np.arange(len(target_subset)), target_subset)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

plt.colorbar()
plt.tight_layout()
plt.show()

# ============================================================
# HARDEST TRAINING EXAMPLES
# ============================================================

# Max probability for each image
max_probs = np.max(train_predictions, axis=1)

# Indices of the 10 lowest confidence predictions
hardest_indices = np.argsort(max_probs)[:100]

print("\n================ HARDEST TRAINING IMAGES ================")

for idx in hardest_indices:

    probs = train_predictions[idx]

    # top 2 predictions
    top2 = np.argsort(probs)[-2:][::-1]

    p1, p2 = probs[top2[0]], probs[top2[1]]

    class1 = target_names[top2[0]]
    class2 = target_names[top2[1]]

    path = train_generator.filepaths[idx]

    print("\nImage:", path)
    print("Top prediction:", class1, "(", p1, ")")
    print("Second prediction:", class2, "(", p2, ")")

# ============================================================
# MOST CONFUSED TRAINING IMAGES (closest top-two predictions)
# ============================================================

records = []

for i, probs in enumerate(train_predictions):

    # indices of the top two classes
    top2 = np.argsort(probs)[-2:][::-1]

    p1 = probs[top2[0]]
    p2 = probs[top2[1]]

    margin = abs(p1 - p2)
    strength = p1 + p2

    records.append((margin, -strength, i, top2, p1, p2))

# Sort by margin first, then by strength
records.sort()

top10 = records[:10]

print("\n================ MOST CONFUSED TRAINING IMAGES ================")

for margin, neg_strength, idx, top2, p1, p2 in top10:

    class1 = target_names[top2[0]]
    class2 = target_names[top2[1]]

    path = train_generator.filepaths[idx]

    print("\nImage:", path)
    print("Top prediction:", class1, f"({p1:.4f})")
    print("Second prediction:", class2, f"({p2:.4f})")
    print("Margin:", f"{margin:.6f}")