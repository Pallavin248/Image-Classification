# Deep Learning - Image Classification

This repository contains a deep learning project for **image classification** using Convolutional Neural Networks (CNNs).  
The project demonstrates data preprocessing, model building, training, evaluation, and visualization of classification results.

---

## 📌 Features
- Image preprocessing and augmentation
- CNN/ResNet-based model architecture
- Multi-class image classification
- Training and validation pipeline
- Model evaluation using accuracy, confusion matrix, and visualizations
- Export of results for analysis

---

# --- Dataset ---
kaggle datasets download -d alsaniipe/flowers-multiclass-datasets --unzip -p dataset

## 📂 Project Structure
Deep-learning-image-classification/
│── data/ # Dataset (optional, not uploaded if large)
│── models/ # Saved models/weights
│── notebooks/ # Jupyter notebooks
│── src/ # Python scripts (training, evaluation, utils)
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── .gitignore # Ignored files (data, logs, etc.)



#MODEL

# --- IMPORTS ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import json
import random
import tarfile

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, r2_score, confusion_matrix, ConfusionMatrixDisplay

# --- STEP 1: EXTRACT DATASET ---
tgz_path = "/content/flower_photos.tgz"
with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(path="/content")

image_dir = "/content/flower_photos"
classes = [cls for cls in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, cls))]

image_paths, flower_labels = [], []
for cls in classes:
    cls_folder = os.path.join(image_dir, cls)
    for img_name in os.listdir(cls_folder):
        if img_name.lower().endswith(".jpg"):
            image_paths.append(os.path.join(cls_folder, img_name))
            flower_labels.append(cls)

# --- STEP 2: SYNTHETIC LABELS FOR COLOR AND OIL CONTENT ---
colors = ['red', 'yellow', 'white', 'pink', 'purple']

def generate_oils():
    return np.round(np.random.rand(3), 2)

data = {
    'image_path': image_paths,
    'flower_type': flower_labels,
    'color': random.choices(colors, k=len(image_paths)),
    'Linalool': [],
    'Geraniol': [],
    'Citronellol': []
}

for _ in image_paths:
    l, g, c = generate_oils()
    data['Linalool'].append(l)
    data['Geraniol'].append(g)
    data['Citronellol'].append(c)

df = pd.DataFrame(data)

# --- STEP 3: ENCODE LABELS AND SCALE OILS ---
flower_encoder = LabelEncoder()
df['flower_encoded'] = flower_encoder.fit_transform(df['flower_type'])

color_encoder = LabelEncoder()
df['color_encoded'] = color_encoder.fit_transform(df['color'])

scaler = MinMaxScaler()
df[['Linalool', 'Geraniol', 'Citronellol']] = scaler.fit_transform(df[['Linalool', 'Geraniol', 'Citronellol']])

# --- STEP 4: REMOVE OUTLIERS IN OIL DATA ---
from scipy.stats import zscore
z_scores = np.abs(zscore(df[['Linalool', 'Geraniol', 'Citronellol']]))
df = df[(z_scores < 3).all(axis=1)]

# --- STEP 5: PREPROCESS IMAGES ---
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return preprocess_input(img)

X = np.array([preprocess_image(p) for p in df['image_path']])
y_flower = tf.keras.utils.to_categorical(df['flower_encoded'])
y_color = tf.keras.utils.to_categorical(df['color_encoded'])
y_oil = df[['Linalool', 'Geraniol', 'Citronellol']].values

# --- STEP 6: SPLIT DATASET ---
X_train, X_test, y_train_f, y_test_f, y_train_c, y_test_c, y_train_o, y_test_o = train_test_split(
    X, y_flower, y_color, y_oil,
    test_size=0.2,
    stratify=df['flower_encoded'],
    random_state=42
)

# --- STEP 7: MODEL BUILDING ---
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

for layer in base_model.layers[:-30]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)

flower_output = Dense(y_flower.shape[1], activation='softmax', name='flower_output')(x)
color_output = Dense(y_color.shape[1], activation='softmax', name='color_output')(x)
oil_output = Dense(3, activation='sigmoid', name='oil_output')(x)

model = Model(inputs=input_tensor, outputs=[flower_output, color_output, oil_output])

model.compile(
    optimizer='adam',
    loss={
        'flower_output': 'categorical_crossentropy',
        'color_output': 'categorical_crossentropy',
        'oil_output': 'mse'
    },
    metrics={
        'flower_output': 'accuracy',
        'color_output': 'accuracy',
        'oil_output': ['mae', 'mse']
    }
)

# --- STEP 8: TRAIN MODEL ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train,
    {'flower_output': y_train_f, 'color_output': y_train_c, 'oil_output': y_train_o},
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

# --- STEP 9: EVALUATE MODEL ---
y_pred_flower, y_pred_color, y_pred_oil = model.predict(X_test)

# Classification report
y_true_flower = np.argmax(y_test_f, axis=1)
y_pred_flower_cls = np.argmax(y_pred_flower, axis=1)
print("\nClassification Report for Flower Type:")
print(classification_report(y_true_flower, y_pred_flower_cls, target_names=flower_encoder.classes_))

# Confusion Matrix - Flower
cm_flower = confusion_matrix(y_true_flower, y_pred_flower_cls)
disp_flower = ConfusionMatrixDisplay(confusion_matrix=cm_flower, display_labels=flower_encoder.classes_)
disp_flower.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Flower Type")
plt.tight_layout()
plt.show()

# Confusion Matrix - Color
y_true_color = np.argmax(y_test_c, axis=1)
y_pred_color_cls = np.argmax(y_pred_color, axis=1)
cm_color = confusion_matrix(y_true_color, y_pred_color_cls)
disp_color = ConfusionMatrixDisplay(confusion_matrix=cm_color, display_labels=color_encoder.classes_)
disp_color.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Flower Color")
plt.tight_layout()
plt.show()

# Regression metrics for oil prediction
r2 = r2_score(y_test_o, y_pred_oil)
mse = np.mean(np.square(y_test_o - y_pred_oil))
mae = np.mean(np.abs(y_test_o - y_pred_oil))
print(f"R^2 Score for Oil Predictions: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# --- STEP 10: PREDICTION FUNCTION ---
def predict_from_image(image_path):
    img = preprocess_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    flower_pred, color_pred, oil_pred = model.predict(img_batch)

    flower = flower_encoder.inverse_transform([np.argmax(flower_pred)])[0]
    color = color_encoder.inverse_transform([np.argmax(color_pred)])[0]
    oil = scaler.inverse_transform(oil_pred)[0]

    result = {
        'predicted_flower_type': flower,
        'predicted_flower_color': color,
        'estimated_oil_concentrations': {
            'Linalool': float(round(oil[0], 2)),
            'Geraniol': float(round(oil[1], 2)),
            'Citronellol': float(round(oil[2], 2))
        }
    }

    img_disp = cv2.imread(image_path)
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].axis('off')
    ax[0].text(0, 0.5, json.dumps(result, indent=2), fontsize=10, fontfamily='monospace')
    ax[1].imshow(img_disp)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

    return result

# --- OPTIONAL: test a random prediction ---
def test_random_prediction():
    sample = df.sample(1).iloc[0]
    return predict_from_image(sample['image_path'])

test_random_prediction()



