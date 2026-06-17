import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
import keras
from keras import layers, models

# --- 1. DIRECTORY CONFIGURATION ---
BASE_DIR = "C:/Personal Data/Kittu/BTech project/chest_xray"
IMG_SIZE = 256

def get_all_pneumonia_paths():
    normal_paths, pneumonia_paths = [], []
    for split in ["train", "test", "val"]:
        for category, path_list in [("NORMAL", normal_paths), ("PNEUMONIA", pneumonia_paths)]:
            p = os.path.join(BASE_DIR, split, category)
            if os.path.exists(p):
                files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(('.png', '.jpg', '.jpeg'))]
                path_list.extend(files)
    
    print(f"📊 Dataset Stats: {len(normal_paths)} Normal | {len(pneumonia_paths)} Pneumonia")
    return normal_paths, pneumonia_paths

# --- 2. DATA LOADER ---
def load_dataset(mode="RAW"):
    normal_paths, pneumonia_paths = get_all_pneumonia_paths()
    X, y = [], []
    clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for paths, label in [(normal_paths, 0), (pneumonia_paths, 1)]:
        for p in tqdm(paths, desc=f"📂 Loading {mode} (Label {label})"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                if mode == "CLAHE":
                    img = clahe_tool.apply(img)
                X.append(np.expand_dims(img/255.0, -1).astype('float32'))
                y.append(label)
    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42, stratify=y)

# --- 3. ARCHITECTURE ---
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 4. EXECUTION ---
if __name__ == "__main__":
    for mode in ["RAW", "CLAHE"]:
        print(f"\n--- 🚀 TRAINING PNEUMONIA {mode} MODEL ---")
        X_train, X_test, y_train, y_test = load_dataset(mode=mode)
        
        model = build_model()
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        
        # Save individual Accuracy/Loss Chart
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title(f'Pneumonia Model Accuracy ({mode})')
        plt.legend()
        plt.savefig(f"pneumonia_accuracy_{mode}.png")
        plt.show()

        # Save individual Confusion Matrix
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: Pneumonia ({mode})')
        plt.savefig(f"pneumonia_cm_{mode}.png")
        plt.show()

        print(f"✅ {mode} Report:\n", classification_report(y_test, y_pred))
        model.save(f"pneumonia_model_{mode.lower()}.keras")