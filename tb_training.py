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
# Including all splits (Train/Test/Val) to ensure 100% data usage
BASE_NORMAL = "C:/Personal Data/Kittu/BTech project/chest_xray"
TB_DIR      = "C:/Personal Data/Kittu/BTech project/tb_xray"
IMG_SIZE    = 256

def get_all_paths():
    normal_paths = []
    for split in ["train", "test", "val"]:
        p = os.path.join(BASE_NORMAL, split, "NORMAL")
        if os.path.exists(p):
            normal_paths.extend([os.path.join(p, f) for f in os.listdir(p) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    tb_paths = [os.path.join(TB_DIR, f) for f in os.listdir(TB_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📊 Dataset Stats: {len(normal_paths)} Normal images | {len(tb_paths)} TB images")
    return normal_paths, tb_paths

# --- 2. DUAL-MODE DATA LOADER ---
def load_dataset(mode="RAW"):
    normal_paths, tb_paths = get_all_paths()
    X, y = [], []
    clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for paths, label in [(normal_paths, 0), (tb_paths, 1)]:
        desc = f"📂 Loading {mode} Data (Label {label})"
        for p in tqdm(paths, desc=desc):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                if mode == "CLAHE":
                    img = clahe_tool.apply(img)
                X.append(np.expand_dims(img/255.0, -1).astype('float32'))
                y.append(label)
    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42, stratify=y)

# --- 3. MODEL ARCHITECTURE ---
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

# --- 4. EXECUTION & COMPARISON ---
if __name__ == "__main__":
    results = {}

    for mode in ["RAW", "CLAHE"]:
        print(f"\n--- 🚀 PHASE: TRAINING {mode} MODEL ---")
        X_train, X_test, y_train, y_test = load_dataset(mode=mode)
        
        model = build_model()
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        # Store metrics for comparison
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        results[mode] = {
            'history': history.history,
            'cm': confusion_matrix(y_test, y_pred),
            'acc': history.history['val_accuracy'][-1]
        }
        model.save(f"tb_model_{mode.lower()}.keras")

    # --- 5. VISUAL COMPARATIVE STUDY ---
    plt.figure(figsize=(15, 10))

    # Plot 1: Accuracy Comparison
    plt.subplot(2, 2, 1)
    plt.plot(results['RAW']['history']['val_accuracy'], label='RAW Validation Acc', marker='o')
    plt.plot(results['CLAHE']['history']['val_accuracy'], label='CLAHE Validation Acc', marker='s')
    plt.title('Validation Accuracy: RAW vs CLAHE')
    plt.legend()

    # Plot 2: Loss Comparison
    plt.subplot(2, 2, 2)
    plt.plot(results['RAW']['history']['val_loss'], label='RAW Validation Loss', linestyle='--')
    plt.plot(results['CLAHE']['history']['val_loss'], label='CLAHE Validation Loss', linestyle='--')
    plt.title('Validation Loss: RAW vs CLAHE')
    plt.legend()

    # Plot 3 & 4: Confusion Matrices
    for i, mode in enumerate(['RAW', 'CLAHE']):
        plt.subplot(2, 2, i+3)
        sns.heatmap(results[mode]['cm'], annot=True, fmt='d', cmap='Purples' if mode=="RAW" else "Greens")
        plt.title(f'Confusion Matrix ({mode})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

    print("\n✅ Comparative Study Complete. Models saved as .keras files.")