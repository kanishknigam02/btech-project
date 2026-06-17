import os
import cv2
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- USING YOUR CONFIRMED STABLE IMPORTS ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation

# --- 1. GLOBAL CONFIGURATION ---
BASE_DIR = "C:/Personal Data/Kittu/BTech project/chest_xray"
TB_DIR   = "C:/Personal Data/Kittu/BTech project/tb_xray"
# NEW FOLDER: To keep RAW and CLAHE results separate
OUT_DIR  = "C:/Personal Data/Kittu/BTech project/New folder/RAW_Specialist_Reports"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 256
MAX_SAMPLES_PER_CLASS = 1000 

def build_specialist_cnn():
    model = Sequential([
        RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        RandomRotation(0.1),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. DATA ENGINE (RAW - NO ENHANCEMENT) ---
def load_raw_data(disease_name="TB"):
    X, y = [], []
    paths = {"NORMAL": [], "DISEASE": []}
    valid_ext = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    for split in ["train", "test", "val"]:
        p = os.path.join(BASE_DIR, split, "NORMAL")
        if os.path.exists(p):
            paths["NORMAL"].extend([os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(valid_ext)])

    if disease_name == "TB":
        paths["DISEASE"] = [os.path.join(TB_DIR, f) for f in os.listdir(TB_DIR) if f.lower().endswith(valid_ext)]
    else:
        for split in ["train", "test", "val"]:
            p = os.path.join(BASE_DIR, split, "PNEUMONIA")
            if os.path.exists(p):
                paths["DISEASE"].extend([os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(valid_ext)])

    min_count = min(len(paths["NORMAL"]), len(paths["DISEASE"]), MAX_SAMPLES_PER_CLASS)
    random.seed(42)
    paths["NORMAL"] = random.sample(paths["NORMAL"], min_count)
    paths["DISEASE"] = random.sample(paths["DISEASE"], min_count)
    
    print(f"📊 {disease_name} RAW Experiment: Using {min_count} samples per class.")

    for cls, label in [("NORMAL", 0), ("DISEASE", 1)]:
        for p in tqdm(paths[cls], desc=f"Loading RAW {disease_name}"):
            img = cv2.imread(p, 0)
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # --- CLAHE REMOVED HERE ---
            X.append(np.expand_dims(img/255.0, -1).astype('float32'))
            y.append(label)
            
    return train_test_split(np.array(X), np.array(y), test_size=0.2, stratify=y, random_state=42)

# --- 3. EXECUTION ---
def execute_raw_training(name):
    print(f"\n🚀 PHASE: Training RAW {name} Specialist...")
    X_train, X_test, y_train, y_test = load_raw_data(name)
    
    model = build_specialist_cnn()
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=12, batch_size=32, callbacks=[stop])
    
    # Save Graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.title(f'RAW {name} Acc'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.title(f'RAW {name} Loss'); plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f"RAW_History_{name}.png"))
    
    # Save Heatmap
    preds = (model.predict(X_test) > 0.5).astype("int32")
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', name], yticklabels=['Normal', name])
    plt.title(f'RAW Diagnostic Matrix: {name}')
    plt.savefig(os.path.join(OUT_DIR, f"RAW_Heatmap_{name}.png"))
    
    with open(os.path.join(OUT_DIR, f"RAW_Report_{name}.txt"), "w") as f:
        f.write(classification_report(y_test, preds, target_names=['Normal', name], labels=[0, 1]))

    model.save(os.path.join(OUT_DIR, f"specialist_raw_{name.lower()}.keras"))
    del X_train, X_test, y_train, y_test, model
    gc.collect()

if __name__ == "__main__":
    execute_raw_training("TB")
    execute_training("PNEUMONIA") if "execute_training" in globals() else execute_raw_training("PNEUMONIA")