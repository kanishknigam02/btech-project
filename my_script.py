import os
import cv2
import gc
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- 1. GLOBAL CONFIGURATION (The "Standard") ---
BASE_DIR = "C:/Personal Data/Kittu/BTech project/chest_xray"
TB_DIR   = "C:/Personal Data/Kittu/BTech project/tb_xray"
# Optimized paths to avoid circular imports and special characters

OUT_DIR  = "C:/Personal Data/Kittu/BTech project/New folder/Comparative_Study_Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 256        # Higher res for competitive study
EPOCHS   = 12         # Sufficient epochs on Ryzen 5
BATCH_SIZE = 32
MAX_SAMPLES_PER_CLASS = 1500 # Optimized for 8GB RAM limit

def build_shared_cnn():
    """Builds an identical model structure for fair comparison."""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax') # Normal, Pneumonia, TB
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 2. THE DATA LOADER ---
# (Handles identical sampling and optional CLAHE)
def load_complete_dataset(use_clahe=False):
    """Loads standardized dataset, optionally applying CLAHE."""
    X, y = [], []
    clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Paths gathering (Simplified paths as finalized previously)
    paths = {"NORMAL": [], "PNEUMONIA": [], "TUBERCULOSIS": []}
    
    # Load standardized Chest X-ray folders
    for split in ["train", "test", "val"]:
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path): continue
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = os.path.join(split_path, cls)
            if os.path.exists(cls_dir):
                files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                paths[cls].extend(files)

    if os.path.exists(TB_DIR):
        tb_files = [os.path.join(TB_DIR, f) for f in os.listdir(TB_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        paths["TUBERCULOSIS"].extend(tb_files)

    # ⚖️ Ensure Identical Downsampling for Fairness
    random.seed(42) # Lock random seed for reproducibility
    for cls in paths:
        if len(paths[cls]) > MAX_SAMPLES_PER_CLASS:
            paths[cls] = random.sample(paths[cls], MAX_SAMPLES_PER_CLASS)
        print(f"⚖️ {cls} samples for experiment: {len(paths[cls])}")

    # Process images loop
    for cls, idx in [("NORMAL", 0), ("PNEUMONIA", 1), ("TUBERCULOSIS", 2)]:
        for p in tqdm(paths[cls], desc=f"Loading {cls} (CLAHE={use_clahe})"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            if use_clahe: img = clahe_tool.apply(img)
            X.append(np.expand_dims(img/255.0, -1).astype('float32'))
            y.append(idx)
    
    # Use stratification for balanced splits
    return train_test_split(np.array(X), np.array(y), test_size=0.2, stratify=y, random_state=42)

# --- 3. THE TRAINING & ANALYSIS LOOP ---
def train_and_evaluate_experiment(experiment_name, X_train, X_test, y_train, y_test):
    """Trains the model and generates a robust visual suite for the experiment."""
    print(f"\n🚀 Running full Experiment: {experiment_name} images...")
    
    model = build_shared_cnn()
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # Save the specialized trained brain
    model.save(os.path.join(OUT_DIR, f"model_{experiment_name}.keras"))
    
    # Generate Raw Predictions
    print("\n📊 Calculating predictions...")
    preds = np.argmax(model.predict(X_test), axis=1)
    
    # 📝 SAVE Visual 1: Epoch Accuracy Line Graph
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy'); plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f"{experiment_name}: CNN Diagnostic Accuracy Curve"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(OUT_DIR, f"1_AccuracyCurve_{experiment_name}.png"))
    
    # 📝 SAVE Visual 2: Epoch Loss Line Graph
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{experiment_name}: CNN Loss Reduction Curve"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(OUT_DIR, f"2_LossCurve_{experiment_name}.png"))

    # 📝 SAVE Visual 3: Standard Confusion Matrix (Heatmap)
    target_names=["Normal", "Pneumonia", "TB"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"{experiment_name}: Diagnostic Confusion Matrix")
    plt.savefig(os.path.join(OUT_DIR, f"3_ConfusionMatrix_{experiment_name}.png"))

    # 📝 SAVE Visual 4: Normalized Confusion Matrix (Heatmap, Percentages)
    plt.figure(figsize=(8, 6))
    cm_norm = confusion_matrix(y_test, preds, normalize='true')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"{experiment_name}: Normalized Confusion Matrix")
    plt.savefig(os.path.join(OUT_DIR, f"4_ConfusionMatrixNorm_{experiment_name}.png"))

    # Generate hard stats for summary
    report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
    report_txt = classification_report(y_test, preds, target_names=target_names)
    with open(os.path.join(OUT_DIR, f"5_ClassificationReport_{experiment_name}.txt"), "w") as f:
        f.write(report_txt)
        
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Sensitivity_P": report["Pneumonia"]["recall"],
        "Sensitivity_TB": report["TB"]["recall"],
        "Precision_P": report["Pneumonia"]["precision"],
        "Precision_TB": report["TB"]["precision"],
        "F1_P": report["Pneumonia"]["f1-score"],
        "F1_TB": report["TB"]["f1-score"],
    }

# --- 4. THE MASTER COMPARATIVE VISUALIZER ---
def generate_maximum_visuals(results_data):
    """Generates a full suite of aggregate comparison bar charts."""
    df = pd.DataFrame(results_data)
    print("\n📊 Generating Master Comparative Visuals...")
    
    # 📊 Visual 6: OVERALL ACCURACY (BAR CHART) - Simple Hero Image
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Experiment", y="Accuracy", data=df, palette="deep")
    plt.title("Impact of CLAHE Enhancement on Diagnostic Accuracy", fontsize=14)
    plt.ylabel("Overall Accuracy (%)"); plt.ylim(0.75, 1.0)
    for i, v in enumerate(df["Accuracy"]): plt.text(i, v + 0.005, f"{v*100:.1f}%", ha='center', fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, "6_SimpleAccuracyComparison_Bar.png"))
    
    # 📊 Visual 7: DISEASE SENSITIVITY (RECALL) (BAR CHART) - Life-Saving Metric
    plt.figure(figsize=(10, 6))
    sensitivity_metrics = ["Sensitivity_P", "Sensitivity_TB"]
    sens_df = df.melt(id_vars="Experiment", value_vars=sensitivity_metrics, var_name="Metric", value_name="Score")
    sns.barplot(x="Metric", y="Score", hue="Experiment", data=sens_df, palette="muted")
    plt.title("Comparative Disease Sensitivity (Ability to Detect Disease)", fontsize=14)
    plt.xticks([0, 1], ["Pneumonia Sensitivity", "Tuberculosis Sensitivity"])
    plt.ylabel("Sensitivity Score (Recall)"); plt.ylim(0.70, 1.0)
    plt.savefig(os.path.join(OUT_DIR, "7_SensitivityComparison_ClusteredBar.png"))

    # 📊 Visual 8: PRECISION (BAR CHART) - Trustworthiness Metric
    plt.figure(figsize=(10, 6))
    precision_metrics = ["Precision_P", "Precision_TB"]
    prec_df = df.melt(id_vars="Experiment", value_vars=precision_metrics, var_name="Metric", value_name="Score")
    sns.barplot(x="Metric", y="Score", hue="Experiment", data=prec_df, palette="pastel")
    plt.title("Comparative Disease Precision (Trustworthiness of Diagnosis)", fontsize=14)
    plt.xticks([0, 1], ["Pneumonia Precision", "Tuberculosis Precision"])
    plt.ylabel("Precision Score"); plt.ylim(0.70, 1.0)
    plt.savefig(os.path.join(OUT_DIR, "8_PrecisionComparison_ClusteredBar.png"))

    # 📊 Visual 9: F1-SCORE IMPROVEMENT (BAR CHART)
    plt.figure(figsize=(10, 6))
    f1_metrics = ["F1_P", "F1_TB"]
    f1_df = df.melt(id_vars="Experiment", value_vars=f1_metrics, var_name="Metric", value_name="Score")
    sns.barplot(x="Metric", y="Score", hue="Experiment", data=f1_df, palette="bright")
    plt.title("Overall Metric F1-Score Improvement (Balance)", fontsize=14)
    plt.xticks([0, 1], ["Pneumonia F1-Score", "Tuberculosis F1-Score"])
    plt.ylabel("F1-Score"); plt.ylim(0.70, 1.0)
    plt.savefig(os.path.join(OUT_DIR, "9_F1ScoreComparison_ClusteredBar.png"))

    plt.close('all') # Cleanup all figure memory

# --- 5. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    print(f"🚀 Dual-Experiment Comparative Pipeline Started. Deliverables in: {OUT_DIR}")
    results_suite = []
    
    # --- 🔵 EXPERIMENT A: RAW ---
    X_train, X_test, y_train, y_test = load_complete_dataset(use_clahe=False)
    results_raw = train_and_evaluate_experiment("A_RAW", X_train, X_test, y_train, y_test)
    results_raw["Experiment"] = "Model A (Raw)"
    results_suite.append(results_raw)
    
    # ⚠️ CRITICAL MEMORY CLEANUP for 8GB RAM 
    print("\n♻️ Clearing Memory after Experiment A...")
    del X_train, X_test, y_train, y_test
    gc.collect() 

    # --- 🟢 EXPERIMENT B: CLAHE ---
    X_train, X_test, y_train, y_test = load_complete_dataset(use_clahe=True)
    results_clahe = train_and_evaluate_experiment("B_CLAHE", X_train, X_test, y_train, y_test)
    results_clahe["Experiment"] = "Model B (CLAHE)"
    results_suite.append(results_clahe)
    
    # Final Visual Aggregation
    generate_maximum_visuals(results_suite)
    
    print(f"\n✅ COMPLETED. {len(os.listdir(OUT_DIR))}Deliverables are in 'Comparative_Study_Outputs'.")