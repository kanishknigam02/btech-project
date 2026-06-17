from sklearn.metrics import classification_report

# Replace these with the actual values from your Confusion Matrix
# RAW: [316, 1, 7, 125] | CLAHE: [317, 0, 4, 128]
print("--- 📄 RAW MODEL REPORT ---")
print(classification_report([0]*317 + [1]*132, [0]*316 + [1]*1 + [0]*7 + [1]*125, target_names=['Normal', 'TB']))

print("\n--- 📄 CLAHE MODEL REPORT ---")
print(classification_report([0]*317 + [1]*132, [0]*317 + [1]*0 + [0]*4 + [1]*128, target_names=['Normal', 'TB']))