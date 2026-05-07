import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "dataset"

#actions = ["YES", "NO", "OK", "HELLO", "STOP", "PEACE", "CALL_ME", "ROCK"]
actions = ["YES", "NO", "OK", "HELLO", "STOP"]

X = []
y = []

print("Loading dataset...")

for label in actions:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"Missing folder: {folder}")
        continue

    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)

        try:
            data = np.load(filepath)

            if data.shape[0] == 63:  # safety check
                X.append(data)
                y.append(label)

        except:
            print(f"Error loading {filepath}")

X = np.array(X)
y = np.array(y)

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    print("ERROR: No data found")
    exit()

print("Training model...")

model = RandomForestClassifier(n_estimators=300)
model.fit(X, y)

print("Model trained")

joblib.dump(model, "gesture_model.pkl")

print("Model saved as gesture_model.pkl")