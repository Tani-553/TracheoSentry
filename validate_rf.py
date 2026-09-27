import csv
import glob
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from ml.features import extract_features_from_path


# Load labels
labels = {}

with open("data/tc_subset.csv", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = os.path.basename(row["WavPath"])

        # Convert respiratory labels into four test classes
        crackles = row["Crackles"]
        wheezes = row["Wheezes"]

        if crackles == "absent" and wheezes == "absent":
            label = "normal"
        elif crackles == "present" and wheezes == "absent":
            label = "crackles"
        elif crackles == "absent" and wheezes == "present":
            label = "wheezes"
        else:
            label = "both"

        labels[filename] = label


# Extract features
files = glob.glob("data/tc_audio/*.wav")

X = []
y = []

for path in files:
    filename = os.path.basename(path)

    if filename not in labels:
        continue

    X.append(extract_features_from_path(path))
    y.append(labels[filename])

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Classes:", sorted(set(y)))


# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


# Same RF configuration as the existing project
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions))