import csv
import glob
import os

labels = {}

with open("data/tc_subset.csv", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = os.path.basename(row["WavPath"])
        labels[filename] = (row["Crackles"], row["Wheezes"])

files = glob.glob("data/tc_audio/*.wav")

matched = [
    os.path.basename(f)
    for f in files
    if os.path.basename(f) in labels
]

print("WAV files:", len(files))
print("Labels matched:", len(matched))
print("Missing labels:", len(files) - len(matched))

if matched:
    print("Example:", matched[0])
    print("Label:", labels[matched[0]])