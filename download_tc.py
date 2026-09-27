from datasets import load_dataset, Audio
import os
import csv

os.makedirs("data/tc_audio", exist_ok=True)

wanted = set()

with open("data/tc_subset.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        wanted.add(row["WavPath"])

print("Target recordings:", len(wanted))

ds = load_dataset(
    "TwinkStart/RespiratorySound",
    split="respiratory_crackles",
    streaming=True
)

ds = ds.cast_column("audio", Audio(decode=False))

downloaded = 0

for x in ds:
    if x["WavPath"] not in wanted:
        continue

    filename = os.path.basename(x["WavPath"])
    output_path = os.path.join("data", "tc_audio", filename)

    with open(output_path, "wb") as f:
        f.write(x["audio"]["bytes"])

    downloaded += 1
    print(f"Downloaded {downloaded}/40: {filename}")

    if downloaded == 40:
        break

print("TOTAL DOWNLOADED:", downloaded)