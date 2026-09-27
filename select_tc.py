from datasets import load_dataset

ds = load_dataset(
    "TwinkStart/RespiratorySound",
    split="respiratory_crackles",
    streaming=True,
    columns=["WavPath", "Crackles", "Wheezes"]
)

targets = {
    ("absent", "absent"): 10,
    ("present", "absent"): 10,
    ("absent", "present"): 10,
    ("present", "present"): 10,
}

counts = {key: 0 for key in targets}
rows = []

for x in ds:
    if "Tc" not in x["WavPath"]:
        continue

    label = (x["Crackles"], x["Wheezes"])

    if label in targets and counts[label] < targets[label]:
        rows.append(x)
        counts[label] += 1

    if all(counts[key] >= targets[key] for key in targets):
        break

with open("data/tc_subset.csv", "w", newline="") as f:
    f.write("WavPath,Crackles,Wheezes\n")

    for x in rows:
        f.write(
            f'{x["WavPath"]},{x["Crackles"]},{x["Wheezes"]}\n'
        )

print("Saved:", len(rows), "records")
print("Counts:", counts)