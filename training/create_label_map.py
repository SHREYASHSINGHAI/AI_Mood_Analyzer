import json
import os

# Define paths
EMOTIONS_FILE = os.path.join("data", "mapping", "emotions.txt")
ARTIFACTS_DIR = os.path.join("artifacts")

LABEL_MAP_FILE = os.path.join(ARTIFACTS_DIR, "label_map.json")

# Load emotions
with open(EMOTIONS_FILE, "r") as f:
    emotions = [line.strip() for line in f.readlines()]

# Create mapping
label_map = {i: label for i, label in enumerate(emotions)}

# Save label map
with open(LABEL_MAP_FILE, "w") as f:
    json.dump(label_map, f, indent=4)

print(f"Label map saved to {LABEL_MAP_FILE}")
