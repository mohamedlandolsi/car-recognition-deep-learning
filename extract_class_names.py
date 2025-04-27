import scipy.io
import json
import numpy as np

# Load the MATLAB file
print("Loading cars_annos.mat file...")
mat = scipy.io.loadmat('cars_annos.mat')

# Extract the class names
print("Extracting class names...")
class_names = [name[0] for name in mat['class_names'][0]]
print(f"Found {len(class_names)} car classes")

# Print some examples
print("Examples of car class names:")
for i in range(min(5, len(class_names))):
    print(f"  {i}: {class_names[i]}")

# Some indices from your API response
for idx in [75, 96, 53]:
    if idx < len(class_names):
        print(f"Class {idx} is: {class_names[idx]}")
    else:
        print(f"Class {idx} is out of range (max index: {len(class_names)-1})")

# Save to class_names.json
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=4)

print("Saved complete class names to class_names.json")