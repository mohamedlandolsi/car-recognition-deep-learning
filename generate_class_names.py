import os
import json

# Get all class names from the train folder
train_path = 'organized_cars_dataset/train'
class_names = sorted(os.listdir(train_path))

print(f'Found {len(class_names)} classes')

# Save to class_names.json
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=4)

print('Saved class names to class_names.json')
print('First 5 classes:', class_names[:5])