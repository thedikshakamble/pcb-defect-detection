import os
import glob
import random

image_extensions = ('*_test.jpg', '*_temp.jpg')
image_files = []

# Collect all image files matching both extensions
for ext in image_extensions:
    image_files.extend(glob.glob(f'./group*/[0-9]*/{ext}'))

print(f"🔍 Found {len(image_files)} candidate images")

valid_pairs = []

for img_path in image_files:
    base = os.path.basename(img_path)  # e.g. 50600012_test.jpg
    group_dir = os.path.dirname(img_path)               # e.g. ./group50600/50600
    parent_group = os.path.basename(os.path.dirname(group_dir))  # e.g. 50600
    group_base = os.path.basename(group_dir)            # e.g. 50600

    # Remove _test/_temp suffix to get the base ID
    base_id = base.replace('_test.jpg', '').replace('_temp.jpg', '')

    # Build label path: ./groupXXXXX/XXXXX_not/xxxxx.txt
    label_path = os.path.join(os.path.dirname(group_dir), f"{group_base}_not", f"{base_id}.txt")

    if os.path.exists(label_path):
        valid_pairs.append((img_path, label_path))

print(f"✅ Found {len(valid_pairs)} valid image-label pairs.")

# Shuffle and split
random.shuffle(valid_pairs)
split_index = int(0.8 * len(valid_pairs))
trainval, test = valid_pairs[:split_index], valid_pairs[split_index:]

# Save to files
with open("trainval.txt", "w") as f:
    for img, lbl in trainval:
        f.write(f"{img} {lbl}\n")

with open("test.txt", "w") as f:
    for img, lbl in test:
        f.write(f"{img} {lbl}\n")

print(f"✅ Saved {len(trainval)} to trainval.txt and {len(test)} to test.txt.")
