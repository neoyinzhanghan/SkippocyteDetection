import os
import random
from PIL import Image
from tqdm import tqdm

input_dir = "/media/hdd3/neo/blasts_skippocytes_others"
save_dir = "/media/hdd3/neo/blasts_skippocytes_others_split"

# Create the save_dir if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, "train"))
    os.makedirs(os.path.join(save_dir, "val"))
    os.makedirs(os.path.join(save_dir, "test"))
    # Class directories
    os.makedirs(os.path.join(save_dir, "train", "Blasts"))
    os.makedirs(os.path.join(save_dir, "train", "Skippocytes"))
    os.makedirs(os.path.join(save_dir, "train", "OtherWBCs"))
    os.makedirs(os.path.join(save_dir, "val", "Blasts"))
    os.makedirs(os.path.join(save_dir, "val", "Skippocytes"))
    os.makedirs(os.path.join(save_dir, "val", "OtherWBCs"))
    os.makedirs(os.path.join(save_dir, "test", "Blasts"))
    os.makedirs(os.path.join(save_dir, "test", "Skippocytes"))
    os.makedirs(os.path.join(save_dir, "test", "OtherWBCs"))

all_blasts_files = [
    os.path.join(input_dir, "Blasts", file)
    for file in os.listdir(os.path.join(input_dir, "Blasts"))
]
all_skippocytes_files = [
    os.path.join(input_dir, "Skippocytes", file)
    for file in os.listdir(os.path.join(input_dir, "Skippocytes"))
]
all_otherwbc_files = [
    os.path.join(input_dir, "OtherWBCs", file)
    for file in os.listdir(os.path.join(input_dir, "OtherWBCs"))
]

# Shuffle and split the data
random.shuffle(all_blasts_files)
random.shuffle(all_skippocytes_files)
random.shuffle(all_otherwbc_files)

train_blasts = all_blasts_files[:int(0.8 * len(all_blasts_files))]
val_blasts = all_blasts_files[int(0.8 * len(all_blasts_files)):int(0.9 * len(all_blasts_files))]
test_blasts = all_blasts_files[int(0.9 * len(all_blasts_files)):]

train_skippocytes = all_skippocytes_files[:int(0.8 * len(all_skippocytes_files))]
val_skippocytes = all_skippocytes_files[int(0.8 * len(all_skippocytes_files)):int(0.9 * len(all_skippocytes_files))]
test_skippocytes = all_skippocytes_files[int(0.9 * len(all_skippocytes_files)):]

train_otherwbc = all_otherwbc_files[:int(0.8 * len(all_otherwbc_files))]
val_otherwbc = all_otherwbc_files[int(0.8 * len(all_otherwbc_files)):int(0.9 * len(all_otherwbc_files))]
test_otherwbc = all_otherwbc_files[int(0.9 * len(all_otherwbc_files)):]

# Save the data
for img_path in tqdm(train_blasts, desc="Saving train Blasts"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "Blasts", os.path.basename(img_path)))

for img_path in tqdm(val_blasts, desc="Saving val Blasts"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "Blasts", os.path.basename(img_path)))

for img_path in tqdm(test_blasts, desc="Saving test Blasts"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "Blasts", os.path.basename(img_path)))

for img_path in tqdm(train_skippocytes, desc="Saving train Skippocytes"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "Skippocytes", os.path.basename(img_path)))

for img_path in tqdm(val_skippocytes, desc="Saving val Skippocytes"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "Skippocytes", os.path.basename(img_path)))

for img_path in tqdm(test_skippocytes, desc="Saving test Skippocytes"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "Skippocytes", os.path.basename(img_path)))

for img_path in tqdm(train_otherwbc, desc="Saving train OtherWBCs"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "OtherWBCs", os.path.basename(img_path)))

for img_path in tqdm(val_otherwbc, desc="Saving val OtherWBCs"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "OtherWBCs", os.path.basename(img_path)))

for img_path in tqdm(test_otherwbc, desc="Saving test OtherWBCs"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "OtherWBCs", os.path.basename(img_path)))

print("Done!")
