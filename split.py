import os
import random
from PIL import Image
from tqdm import tqdm

input_dir = "/media/hdd3/neo/skippocyte_data"
save_dir = "/media/hdd3/neo/skippocyte_data_split"

# create the save_dir if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, "train"))
    os.makedirs(os.path.join(save_dir, "val"))
    os.makedirs(os.path.join(save_dir, "test"))
    os.makedirs(os.path.join(save_dir, "train", "good"))
    os.makedirs(os.path.join(save_dir, "train", "bad"))
    os.makedirs(os.path.join(save_dir, "val", "good"))
    os.makedirs(os.path.join(save_dir, "val", "bad"))
    os.makedirs(os.path.join(save_dir, "test", "good"))
    os.makedirs(os.path.join(save_dir, "test", "bad"))

all_good_files = [
    os.path.join(input_dir, "good", file)
    for file in os.listdir(os.path.join(input_dir, "good"))
]
all_bad_files = [
    os.path.join(input_dir, "bad", file)
    for file in os.listdir(os.path.join(input_dir, "bad"))
]

# split the data
random.shuffle(all_good_files)
random.shuffle(all_bad_files)

print(len(all_good_files), len(all_bad_files))

train_good = all_good_files[: int(0.8 * len(all_good_files))]
val_good = all_good_files[
    int(0.8 * len(all_good_files)) : int(0.9 * len(all_good_files))
]
test_good = all_good_files[int(0.9 * len(all_good_files)) :]

train_bad = all_bad_files[: int(0.8 * len(all_bad_files))]
val_bad = all_bad_files[int(0.8 * len(all_bad_files)) : int(0.9 * len(all_bad_files))]
test_bad = all_bad_files[int(0.9 * len(all_bad_files)) :]

# save the data
print("Saving data...")
print(len(train_good), len(val_good), len(test_good))
for img_path in tqdm(train_good, desc="Saving train good"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "good", os.path.basename(img_path)))

for img_path in tqdm(val_good, desc="Saving val good"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "good", os.path.basename(img_path)))

for img_path in tqdm(test_good, desc="Saving test good"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "good", os.path.basename(img_path)))

for img_path in tqdm(train_bad, desc="Saving train bad"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "bad", os.path.basename(img_path)))

for img_path in tqdm(val_bad, desc="Saving val bad"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "bad", os.path.basename(img_path)))

for img_path in tqdm(test_bad, desc="Saving test bad"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "bad", os.path.basename(img_path)))

print("Done!")
