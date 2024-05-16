import os
import shutil
import pandas as pd
from tqdm import tqdm

data_dirs = [
    "/media/ssd2/dh_labelled_data/DeepHeme1/UCSF_repo",
    "/media/ssd2/dh_labelled_data/DeepHeme1/MSK_repo_normal",
    "/media/ssd2/dh_labelled_data/DeepHeme1/MSK_repo_mixed",
    "/media/ssd2/dh_labelled_data/DeepHeme2/PBS/labelled_cartridge_2",
    "/media/ssd2/dh_labelled_data/DeepHeme2/PBS/labelled_cartridge_1",
    "/media/ssd2/dh_labelled_data/DeepHeme2/PBS/labelled_cartridge_0",
    "/media/ssd2/dh_labelled_data/DeepHeme2/BMA/cartridge_1",
    "/media/hdd3/neo/LabelledBMASkippocytes",
    "/media/hdd1/neo/blasts_normal_confirmed",
]
save_dir = "/media/hdd3/neo/blasts_skippocytes_full"

skipped_classes = ["ER5", "ER6", "U4", "U1", "PL2", "PL3"]

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "good"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "bad"), exist_ok=True)

# each data_dir in data_dirs contain a bunch of subdirectories, each containing images, the name of the subdirectory is the label
# if the label is in skipped_classes, the images are shutil copied to the bad directory DO NOT MOVE THEM!
# when saving new file, use indexing 0,1,2, ... for the files, and keep a metadata file with the original path and label, and the idx


metadata = {
    "idx": [],
    "path": [],
    "label": [],
    "class": [],
}

current_idx = 0

for data_dir in tqdm(data_dirs, desc="Processing Data Directories"):
    for label in tqdm(os.listdir(data_dir), desc="Processing Labels"):

        # make sure the label is a directory
        if not os.path.isdir(os.path.join(data_dir, label)):
            continue

        # if label contains other, Other, the label should be moved to bad
        if "other" in label.lower():
            for img in os.listdir(os.path.join(data_dir, label)):

                # make sure the img is an image file
                if not img.endswith(".jpg") and not img.endswith(".png"):
                    continue

                if img.endswith(".png"):
                    old_path = os.path.join(data_dir, label, img)
                    new_path = os.path.join(save_dir, "bad", f"{current_idx}.jpg")
                    os.system(f"convert {old_path} {new_path}")
                else:
                    old_path = os.path.join(data_dir, label, img)
                    new_path = os.path.join(save_dir, "bad", f"{current_idx}.jpg")
                    shutil.copyfile(old_path, new_path)

                metadata["idx"].append(current_idx)
                metadata["path"].append(old_path)
                metadata["label"].append(label)
                metadata["class"].append("bad")

                current_idx += 1

        if label in skipped_classes:
            for img in os.listdir(os.path.join(data_dir, label)):

                # make sure the img is an image file
                if not img.endswith(".jpg") and not img.endswith(".png"):
                    continue

                if img.endswith(".png"):
                    old_path = os.path.join(data_dir, label, img)
                    new_path = os.path.join(save_dir, "bad", f"{current_idx}.jpg")
                    os.system(f"convert {old_path} {new_path}")
                else:
                    old_path = os.path.join(data_dir, label, img)
                    new_path = os.path.join(save_dir, "bad", f"{current_idx}.jpg")
                    shutil.copyfile(old_path, new_path)

                metadata["idx"].append(current_idx)
                metadata["path"].append(old_path)
                metadata["label"].append(label)
                metadata["class"].append("bad")

                current_idx += 1
        elif label == "M1":
            for img in os.listdir(os.path.join(data_dir, label)):

                # make sure the img is an image file, check jpg and png
                if not (img.endswith(".jpg") and not img.endswith(".png")):
                    continue

                # if the file ends with png, convert it to jpg
                if img.endswith(".png"):
                    old_path = os.path.join(data_dir, label, img)
                    new_path = os.path.join(save_dir, "good", f"{current_idx}.jpg")
                    os.system(f"convert {old_path} {new_path}")
                else:
                    old_path = os.path.join(data_dir, label, img)
                    new_path = os.path.join(save_dir, "good", f"{current_idx}.jpg")
                    shutil.copyfile(old_path, new_path)

                metadata["idx"].append(current_idx)
                metadata["path"].append(old_path)
                metadata["label"].append(label)
                metadata["class"].append("good")

                current_idx += 1

        else:
            # skip the label
            continue

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
print("Done.")
