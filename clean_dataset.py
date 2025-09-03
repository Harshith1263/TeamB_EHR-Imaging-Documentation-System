import os
import cv2
import hashlib
import shutil
import pandas as pd
import random
import zipfile

# Paths
IMAGES_DIR = "data/images"
CLEANED_IMAGES_DIR = "data/images_cleaned"
CSV_DIR = IMAGES_DIR  # all CSVs are inside images folder
CLEANED_CSV_DIR = "data/csv_cleaned"
os.makedirs(CLEANED_IMAGES_DIR, exist_ok=True)
os.makedirs(CLEANED_CSV_DIR, exist_ok=True)

# Image cleaning parameters
TARGET_SIZE = (224, 224)
SUPPORTED_EXT = ['.jpg', '.jpeg', '.png']

# Logging
cleaned_count = 0
failed_count = 0
duplicates_skipped = 0
hashes = set()


for root, dirs, files in os.walk(IMAGES_DIR):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in SUPPORTED_EXT:
            continue

        orig_path = os.path.join(root, file)
        rel_path = os.path.relpath(root, IMAGES_DIR)
        target_folder = os.path.join(CLEANED_IMAGES_DIR, rel_path)
        os.makedirs(target_folder, exist_ok=True)
        # Save as PNG now
        target_path = os.path.join(target_folder, os.path.splitext(file)[0] + ".png")

        try:
            # Read image
            img = cv2.imread(orig_path)
            if img is None or img.size == 0:
                failed_count += 1
                continue

            # Check for duplicates using PNG encoding
            img_hash = hashlib.md5(cv2.imencode('.png', img)[1].tobytes()).hexdigest()
            if img_hash in hashes:
                duplicates_skipped += 1
                continue
            hashes.add(img_hash)

            # Resize and save as PNG
            img_resized = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(target_path, img_resized)
            cleaned_count += 1

        except Exception as e:
            failed_count += 1
            continue


# Auto-balance classes (works recursively for subfolders)
for modality_root, subdirs, _ in os.walk(CLEANED_IMAGES_DIR):
    for subdir in subdirs:
        folder_path = os.path.join(modality_root, subdir)
        images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not images:
            continue
        sibling_folders = [os.path.join(modality_root, d) for d in subdirs]
        sibling_counts = [len([f for f in os.listdir(s) if os.path.isfile(os.path.join(s, f))]) for s in sibling_folders]
        min_count = min(sibling_counts)
        for s_folder in sibling_folders:
            img_files = [f for f in os.listdir(s_folder) if os.path.isfile(os.path.join(s_folder, f))]
            if len(img_files) > min_count:
                sampled = random.sample(img_files, min_count)
                for f in img_files:
                    if f not in sampled:
                        os.remove(os.path.join(s_folder, f))


# CSV Processing
csv_files = ["200k_patients_EHR_datasets.csv", "leukemia_ehr_full.csv", "cxr_df.csv.zip"]
merged_df = pd.DataFrame()

for csv_file in csv_files:
    csv_path = os.path.join(CSV_DIR, csv_file)
    if not os.path.exists(csv_path):
        continue

    try:
        if csv_file.endswith(".zip"):
            with zipfile.ZipFile(csv_path, 'r') as zip_ref:
                zip_ref.extractall(CLEANED_CSV_DIR)
                extracted_files = zip_ref.namelist()
                for f in extracted_files:
                    df = pd.read_csv(os.path.join(CLEANED_CSV_DIR, f))
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
        else:
            df = pd.read_csv(csv_path)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    except Exception as e:
        print(f" Failed to read {csv_file}: {e}")

if not merged_df.empty:
    merged_df.drop_duplicates(inplace=True)
    merged_df.to_csv(os.path.join(CLEANED_CSV_DIR, "final_merged.csv"), index=False)
    print(f" CSVs merged: {len(merged_df)} rows")
else:
    print(" No CSVs found to merge")

