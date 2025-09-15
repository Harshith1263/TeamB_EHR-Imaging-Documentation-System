import os
from PIL import Image
from image_processing import downscale_upscale, extract_patches, augment_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MILESTONE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
TRAIN_DIR = os.path.join(MILESTONE_DIR, "train")
LR_DIR = os.path.join(TRAIN_DIR, "LR")
HR_DIR = os.path.join(TRAIN_DIR, "HR")

os.makedirs(LR_DIR, exist_ok=True)
os.makedirs(HR_DIR, exist_ok=True)

PATCH_SIZE = 64
STRIDE = 32

def prepare_train_data():
    images = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith('.png')]

    if len(images) == 0:
        print("No images found in train folder")
        return

    idx = 0
    for img_file in images:
        img_path = os.path.join(TRAIN_DIR, img_file)
        hr = Image.open(img_path).convert('RGB')
        lr = downscale_upscale(hr)

        # Save full original images
        hr.save(os.path.join(HR_DIR, f"img_{idx:04d}_HR.png"))
        lr.save(os.path.join(LR_DIR, f"img_{idx:04d}_LR.png"))

        # Extract patches including edges
        hr_patches = extract_patches(hr, PATCH_SIZE, STRIDE)
        lr_patches = extract_patches(lr, PATCH_SIZE, STRIDE)

        # Save original + augmented patches
        for i in range(len(hr_patches)):
            # Save original patch
            hr_patches[i].save(os.path.join(HR_DIR, f"img_{idx:04d}_patch_{i:03d}_HR.png"))
            lr_patches[i].save(os.path.join(LR_DIR, f"img_{idx:04d}_patch_{i:03d}_LR.png"))

            # Save augmented patch
            hr_aug = augment_image(hr_patches[i])
            lr_aug = augment_image(lr_patches[i])
            hr_aug.save(os.path.join(HR_DIR, f"img_{idx:04d}_patch_{i:03d}_HR_aug.png"))
            lr_aug.save(os.path.join(LR_DIR, f"img_{idx:04d}_patch_{i:03d}_LR_aug.png"))

        idx += 1

if __name__ == "__main__":
    prepare_train_data()
