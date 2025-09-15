import os
import torch
from torchvision import transforms
from PIL import Image
from srcnn_architecture import SRCNN

# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MILESTONE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
TEST_DIR = os.path.join(MILESTONE_DIR, "test")
OUTPUT_DIR = os.path.join(MILESTONE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained model
MODEL_PATH = os.path.join(MILESTONE_DIR, "models", "srcnn_trained_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transforms 
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def downscale_upscale(img, lr_size=(64, 64), hr_size=(256, 256)):
    # Simulate low-resolution by downscaling then upscaling
    lr = img.resize(lr_size, Image.BICUBIC)
    return lr.resize(hr_size, Image.BICUBIC), lr

def run_inference():
    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".png")]
    if not images:
        print("No test images found in folder.")
        return

    for idx, img_file in enumerate(images):
        img_path = os.path.join(TEST_DIR, img_file)
        print(f"Processing: {img_path}")
        hr = Image.open(img_path).convert("RGB")

        # Generate LR and bicubic-upsampled version
        lr_upscaled, lr_only = downscale_upscale(hr)

        # Run through SRCNN
        lr_tensor = to_tensor(lr_upscaled).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr_img = to_pil(sr_tensor.squeeze(0).cpu().clamp(0, 1))

        # Save side-by-side comparison
        comparison = Image.new("RGB", (hr.width * 3, hr.height))
        comparison.paste(hr, (0, 0))          # Original HR
        comparison.paste(lr_upscaled, (hr.width, 0))  # Bicubic LR->HR
        comparison.paste(sr_img, (hr.width * 2, 0))   # SRCNN output

        out_path = os.path.join(OUTPUT_DIR, f"comparison_{idx:03d}.png")
        comparison.save(out_path)

if __name__ == "__main__":
    run_inference()
