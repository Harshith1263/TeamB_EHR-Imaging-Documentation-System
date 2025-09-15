import numpy as np
from PIL import Image, ImageEnhance
import random

def downscale_upscale(image, lr_size=(64,64), hr_size=(256,256)):
    # Downscale and then upscale image using bicubic interpolation
    lr = image.resize(lr_size, Image.BICUBIC)
    sr = lr.resize(hr_size, Image.BICUBIC)
    return sr

def extract_patches(img, patch_size=64, stride=32):
    
    # Extract overlapping patches from image including edges
    patches = []
    img_np = np.array(img)
    h, w, c = img_np.shape

    # Compute starting positions
    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))

    # Include edge patch if last patch is skipped
    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    for y in y_steps:
        for x in x_steps:
            patch = img_np[y:y+patch_size, x:x+patch_size, :]
            patches.append(Image.fromarray(patch))

    return patches

def augment_image(img):
    # Apply random flip, rotation, brightness, and slight noise
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.9, 1.1)
    img = enhancer.enhance(factor)
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 2, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)
