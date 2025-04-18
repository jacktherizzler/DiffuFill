import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# generate random mask
def generate_random_mask(height, width, max_size=0.5):
    mask = np.zeros((height, width), dtype=np.float32)
    size = int(min(height, width) * np.random.uniform(0.1, max_size))
    x = np.random.randint(0, width - size)
    y = np.random.randint(0, height - size)
    mask[y:y+size, x:x+size] = 1.0
    return mask

# generate center mask
def generate_center_mask(height, width, size_ratio=0.5):
    mask = np.zeros((height, width), dtype=np.float32)
    size = int(min(height, width) * size_ratio)
    center_x, center_y = width // 2, height // 2
    start_x = center_x - size // 2
    start_y = center_y - size // 2
    mask[start_y:start_y+size, start_x:start_x+size] = 1.0
    return mask

# load and preprocess image
def load_image(image_path, size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img

# save image
def save_image(image, path):
    image = (image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

# calculate psnr
def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

# calculate ssim
def calculate_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True, data_range=1.0, channel_axis=2)

# convert image to tensor
def to_tensor(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img