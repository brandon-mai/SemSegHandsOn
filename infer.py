import argparse
import os

import cv2
from PIL import Image
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from best_model import ResNetUNet


# Define the argument parser
parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
args = parser.parse_args()


# Load the model checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('best_model.pth', map_location=device)
model = ResNetUNet(num_classes=3)
model.load_state_dict(checkpoint['model'])
model.to(device)


# Definitions
img_path = args.image_path
img_size = 256

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for k in color_dict.keys():
        output[mask==k] = color_dict[k]
    return np.uint8(output)


# Inference
model.eval()

ori_img = cv2.imread(img_path)
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
ori_w = ori_img.shape[0]
ori_h = ori_img.shape[1]
img = cv2.resize(ori_img, (img_size, img_size))
transformed = val_transform(image=img)
input_img = transformed["image"]
input_img = input_img.unsqueeze(0).to(device)

with torch.no_grad():
    output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
mask = cv2.resize(output_mask, (ori_h, ori_w))
mask = np.argmax(mask, axis=2)
mask_rgb = mask_to_rgb(mask, color_dict)
mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_{}".format(img_path), mask_bgr)


# Plotting
ori_img_rgba = Image.fromarray(ori_img).convert("RGBA")
mask_rgba = Image.fromarray(mask_rgb).convert("RGBA")
mask_rgba.putalpha(128)
overlay_img = Image.alpha_composite(ori_img_rgba, mask_rgba)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(ori_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_rgb)
plt.title('Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay_img)
plt.title('Image with Mask Overlay')
plt.axis('off')

plt.show()