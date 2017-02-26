import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

import lib.vgg19 as vgg19
from lib.style_transfer import StyleTransfer

# Load style and content images
style_path = 'input/images/art/starry_night_van_gogh.jpg'
content_path = 'input/images/content/tubingen.jpg'
style_img = cv2.imread(style_path).astype(np.uint8)
content_img = cv2.imread(content_path).astype(np.uint8)

# Load conv only layers
vgg = vgg19.Vgg19('input/vgg/vgg19_conv.npy', True)

transfer = StyleTransfer(content_img, style_img, vgg, 1024)
combined_img, combined_checkpoints = transfer.generate_combined_image(0.001, 1, 1, 1000)
cv2.imwrite('output/combined.jpg', combined_img)
