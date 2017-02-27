import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import glob

import lib.vgg19 as vgg19
from lib.style_transfer import StyleTransfer

def generate(content_path, style_path, content_weight, style_weight, variational_weight, steps, max_dim):
    content_img = cv2.imread(content_path).astype(np.uint8)
    style_img = cv2.imread(style_path).astype(np.uint8)
    vgg = vgg19.Vgg19('input/vgg/vgg19_conv.npy', True)

    content = content_path.split("/")[-1]
    style = style_path.split("/")[-1].split(".")[0]
    print("Generating content {}, style {}".format(content, style))

    transfer = StyleTransfer(content_img, style_img, vgg, max_dim)
    combined_img, combined_checkpoints = transfer.generate_combined_image(content_weight, style_weight, variational_weight, steps)
    cv2.imwrite('output/combined_{}_{}.jpg'.format(content, style), combined_img)

generate(
    'input/images/content/diana_cropped.png',
    'input/images/art/rain_princess_afremov.jpg',
    0.001, 2, 1, 20, 512
)

generate(
    'input/images/content/diana_cropped.png',
    'input/images/art/guitars_picasso.jpg',
    0.001, 2, 1, 20, 512
)

generate(
    'input/images/content/diana_cropped.png',
    'input/images/art/starry_night_van_gogh.jpg',
    0.0001, 2, 1, 20, 512
)

generate(
    'input/images/portrait/diana_face.png',
    'input/images/portrait/portrait_chuck_close.jpg',
    0.0001, 2, 1, 20, 512
)
