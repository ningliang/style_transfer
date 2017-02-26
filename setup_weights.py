import numpy as np
import re
import math

from functools import reduce

# VGG parameters
vgg_params = np.load('input/vgg/vgg19.npy', encoding='latin1').item()

# Count the number of parameters
count = 0
conv_count = 0
for key, value in vgg_params.items():
    for param in value:
        product = reduce(lambda x, y: x * y, param.shape)
        count += product
        if key.startswith('conv'):
            conv_count += product

print("{:,d} params, {:,d} conv params".format(count, conv_count))

# Output just the conv layers to a npy file
conv_dict = { key: value for key, value in vgg_params.items() if not key.startswith('fc') }
np.save(open('input/vgg/vgg19_conv.npy', 'wb'), conv_dict)
