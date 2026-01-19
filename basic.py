import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Here we show rgb-image in RGB-color-space
rgb_img = Image.open("images/image.jpg") 
plt.imshow(rgb_img)