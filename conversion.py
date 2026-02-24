import cv2
import matplotlib.pyplot as plt
import numpy as np

image_bgr = cv2.imread('images/image.jpg')
if image_bgr is None:
    print("Error: Image not found.")
else:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_bgr)
    plt.title('Incorrect: Direct BGR in Matplotlib')
    plt.axis('off')

    image_rgb_cv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb_cv)
    plt.title('Correct: cv2.cvtColor(BGR2RGB)')
    plt.axis('off')

# if I prefer to use NumPy slicing to convert BGR to RGB, I can do it like this:
    image_rgb_np = image_bgr[:, :, ::-1]
    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb_np)
    plt.title('Correct (NumPy equivalent: [:, :, ::-1])')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("Success: The image is now displayed with correct colors.")