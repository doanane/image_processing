import cv2
import matplotlib.pyplot as plt
from pathlib import Path


image_path = Path(__file__).resolve().parent / 'images' / 'image2.jpg'
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if img is None:
	raise FileNotFoundError(f"Could not load image from: {image_path}")


equalized_img = cv2.equalizeHist(img)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title('Histogram Equalized')
plt.axis('off')

plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
plt.hist(equalized_img.ravel(), 256, [0, 256])
plt.title('Equalized Histogram')

plt.show()