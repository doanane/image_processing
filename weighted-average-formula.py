import cv2
import numpy as np
from pathlib import Path

# Load the color image from the images folder next to this script
image_path = Path(__file__).resolve().parent / 'images' / 'image2.jpg'
img = cv2.imread(str(image_path))

if img is None:
	raise FileNotFoundError(f"Could not load image from: {image_path}")

# Split into Blue, Green, Red (OpenCV loads as BGR)
B = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
R = img[:, :, 2].astype(np.float32)

# Apply the weighted average formula
gray_manual = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Display result (optional)
cv2.imshow('Manual Grayscale', gray_manual)
cv2.waitKey(0)
cv2.destroyAllWindows() #this will help me close the window after a key is pressed