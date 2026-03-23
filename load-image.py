# 20. Write a code to load an image and plot the distribution of pixel intensities of an image.

import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale for intensity distribution
img = cv2.imread('data.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the histogram (distribution of intensities)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Plot the histogram
plt.figure()
plt.title('Pixel Intensity Distribution')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()