import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data.jpg', cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.title('Pixel Intensity Distribution')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
plt.axes('off')