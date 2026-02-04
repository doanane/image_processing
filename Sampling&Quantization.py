import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/dog.png")

img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray =cv2.imread("images/dog.png", cv2.IMREAD_GRAYSCALE)

print("data type", img.dtype)
print("min pixel:", img.min())
print("max pixel:", img.max())

print("Image shape:", img.shape)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.axis("off")
plt.imshow(img_rgb)


plt.subplot(1,2,2)
plt.title("Original Grayscale Image")
plt.axis("off")
plt.imshow(img_gray, cmap='gray')
plt.show()