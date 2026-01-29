"""
File: 01_basics.py
Title: Python & NumPy Basics for Image Processing
Description: Introduction to Python, NumPy arrays, and basic operations
"""




import numpy as np  



import cv2  


import matplotlib.pyplot as plt  










print("=== Creating Sample Arrays ===")


arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D Array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")  
print(f"Data type: {arr_1d.dtype}")  


arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])
print(f"\n2D Array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")  
print(f"Dimensions: {arr_2d.ndim}")  


arr_3d = np.array([[[255, 0, 0], [0, 255, 0]],
                   [[0, 0, 255], [255, 255, 0]]])
print(f"\n3D Array shape: {arr_3d.shape}")  










print("\n=== Array Operations ===")


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])


print(f"Array a:\n{a}")
print(f"Array b:\n{b}")
print(f"\nAddition (a + b):\n{a + b}")
print(f"\nMultiplication (a * b):\n{a * b}")  
print(f"\nMatrix multiplication (dot product):\n{np.dot(a, b)}")


scalar = 10
print(f"\nBroadcasting (a + 10):\n{a + scalar}")


flat_array = np.array([1, 2, 3, 4, 5, 6])
reshaped = flat_array.reshape(2, 3)
print(f"\nReshaped array:\n{reshaped}")





print("\n=== Creating Image-like Arrays ===")


black_image = np.zeros((5, 5), dtype=np.uint8)
print(f"Black image (5x5):\n{black_image}")


white_image = np.ones((3, 3), dtype=np.uint8) * 255
print(f"\nWhite image (3x3):\n{white_image}")


gradient = np.array([[0, 64, 128, 192, 255],
                     [0, 64, 128, 192, 255],
                     [0, 64, 128, 192, 255]])
print(f"\nGradient image:\n{gradient}")

print(f"Shape: {gradient.shape}")  
print(f"Dimensions: {gradient.ndim}")




print("\n=== Array Indexing & Slicing ===")

image = np.array([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]])

print(f"Original image:\n{image}")


pixel = image[0, 2]  
print(f"\nPixel at (0,2): {pixel}")

row = image[1, :]  
print(f"Row 1: {row}")

column = image[:, 2]  
print(f"Column 2: {column}")

roi = image[0:2, 1:3]  
print(f"\nRegion of Interest:\n{roi}")





print("\n=== Practice Exercise ===")

"""
Exercise: Create a 5x5 checkerboard pattern
0 255 0 255 0
255 0 255 0 255
0 255 0 255 0
255 0 255 0 255
0 255 0 255 0
"""


checkerboard = np.zeros((5, 5), dtype=np.uint8)
for i in range(5):
    for j in range(5):
        if (i + j) % 2 == 0:
            checkerboard[i, j] = 0
        else:
            checkerboard[i, j] = 255

print(f"Checkerboard (loop method):\n{checkerboard}")


checkerboard2 = np.indices((5, 5)).sum(axis=0) % 2 * 255
print(f"\nCheckerboard (vectorized method):\n{checkerboard2}")

print("\n=== Key Takeaways ===")
print("1. Images are NumPy arrays")
print("2. Grayscale: 2D array, Color: 3D array")
print("3. Use array[row, col] to access pixels")
print("4. NumPy operations are vectorized (fast)")