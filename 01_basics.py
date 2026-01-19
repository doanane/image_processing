"""
File: 01_basics.py
Title: Python & NumPy Basics for Image Processing
Description: Introduction to Python, NumPy arrays, and basic operations
"""

# ============================================================================
# SECTION 1: IMPORTING LIBRARIES
# ============================================================================
import numpy as np  # Import NumPy library with alias 'np'
# Why np? Standard convention, saves typing
# NumPy handles numerical arrays efficiently

import cv2  # Import OpenCV library
# OpenCV (Open Source Computer Vision) for image processing

import matplotlib.pyplot as plt  # Import matplotlib for visualization
# plt is standard alias for pyplot module

# ============================================================================
# SECTION 2: UNDERSTANDING NUMPY ARRAYS (THE CORE OF IMAGES)
# ============================================================================

# Images in Python are represented as NumPy arrays
# Grayscale image: 2D array (height × width)
# Color image: 3D array (height × width × channels)

print("=== Creating Sample Arrays ===")

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D Array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")  # (5,) - 5 elements
print(f"Data type: {arr_1d.dtype}")  # int64 (default)

# Create a 2D array (like a grayscale image)
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])
print(f"\n2D Array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")  # (3, 3) - 3 rows, 3 columns
print(f"Dimensions: {arr_2d.ndim}")  # 2 dimensions

# Create a 3D array (like a color image - RGB)
arr_3d = np.array([[[255, 0, 0], [0, 255, 0]],
                   [[0, 0, 255], [255, 255, 0]]])
print(f"\n3D Array shape: {arr_3d.shape}")  # (2, 2, 3) - 2 rows, 2 cols, 3 channels

# ============================================================================
# SECTION 3: ARRAY OPERATIONS (ESSENTIAL FOR IMAGE PROCESSING)
# ============================================================================

print("\n=== Array Operations ===")

# Create sample arrays for operations
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Element-wise operations
print(f"Array a:\n{a}")
print(f"Array b:\n{b}")
print(f"\nAddition (a + b):\n{a + b}")
print(f"\nMultiplication (a * b):\n{a * b}")  # Element-wise, NOT matrix multiplication
print(f"\nMatrix multiplication (dot product):\n{np.dot(a, b)}")

# Broadcasting (NumPy's powerful feature)
scalar = 10
print(f"\nBroadcasting (a + 10):\n{a + scalar}")

# Reshaping arrays
flat_array = np.array([1, 2, 3, 4, 5, 6])
reshaped = flat_array.reshape(2, 3)
print(f"\nReshaped array:\n{reshaped}")

# ============================================================================
# SECTION 4: IMAGE-LIKE ARRAY CREATION
# ============================================================================

print("\n=== Creating Image-like Arrays ===")

# Create a black image (all zeros)
black_image = np.zeros((5, 5), dtype=np.uint8)
print(f"Black image (5x5):\n{black_image}")

# Create a white image
white_image = np.ones((3, 3), dtype=np.uint8) * 255
print(f"\nWhite image (3x3):\n{white_image}")

# Create a gradient image
gradient = np.array([[0, 64, 128, 192, 255],
                     [0, 64, 128, 192, 255],
                     [0, 64, 128, 192, 255]])
print(f"\nGradient image:\n{gradient}")

# ============================================================================
# SECTION 5: INDEXING AND SLICING (ACCESSING PIXELS)
# ============================================================================

print("\n=== Array Indexing & Slicing ===")

image = np.array([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]])

print(f"Original image:\n{image}")

# Access single pixel
pixel = image[1, 2]  # Row 1, Column 2
print(f"\nPixel at (1,2): {pixel}")

# Access a row
row = image[1, :]  # All columns in row 1
print(f"Row 1: {row}")

# Access a column
column = image[:, 2]  # All rows in column 2
print(f"Column 2: {column}")

# Access a region (Region of Interest - ROI)
roi = image[0:2, 1:3]  # Rows 0-1, Columns 1-2
print(f"\nRegion of Interest:\n{roi}")

# ============================================================================
# SECTION 6: PRACTICE EXERCISE
# ============================================================================

print("\n=== Practice Exercise ===")

"""
Exercise: Create a 5x5 checkerboard pattern
0 255 0 255 0
255 0 255 0 255
0 255 0 255 0
255 0 255 0 255
0 255 0 255 0
"""

# Method 1: Using loops (for understanding)
checkerboard = np.zeros((5, 5), dtype=np.uint8)
for i in range(5):
    for j in range(5):
        if (i + j) % 2 == 0:
            checkerboard[i, j] = 0
        else:
            checkerboard[i, j] = 255

print(f"Checkerboard (loop method):\n{checkerboard}")

# Method 2: NumPy vectorized approach (more efficient)
checkerboard2 = np.indices((5, 5)).sum(axis=0) % 2 * 255
print(f"\nCheckerboard (vectorized method):\n{checkerboard2}")

print("\n=== Key Takeaways ===")
print("1. Images are NumPy arrays")
print("2. Grayscale: 2D array, Color: 3D array")
print("3. Use array[row, col] to access pixels")
print("4. NumPy operations are vectorized (fast)")