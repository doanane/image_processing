"""
File: 02_opencv_intro.py
Title: OpenCV Basics - Reading, Displaying, and Saving Images
Description: Learn how to handle images with OpenCV
"""

import cv2  # OpenCV library
import numpy as np  # NumPy for array operations
import matplotlib.pyplot as plt  # For displaying images
import os  # For file operations

print("=== OpenCV Image Handling Basics ===\n")

# ============================================================================
# SECTION 1: READING IMAGES
# ============================================================================

print("1. Reading Images")

# Create a sample image directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Now read images with OpenCV
# cv2.imread() reads an image from file
# Parameters: filename, flag
# Flags: 
#   cv2.IMREAD_COLOR (1): Loads color image (default)
#   cv2.IMREAD_GRAYSCALE (0): Loads grayscale
#   cv2.IMREAD_UNCHANGED (-1): Loads as-is

# Read image.jpg as color
img1 = cv2.imread('images/image.jpg', cv2.IMREAD_COLOR)
if img1 is not None:
    print(f"image.jpg loaded - Shape: {img1.shape}, Data type: {img1.dtype}")
else:
    print("Could not load image.jpg")

# Read image2.jpg as color
img2 = cv2.imread('images/image2.jpg', cv2.IMREAD_COLOR)
if img2 is not None:
    print(f"image2.jpg loaded - Shape: {img2.shape}, Data type: {img2.dtype}")
else:
    print("Could not load image2.jpg")

# Read gradient as grayscale
img_gray = cv2.imread('images/gradient.png', cv2.IMREAD_GRAYSCALE)
if img_gray is not None:
    print(f"Grayscale gradient shape: {img_gray.shape}")
    print(f"Image data type: {img_gray.dtype}")

# Read gradient as color
img_color = cv2.imread('images/gradient.png', cv2.IMREAD_COLOR)
if img_color is not None:
    print(f"Color gradient shape: {img_color.shape}")

# ============================================================================
# SECTION 2: DISPLAYING IMAGES WITH OPENCV
# ============================================================================

print("\n2. Displaying Images with OpenCV")

# Display the first uploaded image
if img1 is not None:
    cv2.imshow('Uploaded Image 1', img1)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display the second uploaded image
if img2 is not None:
    cv2.imshow('Uploaded Image 2', img2)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display gradient image
if img_color is not None:
    cv2.imshow('Gradient Image', img_color)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================================================================
# SECTION 3: SAVING IMAGES
# ============================================================================

print("\n3. Saving Images")

# Create a resized version of uploaded image1
if img1 is not None:
    # Resize to smaller dimensions
    resized_img1 = cv2.resize(img1, (200, 150))
    cv2.imwrite('images/image_resized.jpg', resized_img1)
    print("Saved resized 'images/image_resized.jpg'")
    file_size = os.path.getsize('images/image_resized.jpg')
    print(f"File size: {file_size} bytes")

# Create a grayscale version of uploaded image2
if img2 is not None:
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images/image2_gray.jpg', gray_img2)
    print("Saved grayscale 'images/image2_gray.jpg'")
    file_size = os.path.getsize('images/image2_gray.jpg')
    print(f"File size: {file_size} bytes")

# ============================================================================
# SECTION 4: BASIC IMAGE PROPERTIES
# ============================================================================

print("\n4. Image Properties")

# Analyze uploaded images
if img1 is not None:
    print(f"\nImage 1 Properties:")
    print(f"  Shape: {img1.shape}")
    print(f"  Height: {img1.shape[0]} pixels")
    print(f"  Width: {img1.shape[1]} pixels")
    if len(img1.shape) == 3:
        print(f"  Channels: {img1.shape[2]}")
    print(f"  Total pixels: {img1.size}")
    print(f"  Data type: {img1.dtype}")
    
    # Get pixel values from different locations
    h, w = img1.shape[:2]
    print(f"  Pixel at ({h//4}, {w//4}): {img1[h//4, w//4]}")
    print(f"  Pixel at ({h//2}, {w//2}): {img1[h//2, w//2]}")
    print(f"  Pixel at ({3*h//4}, {3*w//4}): {img1[3*h//4, 3*w//4]}")

if img2 is not None:
    print(f"\nImage 2 Properties:")
    print(f"  Shape: {img2.shape}")
    print(f"  Height: {img2.shape[0]} pixels")
    print(f"  Width: {img2.shape[1]} pixels")
    if len(img2.shape) == 3:
        print(f"  Channels: {img2.shape[2]}")
    print(f"  Total pixels: {img2.size}")
    print(f"  Data type: {img2.dtype}")

# ============================================================================
# SECTION 5: COLOR SPACES IN OPENCV
# ============================================================================

print("\n5. Color Spaces")

# Work with uploaded images
if img1 is not None:
    # Convert BGR to HSV
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    print("Converted image 1 from BGR to HSV")
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # Get a sample pixel from the image
    h, w = img1.shape[:2]
    pixel_bgr = img1[h//2, w//2]
    pixel_rgb = img1_rgb[h//2, w//2]
    print(f"  Sample pixel in BGR: {pixel_bgr}")
    print(f"  Same pixel in RGB: {pixel_rgb}")
    
    # Display both versions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1 - RGB View')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img1_hsv)
    plt.title('Image 1 - HSV View')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# SECTION 6: PRACTICE EXERCISE
# ============================================================================

print("\n6. Practice Exercise - Image Blending")

"""
Exercise: Blend two uploaded images together
"""

if img1 is not None and img2 is not None:
    # Resize images to the same dimensions for blending
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    
    # Blend the images with alpha blending
    alpha = 0.6  # Weight for first image
    beta = 1 - alpha  # Weight for second image
    blended = cv2.addWeighted(img1, alpha, img2_resized, beta, 0)
    
    # Save the blended image
    cv2.imwrite('images/blended_images.jpg', blended)
    print(f"Created blended image with alpha={alpha}")
    print("Saved 'images/blended_images.jpg'")
    
    # Display the blend
    cv2.imshow('Blended Images', blended)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
elif img1 is not None:
    print("Image 2 not available for blending exercise")
else:
    print("Images not available for blending exercise")

# ============================================================================
# SECTION 7: ERROR HANDLING
# ============================================================================

print("\n7. Error Handling in OpenCV")

# Safe image loading function
def load_image_safely(filepath):
    """Load an image with error checking"""
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Could not load image from {filepath}")
        return None
    print(f"Successfully loaded {filepath}")
    print(f"  Shape: {img.shape}")
    print(f"  Size: {img.size} pixels")
    return img

# Test the function with various images
print("\nTesting safe loading:")
test_gradient = load_image_safely('images/gradient.png')
test_french = load_image_safely('images/french_flag.png')
test_checkerboard = load_image_safely('images/checkerboard.jpg')

print("\n=== Key Takeaways ===")
print("1. cv2.imread() loads images with BGR format")
print("2. Always check if imread() returned None before processing")
print("3. cv2.imshow() displays images, cv2.waitKey() waits for input")
print("4. cv2.cvtColor() converts between different color spaces")
print("5. cv2.resize() and cv2.addWeighted() for image manipulation")
print("6. cv2.imwrite() saves processed images to disk")