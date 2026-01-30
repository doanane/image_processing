"""
File: 02_opencv_intro.py
Title: OpenCV Basics - Reading, Displaying, and Saving Images
Description: Learn how to handle images with OpenCV
"""

import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
import os  

print("=== OpenCV Image Handling Basics ===\n")





print("1. Reading Images")


os.makedirs('images', exist_ok=True)










img1 = cv2.imread('images/image.jpg', cv2.IMREAD_COLOR)
if img1 is not None:
    print(f"image.jpg loaded - Shape: {img1.shape}, Data type: {img1.dtype}")
else:
    print("Could not load image.jpg")


img2 = cv2.imread('images/image2.jpg', cv2.IMREAD_COLOR)
if img2 is not None:
    print(f"image2.jpg loaded - Shape: {img2.shape}, Data type: {img2.dtype}")
else:
    print("Could not load image2.jpg")


img_gray = cv2.imread('images/gradient.png', cv2.IMREAD_GRAYSCALE)
if img_gray is not None:
    print(f"Grayscale gradient shape: {img_gray.shape}")
    print(f"Image data type: {img_gray.dtype}")


img_color = cv2.imread('images/gradient.png', cv2.IMREAD_COLOR)
if img_color is not None:
    print(f"Color gradient shape: {img_color.shape}")





print("\n2. Displaying Images with OpenCV")


if img1 is not None:
    cv2.imshow('Uploaded Image 1', img1)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if img2 is not None:
    cv2.imshow('Uploaded Image 2', img2)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if img_color is not None:
    cv2.imshow('Gradient Image', img_color)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()





print("\n3. Saving Images")


if img1 is not None:
    
    resized_img1 = cv2.resize(img1, (200, 150))
    cv2.imwrite('images/image_resized.jpg', resized_img1)
    print("Saved resized 'images/image_resized.jpg'")
    file_size = os.path.getsize('images/image_resized.jpg')
    print(f"File size: {file_size} bytes")


if img2 is not None:
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images/image2_gray.jpg', gray_img2)
    print("Saved grayscale 'images/image2_gray.jpg'")
    file_size = os.path.getsize('images/image2_gray.jpg')
    print(f"File size: {file_size} bytes")





print("\n4. Image Properties")


if img1 is not None:
    print(f"\nImage 1 Properties:")
    print(f"  Shape: {img1.shape}")
    print(f"  Height: {img1.shape[0]} pixels")
    print(f"  Width: {img1.shape[1]} pixels")
    if len(img1.shape) == 3:
        print(f"  Channels: {img1.shape[2]}")
    print(f"  Total pixels: {img1.size}")
    print(f"  Data type: {img1.dtype}")
    
    
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





print("\n5. Color Spaces")


if img1 is not None:
    
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    print("Converted image 1 from BGR to HSV")
    
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    
    h, w = img1.shape[:2]
    pixel_bgr = img1[h//2, w//2]
    pixel_rgb = img1_rgb[h//2, w//2]
    print(f"  Sample pixel in BGR: {pixel_bgr}")
    print(f"  Same pixel in RGB: {pixel_rgb}")
    
    
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





print("\n6. Practice Exercise - Image Blending")

"""
Exercise: Blend two uploaded images together
"""

if img1 is not None and img2 is not None:
    
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    
    
    alpha = 0.6  
    beta = 1 - alpha  
    blended = cv2.addWeighted(img1, alpha, img2_resized, beta, 0)
    
    
    cv2.imwrite('images/blended_images.jpg', blended)
    print(f"Created blended image with alpha={alpha}")
    print("Saved 'images/blended_images.jpg'")
    
    
    cv2.imshow('Blended Images', blended)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
elif img1 is not None:
    print("Image 2 not available for blending exercise")
else:
    print("Images not available for blending exercise")





print("\n7. Error Handling in OpenCV")


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