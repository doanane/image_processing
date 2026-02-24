"""
Real-World Scenario: Cleaning Noisy Binary Images
Problem: Small white specks (noise) in document scan
Solution: Erosion removes small noise
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
def erosion_for_noise_removal():
    """Use erosion to remove small noise particles"""
    
    
    clean = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(clean, 'CV', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    
    noisy = clean.copy()
    np.random.seed(42)
    for _ in range(100):
        x, y = np.random.randint(0, 100, 2)
        noisy[y, x] = 255
    
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(noisy, kernel, iterations=1)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean, cmap='gray')
    axes[0].set_title('Original Clean Text')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title('With Salt Noise')
    axes[1].axis('off')
    
    axes[2].imshow(eroded, cmap='gray')
    axes[2].set_title('After Erosion')
    axes[2].axis('off')
    
    plt.suptitle('Erosion for Noise Removal', fontweight='bold')
    plt.show()
    
    print("""
     EROSION ADVANTAGES:
    - Removes small noise particles
    - Preserves larger structures
    - Simple and fast
    
    EROSION DISADVANTAGES:
    - Shrinks the main object
    - May break thin connections
    """)

erosion_for_noise_removal()