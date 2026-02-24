import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_structuring_elements():
    """Visualize different types of structuring elements"""
    
    # Create different structuring elements
    kernel_size = 5
    
    structuring_elements = {
        'Rectangle': cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        'Cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)),
        'Ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        'Custom': np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)  # Plus sign
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, (name, kernel) in enumerate(structuring_elements.items()):
        axes[idx].imshow(kernel, cmap='gray', interpolation='nearest')
        axes[idx].set_title(f'{name}\nShape: {kernel.shape}')
        axes[idx].axis('off')
        
        # Add grid lines to show pixels
        for i in range(kernel.shape[0] + 1):
            axes[idx].axhline(i - 0.5, color='red', linewidth=0.5)
        for j in range(kernel.shape[1] + 1):
            axes[idx].axvline(j - 0.5, color='red', linewidth=0.5)
        
        # Label the 1's
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if kernel[i, j] == 1:
                    axes[idx].text(j, i, '1', ha='center', va='center', 
                                  color='blue', fontweight='bold')
    
    plt.suptitle('Structuring Elements - Your Shape Analysis Tools', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("""
    🔍 STRUCTURING ELEMENT GUIDE:
    
    • Rectangle: Probes in all directions equally
      Use for: General purpose, square/rectangular features
    
    • Cross: Probes in horizontal/vertical directions only
      Use for: Line-like features, connectivity
    
    • Ellipse: Circular/elliptical probe
      Use for: Round objects, biological cells
    
    • Custom: Design your own pattern
      Use for: Specific shape detection
    """)

visualize_structuring_elements()


def demonstrate_erosion():
    """Show how erosion works step by step"""
    
    # Create a simple shape
    img = np.zeros((9, 9), dtype=np.uint8)
    img[2:7, 2:7] = 1  # 5x5 square
    
    # 3x3 structuring element
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    # Manual erosion for understanding
    eroded_manual = np.zeros_like(img)
    
    print("🔬 EROSION - STEP BY STEP:")
    print("=" * 50)
    
    # Show the process
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            # Extract 3x3 region
            region = img[i-1:i+2, j-1:j+2]
            
            # Check if all pixels under kernel are 1
            if np.all(region == 1):
                eroded_manual[i, j] = 1
                status = "✓ KEPT"
            else:
                status = "✗ REMOVED"
            
            # Print some examples
            if (i == 2 and j == 2) or (i == 3 and j == 5) or (i == 5 and j == 5):
                print(f"\nPixel at ({i},{j}):")
                print(f"Region:\n{region}")
                print(f"Result: {status}")
    
    # OpenCV erosion
    eroded_cv = cv2.erode(img, kernel, iterations=1)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original 5x5 Square', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(eroded_manual, cmap='gray', interpolation='nearest')
    axes[1].set_title('After Erosion (Manual)', fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(eroded_cv, cmap='gray', interpolation='nearest')
    axes[2].set_title('After Erosion (OpenCV)', fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Erosion: The Shrinking Operation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("""
    📏 EROSION RESULTS:
    - Original size: 5x5 square
    - Eroded size: 3x3 square
    - Border pixels removed (1 pixel from each edge)
    - Shape preserved but smaller
    """)

demonstrate_erosion()