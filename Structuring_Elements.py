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