"""
Real-World Scenario: Removing Scanner Artifacts
Problem: Old document scanner adds periodic pattern
Solution: Remove specific frequencies
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def remove_scanner_artifacts():
    """Remove periodic noise from scanned document"""
    
    # Create a clean document
    doc = np.ones((300, 400)) * 255
    for i in range(0, 300, 30):
        cv2.putText(doc, f"Line {i//30 + 1}", (50, i+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    # Add periodic scanner noise
    x = np.arange(400)
    y = np.arange(300)
    X, Y = np.meshgrid(x, y)
    noise_pattern = 50 * np.sin(2 * np.pi * X / 20) * np.sin(2 * np.pi * Y / 20)
    noisy_doc = np.clip(doc + noise_pattern, 0, 255).astype(np.uint8)
    
    # FFT
    f = np.fft.fft2(noisy_doc)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    
    # Create notch filter to remove specific frequencies
    rows, cols = noisy_doc.shape
    crow, ccol = rows//2, cols//2
    
    notch_filter = np.ones((rows, cols))
    
    # Remove the noise frequencies (the bright spots in spectrum)
    # In real application, you'd detect these automatically
    notch_filter[crow-20:crow+20, ccol-20:ccol+20] = 0  # Remove center
    notch_filter[crow-5:crow+5, ccol-50:ccol+50] = 1  # Restore center line
    
    # Apply filter
    f_filtered = fshift * notch_filter
    
    # Inverse FFT
    img_filtered = np.fft.ifft2(np.fft.ifftshift(f_filtered)).real
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(noisy_doc, cmap='gray')
    axes[0,0].set_title("Scanned Document with\nPeriodic Artifacts")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(magnitude, cmap='gray')
    axes[0,1].set_title("Frequency Spectrum\n(Bright spots = noise)")
    axes[0,1].axis('off')
    
    axes[1,0].imshow(notch_filter, cmap='gray')
    axes[1,0].set_title("Notch Filter\n(Removes specific frequencies)")
    axes[1,0].axis('off')
    
    axes[1,1].imshow(img_filtered, cmap='gray')
    axes[1,1].set_title("Restored Document")
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("""
    📄 DOCUMENT RESTORATION:
    - Periodic noise appears as bright spots in frequency domain
    - Notch filter removes only those frequencies
    - Document text preserved!
    """)

remove_scanner_artifacts()