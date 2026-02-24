"""
Real-World Scenario: Fabric Inspection
Problem: Detect weaving defects in fabric
Solution: Frequency domain reveals periodic patterns
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_fabric_sample():
    """Create a simulated fabric image"""
    fabric = np.ones((300, 300)) * 128
    
    
    for i in range(0, 300, 20):
        fabric[i:i+2, :] = 200  
        fabric[:, i:i+2] = 200  
    
    
    fabric[150:152, 100:200] = 50  
    
    return fabric

fabric = create_fabric_sample()


f_transform = np.fft.fft2(fabric)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)


plt.figure(figsize=(15, 5))

plt.subplot(1,3,1)
plt.imshow(fabric, cmap='gray')
plt.title("Fabric with Defect")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Frequency Spectrum")
plt.axis('off')
plt.colorbar()

plt.subplot(1,3,3)

center_y = magnitude_spectrum.shape[0] // 2
plt.plot(magnitude_spectrum[center_y, :])
plt.title("Frequency Profile")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()

print("""
WHAT WE SEE:
- The regular weaving pattern creates strong peaks in frequency domain
- The defect appears as additional frequencies
- This makes defects easy to detect automatically!
""")