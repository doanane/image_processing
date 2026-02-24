"""
File: medical_image_denoising.py
Real-World Scenario: A hospital needs to clean up noisy MRI/X-ray images
for better diagnosis. Different noise types require different filters.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


print("MEDICAL IMAGE DENOISING SYSTEM")
print("Real-World Application: Hospital Radiology Department")






def create_medical_sample():
    """Create a simulated medical image with anatomical features"""
    
    
    height, width = 300, 400
    medical_img = np.ones((height, width), dtype=np.uint8) * 200  
    
    
    cv2.rectangle(medical_img, (100, 50), (150, 200), 100, -1)  
    cv2.rectangle(medical_img, (250, 80), (300, 220), 90, -1)   
    
    
    cv2.circle(medical_img, (200, 150), 40, 150, -1)  
    
    
    cv2.circle(medical_img, (280, 120), 10, 50, -1)  
    cv2.circle(medical_img, (120, 180), 8, 220, -1)  
    
    
    for i in range(5):
        x = 180 + i * 20
        cv2.line(medical_img, (x, 140), (x+10, 160), 120, 1)  
    
    return medical_img


clean_medical = create_medical_sample()
print("\nCreated clean medical image (simulated X-ray)")





def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Simulate: Poor lighting conditions, sensor noise
    Real scenario: Night-time X-ray, old X-ray machine
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.05):
    """
    Simulate: Transmission errors, dust on sensor
    Real scenario: Fax machine transmission, dirty scanner
    """
    noisy = image.copy()
    
    salt_mask = np.random.random(image.shape) < prob/2
    noisy[salt_mask] = 255
    
    
    pepper_mask = np.random.random(image.shape) < prob/2
    noisy[pepper_mask] = 0
    
    return noisy

def add_speckle_noise(image, variance=0.04):
    """
    Simulate: Ultrasound imaging, radar interference
    Real scenario: Prenatal ultrasound, satellite radar
    """
    noise = np.random.randn(*image.shape) * np.sqrt(variance)
    noisy = image + image * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


print("\n📊 Simulating different noise types...")

noisy_gaussian = add_gaussian_noise(clean_medical, sigma=30)
print("  • Gaussian Noise: Simulates low-light X-ray conditions")

noisy_saltpepper = add_salt_pepper_noise(clean_medical, prob=0.1)
print("  • Salt & Pepper: Simulates transmission errors")

noisy_speckle = add_speckle_noise(clean_medical, variance=0.05)
print("  • Speckle Noise: Simulates ultrasound interference")





print("\n🔧 Applying restoration techniques...")

def restore_medical_images(noisy_image, noise_type):
    """
    Apply appropriate restoration based on noise type
    Real scenario: Radiologist selects different filters for different conditions
    """
    
    results = {}
    
    
    results['Gaussian Blur'] = cv2.GaussianBlur(noisy_image, (5, 5), 1.5)
    
    
    results['Median Filter'] = cv2.medianBlur(noisy_image, 5)
    
    
    results['Bilateral Filter'] = cv2.bilateralFilter(noisy_image, 9, 75, 75)
    
    
    results['NLM Denoising'] = cv2.fastNlMeansDenoising(noisy_image, None, 30, 7, 21)
    
    return results


restored_gaussian = restore_medical_images(noisy_gaussian, 'gaussian')
restored_sp = restore_medical_images(noisy_saltpepper, 'saltpepper')
restored_speckle = restore_medical_images(noisy_speckle, 'speckle')

print("All restoration techniques applied")





def calculate_psnr(original, restored):
    """
    Peak Signal-to-Noise Ratio
    Higher value = better restoration (typical: 30-50 dB is good)
    Real-world: Used to compare denoising algorithms
    """
    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, restored):
    """
    Structural Similarity Index
    Range: -1 to 1, closer to 1 = more similar
    Real-world: Measures perceived quality
    """
    
    
    mu_x = np.mean(original)
    mu_y = np.mean(restored)
    sigma_x = np.std(original)
    sigma_y = np.std(restored)
    sigma_xy = np.mean((original - mu_x) * (restored - mu_y))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    
    return ssim





def create_diagnostic_dashboard():
    """Create a comprehensive visualization for medical staff"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Medical Image Denoising System - Clinical Dashboard', 
                 fontsize=16, fontweight='bold')
    
    
    plt.subplot(4, 4, 1)
    plt.imshow(clean_medical, cmap='gray')
    plt.title('Original (Clean X-ray)')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(noisy_gaussian, cmap='gray')
    plt.title('Gaussian Noise\n(Low-light X-ray)')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(noisy_saltpepper, cmap='gray')
    plt.title('Salt & Pepper\n(Fax Transmission)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(noisy_speckle, cmap='gray')
    plt.title('Speckle Noise\n(Ultrasound)')
    plt.axis('off')
    
    
    restoration_methods = ['Gaussian Blur', 'Median Filter', 'Bilateral Filter', 'NLM Denoising']
    noise_types = ['Gaussian', 'Salt & Pepper', 'Speckle']
    restored_images = [restored_gaussian, restored_sp, restored_speckle]
    
    for row, (noise_name, restored_dict) in enumerate(zip(noise_types, restored_images)):
        for col, method in enumerate(restoration_methods):
            idx = (row + 1) * 4 + col + 1
            if idx <= 16:
                plt.subplot(4, 4, idx)
                
                
                img = restored_dict[method]
                plt.imshow(img, cmap='gray')
                
                
                if noise_name == 'Gaussian':
                    original = clean_medical
                elif noise_name == 'Salt & Pepper':
                    original = clean_medical
                else:
                    original = clean_medical
                
                psnr = calculate_psnr(original, img)
                ssim = calculate_ssim(original, img)
                
                plt.title(f'{noise_name}\n{method}\nPSNR: {psnr:.1f}dB | SSIM: {ssim:.2f}')
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()


print("\n📊 Generating clinical dashboard...")
create_diagnostic_dashboard()





def recommend_filter(noise_type, clinical_scenario):
    """
    Real-world function: Doctor/technician selects best filter
    based on noise type and clinical need
    """
    
    recommendations = {
        'gaussian': {
            'best': 'Gaussian Blur',
            'reason': 'Mathematically optimal for Gaussian-distributed noise',
            'clinical': 'Use for standard X-ray enhancement'
        },
        'salt_pepper': {
            'best': 'Median Filter',
            'reason': 'Excellent at removing isolated outliers without blurring',
            'clinical': 'Use for faxed medical documents, old scanned films'
        },
        'speckle': {
            'best': 'Bilateral Filter',
            'reason': 'Preserves edges while reducing multiplicative noise',
            'clinical': 'Use for ultrasound enhancement, edge preservation critical'
        }
    }
    
    print(f"\n🔍 Clinical Recommendation for {clinical_scenario}:")
    print(f"   Noise Type: {noise_type}")
    print(f"   Recommended Filter: {recommendations[noise_type]['best']}")
    print(f"   Why: {recommendations[noise_type]['reason']}")
    print(f"   Clinical Application: {recommendations[noise_type]['clinical']}")
    
    return recommendations[noise_type]['best']



print("CLINICAL DECISION SUPPORT SYSTEM")


scenarios = [
    ('gaussian', 'Night-time chest X-ray'),
    ('salt_pepper', 'Old scanned MRI film from 1980s'),
    ('speckle', 'Prenatal ultrasound examination')
]

for noise_type, scenario in scenarios:
    recommend_filter(noise_type, scenario)





def noise_simulator_interactive():
    """
    Interactive tool to understand how different noise affects diagnosis
    """
    
    
    print("INTERACTIVE NOISE SIMULATOR")
    print("Adjust noise levels to see effect on diagnostic quality")
    
    
    
    test_img = np.zeros((150, 150), dtype=np.uint8)
    
    
    cv2.circle(test_img, (75, 75), 10, 200, -1)  
    cv2.circle(test_img, (75, 75), 5, 50, -1)    
    
    
    for angle in range(0, 360, 45):
        x = 75 + int(20 * np.cos(np.radians(angle)))
        y = 75 + int(20 * np.sin(np.radians(angle)))
        cv2.line(test_img, (75, 75), (x, y), 150, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Effect of Noise on Diagnostic Features', fontsize=14, fontweight='bold')
    
    
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title('Original\n(Tumor clearly visible)')
    axes[0, 0].axis('off')
    
    
    noise_levels = [10, 20, 30, 40, 50, 60, 70]
    for i, sigma in enumerate(noise_levels[:3]):
        noisy = add_gaussian_noise(test_img, sigma=sigma)
        axes[0, i+1].imshow(noisy, cmap='gray')
        axes[0, i+1].set_title(f'Gaussian σ={sigma}\n{"Visible" if sigma<30 else "Obscured"}')
        axes[0, i+1].axis('off')
    
    
    sp_levels = [0.02, 0.05, 0.1, 0.2]
    for i, prob in enumerate(sp_levels[:4]):
        noisy = add_salt_pepper_noise(test_img, prob=prob)
        axes[1, i].imshow(noisy, cmap='gray')
        axes[1, i].set_title(f'Salt & Pepper {prob*100:.0f}%\n{"Visible" if prob<0.1 else "Obscured"}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📝 Clinical Observation:")
    print("   • At σ=30, tumor boundaries become unclear")
    print("   • At 10% salt & pepper, small features (5mm) are lost")
    print("   • Diagnostic confidence decreases with noise level")


noise_simulator_interactive()






print("KEY TERMS GLOSSARY")


glossary = {
    "Noise": "Random variations in pixel values that degrade image quality",
    "Signal-to-Noise Ratio (SNR)": "Measure of signal strength relative to noise",
    "Peak Signal-to-Noise Ratio (PSNR)": "Quality metric (higher is better)",
    "Structural Similarity (SSIM)": "Perceptual quality metric (1 = identical)",
    "Kernel/Filter": "Small matrix used for convolution operations",
    "Convolution": "Sliding window operation for filtering",
    "Gaussian Noise": "Noise following normal distribution",
    "Salt & Pepper": "Random white and black pixels",
    "Speckle Noise": "Multiplicative noise common in coherent imaging",
    "Median Filter": "Replaces pixel with median of neighbors",
    "Bilateral Filter": "Edge-preserving smoothing filter",
    "NLM Denoising": "Non-local means, uses similar patches"
}

for term, definition in glossary.items():
    print(f"• {term}: {definition}")






print("PRACTICAL TIPS FOR REAL APPLICATIONS")


tips = [
    ("Medical Imaging", 
     "Always preserve edges - use bilateral filter for ultrasound"),
    
    ("Security Cameras", 
     "Median filter works best for salt & pepper from transmission errors"),
    
    ("Smartphone Photos", 
     "Gaussian blur + sharpening for low-light photos"),
    
    ("Satellite Imagery", 
     "Speckle noise requires specialized filters (Lee, Frost)"),
    
    ("Document Scanning", 
     "Adaptive thresholding after median filter"),
    
    ("Old Photo Restoration",
     "Combination of median (for dust) and Gaussian (for grain)")
]

for application, tip in tips:
    print(f"{application}: {tip}")


print("SIMULATION COMPLETE")
print("You now understand how different noise types affect")
print("medical images and how to choose appropriate restoration")
