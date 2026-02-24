"""
File: image_restoration_demo.py
Complete guide to image restoration with real examples
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
import os

print("IMAGE RESTORATION - COMPLETE GUIDE")
print("Understanding how to recover clean images from degradation")





def create_realistic_portrait():
    """
    Create a realistic portrait image with details
    This simulates our "original clean image"
    """
    height, width = 400, 300
    
    
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    
    cv2.ellipse(image, (150, 150), (60, 80), 0, 0, 360, (180, 160, 150), -1)
    
    
    cv2.circle(image, (120, 120), 10, (50, 50, 50), -1)  
    cv2.circle(image, (180, 120), 10, (50, 50, 50), -1)  
    cv2.circle(image, (120, 120), 3, (255, 255, 255), -1)  
    cv2.circle(image, (180, 120), 3, (255, 255, 255), -1)
    
    
    cv2.line(image, (150, 140), (150, 170), (100, 80, 70), 3)
    
    
    cv2.ellipse(image, (150, 190), (30, 15), 0, 0, 180, (80, 40, 40), 3)
    
    
    for i in range(10):
        x = 100 + i * 10
        cv2.line(image, (x, 60), (x-20, 100), (30, 20, 10), 3)
    
    
    for _ in range(100):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        if 100 < x < 200 and 100 < y < 200:  
            delta = np.random.randint(-10, 10)
            pixel = image[y, x].astype(np.int16)
            image[y, x] = np.clip(pixel + delta, 0, 255).astype(np.uint8)
    
    return image


clean_image = create_realistic_portrait()
print("\nCreated clean portrait image")


plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
plt.title('Original Clean Image (Our Reference)', fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()






print("PART 2: THE DEGRADATION MODEL")
print("Understanding how images get corrupted")

def explain_degradation_model():
    """
    Mathematical model: g(x,y) = h(x,y) * f(x,y) + η(x,y)
    
    Where:
    - g(x,y) = Observed (noisy) image
    - f(x,y) = Original clean image
    - h(x,y) = Degradation function (blur, motion, etc.)
    - η(x,y) = Additive noise
    - * = Convolution operation
    """
    
    print("\n📐 MATHEMATICAL MODEL:")
    print("   g(x,y) = h(x,y) * f(x,y) + η(x,y)")
    print("\n   Where:")
    print("   • g(x,y) = The image we see (degraded)")
    print("   • f(x,y) = The original clean image (what we want)")
    print("   • h(x,y) = Degradation function (blur, camera shake)")
    print("   • η(x,y) = Noise (sensor noise, transmission errors)")
    print("   • * = Convolution (mathematical operation)")

explain_degradation_model()






print("PART 3: SIMULATING REAL-WORLD DEGRADATION")

def add_gaussian_noise(image, mean=0, sigma=25):
    """Simulate sensor noise in low light"""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_motion_blur(image, size=15, angle=45):
    """Simulate camera motion during capture"""
    
    kernel = np.zeros((size, size))
    center = size // 2
    
    
    if angle == 0:
        kernel[center, :] = 1
    elif angle == 90:
        kernel[:, center] = 1
    else:
        for i in range(size):
            j = int(center + (i - center) * np.tan(np.radians(angle)))
            if 0 <= j < size:
                kernel[i, j] = 1
    
    kernel = kernel / kernel.sum()
    
    
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def add_out_of_focus_blur(image, radius=5):
    """Simulate lens defocus"""
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    
    
    center = radius
    for i in range(kernel_size):
        for j in range(kernel_size):
            if (i - center)**2 + (j - center)**2 <= radius**2:
                kernel[i, j] = 1
    
    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def add_compression_artifacts(image, quality=15):
    """Simulate JPEG compression artifacts"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, 1)
    return decoded

def add_scratch_degradation(image):
    """Simulate physical damage to photograph"""
    degraded = image.copy()
    
    
    for _ in range(5):
        x1, y1 = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(-20, 20)
        cv2.line(degraded, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    
    for _ in range(20):
        x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        cv2.circle(degraded, (x, y), np.random.randint(1, 3), (0, 0, 0), -1)
    
    return degraded


print("\n🔧 Simulating various real-world degradations...")

degraded_images = {
    'Gaussian Noise (Sensor Noise)': add_gaussian_noise(clean_image, sigma=30),
    'Motion Blur (Camera Shake)': add_motion_blur(clean_image, size=20, angle=30),
    'Out of Focus': add_out_of_focus_blur(clean_image, radius=8),
    'JPEG Compression': add_compression_artifacts(clean_image, quality=10),
    'Physical Damage': add_scratch_degradation(clean_image)
}


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (Clean)', fontweight='bold')
axes[0].axis('off')

for i, (degradation_name, degraded) in enumerate(degraded_images.items(), 1):
    if i < 6:
        axes[i].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
        axes[i].set_title(degradation_name, fontweight='bold')
        axes[i].axis('off')

plt.suptitle('Real-World Image Degradations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()






print("PART 4: IMAGE RESTORATION TECHNIQUES")

class ImageRestorer:
    """Comprehensive image restoration class"""
    
    def __init__(self):
        self.techniques = {}
    
    def inverse_filter(self, blurred, psf, K=0.01):
        """
        Inverse Filtering
        Best for: Known blur kernel, low noise
        Formula: F(u,v) = G(u,v) / H(u,v)
        """
        
        dft = np.fft.fft2(blurred.astype(np.float32))
        dft_shift = np.fft.fftshift(dft)
        
        
        psf_padded = np.zeros_like(blurred, dtype=np.float32)
        h, w = psf.shape
        psf_padded[:h, :w] = psf
        psf_dft = np.fft.fft2(psf_padded)
        psf_dft_shift = np.fft.fftshift(psf_dft)
        
        
        psf_mag = np.abs(psf_dft_shift)
        psf_mag[psf_mag < K] = K
        
        
        restored_dft = dft_shift / psf_dft_shift
        restored = np.fft.ifft2(np.fft.ifftshift(restored_dft))
        restored = np.abs(restored)
        
        return np.clip(restored, 0, 255).astype(np.uint8)
    
    def wiener_filter(self, blurred, psf, K=0.01):
        """
        Wiener Filter (Optimal for Gaussian noise)
        Minimizes mean square error
        """
        
        dft = np.fft.fft2(blurred.astype(np.float32))
        dft_shift = np.fft.fftshift(dft)
        
        psf_padded = np.zeros_like(blurred, dtype=np.float32)
        h, w = psf.shape
        psf_padded[:h, :w] = psf
        psf_dft = np.fft.fft2(psf_padded)
        psf_dft_shift = np.fft.fftshift(psf_dft)
        
        
        psf_conj = np.conj(psf_dft_shift)
        psf_mag_sq = np.abs(psf_dft_shift)**2
        
        wiener = psf_conj / (psf_mag_sq + K)
        restored_dft = dft_shift * wiener
        
        restored = np.fft.ifft2(np.fft.ifftshift(restored_dft))
        restored = np.abs(restored)
        
        return np.clip(restored, 0, 255).astype(np.uint8)
    
    def lucy_richardson(self, blurred, psf, iterations=10):
        """
        Lucy-Richardson Deconvolution
        Iterative method, good for Poisson noise
        """
        restored = blurred.astype(np.float32)
        psf = psf.astype(np.float32)
        psf_flipped = np.flip(psf)
        
        for _ in range(iterations):
            
            estimated_blur = cv2.filter2D(restored, -1, psf)
            estimated_blur[estimated_blur == 0] = 1e-6
            
            
            ratio = blurred.astype(np.float32) / estimated_blur
            
            
            correction = cv2.filter2D(ratio, -1, psf_flipped)
            restored = restored * correction
        
        return np.clip(restored, 0, 255).astype(np.uint8)
    
    def total_variation_denoising(self, noisy, lambda_tv=0.1, iterations=50):
        """
        Total Variation Denoising (Rudin-Osher-Fatemi model)
        Preserves edges while smoothing
        """
        u = noisy.astype(np.float32)
        
        for _ in range(iterations):
            
            ux = np.roll(u, -1, axis=0) - u
            uy = np.roll(u, -1, axis=1) - u
            
            
            grad_mag = np.sqrt(ux**2 + uy**2 + 1e-6)
            
            
            div = np.roll(ux / grad_mag, 1, axis=0) + np.roll(uy / grad_mag, 1, axis=1)
            div -= ux / grad_mag + uy / grad_mag
            
            u = u - lambda_tv * div
        
        return np.clip(u, 0, 255).astype(np.uint8)


restorer = ImageRestorer()





print("\n🔄 Applying restoration techniques...")

def restore_and_compare(clean, degraded, degradation_type):
    """Apply appropriate restoration and compare results"""
    
    
    psf = np.ones((5, 5)) / 25  
    
    restored_images = {}
    
    
    if degradation_type == 'Gaussian Noise (Sensor Noise)':
        
        restored_images['Total Variation'] = restorer.total_variation_denoising(
            cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY), lambda_tv=0.1
        )
        
    elif degradation_type == 'Motion Blur (Camera Shake)':
        
        gray_degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        restored_images['Wiener Filter'] = restorer.wiener_filter(gray_degraded, psf, K=0.02)
        
    elif degradation_type == 'JPEG Compression':
        
        restored_images['Median Filter'] = cv2.medianBlur(degraded, 3)
        
    elif degradation_type == 'Physical Damage':
        
        
        mask = np.zeros(degraded.shape[:2], dtype=np.uint8)
        
        gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        mask[gray > 250] = 255  
        mask[gray < 5] = 255     
        
        restored_images['Inpainting'] = cv2.inpaint(degraded, mask, 3, cv2.INPAINT_TELEA)
    
    return restored_images


fig, axes = plt.subplots(len(degraded_images), 3, figsize=(15, 20))

for idx, (deg_name, degraded) in enumerate(degraded_images.items()):
    
    axes[idx, 0].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axes[idx, 0].set_title(f'Degraded:\n{deg_name}', fontsize=10)
    axes[idx, 0].axis('off')
    
    
    restored_dict = restore_and_compare(clean_image, degraded, deg_name)
    
    if restored_dict:
        
        best_result = list(restored_dict.values())[0]
        best_name = list(restored_dict.keys())[0]
        
        if len(best_result.shape) == 2:
            axes[idx, 1].imshow(best_result, cmap='gray')
        else:
            axes[idx, 1].imshow(cv2.cvtColor(best_result, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f'Restored:\n{best_name}', fontsize=10)
        axes[idx, 1].axis('off')
    else:
        axes[idx, 1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title('Restoration\nNot Applied', fontsize=10)
        axes[idx, 1].axis('off')
    
    
    axes[idx, 2].imshow(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
    axes[idx, 2].set_title('Original\n(Clean)', fontsize=10)
    axes[idx, 2].axis('off')

plt.suptitle('Image Restoration Results Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()






print("PART 6: QUALITY ASSESSMENT METRICS")

class QualityMetrics:
    """Measure restoration quality"""
    
    @staticmethod
    def psnr(original, restored):
        """Peak Signal-to-Noise Ratio"""
        if original.shape != restored.shape:
            restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
        
        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def ssim(original, restored):
        """Structural Similarity Index (simplified)"""
        if original.shape != restored.shape:
            restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
        
        
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if len(restored.shape) == 3:
            restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        
        mu_x = np.mean(original)
        mu_y = np.mean(restored)
        
        
        sigma_x = np.var(original)
        sigma_y = np.var(restored)
        sigma_xy = np.mean((original - mu_x) * (restored - mu_y))
        
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim
    
    @staticmethod
    def mse(original, restored):
        """Mean Square Error"""
        if original.shape != restored.shape:
            restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
        
        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        return mse


metrics = QualityMetrics()

print("\n📊 Quality Assessment Results:")
print("-" * 50)
print(f"{'Degradation Type':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<10}")
print("-" * 50)

for deg_name, degraded in degraded_images.items():
    
    if len(clean_image.shape) == 3:
        clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    else:
        clean_gray = clean_image
    
    if len(degraded.shape) == 3:
        degraded_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    else:
        degraded_gray = degraded
    
    psnr_val = metrics.psnr(clean_gray, degraded_gray)
    ssim_val = metrics.ssim(clean_gray, degraded_gray)
    mse_val = metrics.mse(clean_gray, degraded_gray)
    
    print(f"{deg_name:<30} {psnr_val:<12.2f} {ssim_val:<10.3f} {mse_val:<10.2f}")






print("PART 7: REAL-WORLD SCENARIO - RESTORING AN OLD PHOTO")

def restore_old_photograph(damaged_photo):
    """
    Complete pipeline for restoring an old, damaged photograph
    """
    print("\n📸 Starting photo restoration pipeline...")
    
    
    if len(damaged_photo.shape) == 3:
        gray = cv2.cvtColor(damaged_photo, cv2.COLOR_BGR2GRAY)
    else:
        gray = damaged_photo.copy()
    
    print("  Step 1: Converted to grayscale")
    
    
    step2 = cv2.medianBlur(gray, 3)
    print("  Step 2: Removed scratches and dust")
    
    
    step3 = cv2.GaussianBlur(step2, (3, 3), 0)
    print("  Step 3: Reduced grain noise")
    
    
    step4 = cv2.equalizeHist(step3)
    print("  Step 4: Enhanced contrast")
    
    
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    step5 = cv2.filter2D(step4, -1, kernel)
    print("  Step 5: Sharpened details")
    
    
    step6 = cv2.convertScaleAbs(step5, alpha=1.1, beta=5)
    print("  Step 6: Final tone adjustment")
    
    return step6


damaged_photo = clean_image.copy()
damaged_photo = add_scratch_degradation(damaged_photo)
damaged_photo = add_gaussian_noise(damaged_photo, sigma=20)


restored_photo = restore_old_photograph(damaged_photo)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (Clean)', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(damaged_photo, cv2.COLOR_BGR2RGB))
axes[1].set_title('Damaged Photo', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(restored_photo, cmap='gray')
axes[2].set_title('Restored Photo', fontweight='bold')
axes[2].axis('off')

plt.suptitle('Old Photograph Restoration Example', fontsize=14, fontweight='bold')
plt.show()


print("\n📈 Restoration Improvement:")
psnr_before = metrics.psnr(
    cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(damaged_photo, cv2.COLOR_BGR2GRAY)
)
psnr_after = metrics.psnr(
    cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY),
    restored_photo
)
print(f"  PSNR before restoration: {psnr_before:.2f} dB")
print(f"  PSNR after restoration:  {psnr_after:.2f} dB")
print(f"  Improvement: {psnr_after - psnr_before:.2f} dB")





def interactive_restoration_demo():
    """
    Interactive demonstration showing progressive restoration
    """
    
    print("PART 8: INTERACTIVE RESTORATION DEMO")
    print("Watch as the image gets progressively restored")
        
    
    test_img = clean_image.copy()
    test_img = add_gaussian_noise(test_img, sigma=25)
    test_img = add_motion_blur(test_img, size=15, angle=30)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    
    axes[0, 0].imshow(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (Goal)', fontweight='bold')
    axes[0, 0].axis('off')
    
    
    axes[0, 1].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Degraded Image\n(Input)', fontweight='bold')
    axes[0, 1].axis('off')
    
    
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    psf = np.ones((5, 5)) / 25
    step1 = restorer.wiener_filter(gray_test, psf, K=0.01)
    axes[0, 2].imshow(step1, cmap='gray')
    axes[0, 2].set_title('Step 1: Deblurring', fontweight='bold')
    axes[0, 2].axis('off')
    
    
    step2 = restorer.total_variation_denoising(step1, lambda_tv=0.1)
    axes[1, 0].imshow(step2, cmap='gray')
    axes[1, 0].set_title('Step 2: Denoising', fontweight='bold')
    axes[1, 0].axis('off')
    
    
    step3 = cv2.equalizeHist(step2)
    axes[1, 1].imshow(step3, cmap='gray')
    axes[1, 1].set_title('Step 3: Contrast\nEnhancement', fontweight='bold')
    axes[1, 1].axis('off')
    
    
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    step4 = cv2.filter2D(step3, -1, kernel)
    axes[1, 2].imshow(step4, cmap='gray')
    axes[1, 2].set_title('Step 4: Final\nSharpening', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Progressive Image Restoration Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
    clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    
    print("\n📊 Quality at each step:")
    steps = [
        ('Degraded', gray_test),
        ('After Deblurring', step1),
        ('After Denoising', step2),
        ('After Contrast', step3),
        ('Final', step4)
    ]
    
    for step_name, step_img in steps:
        psnr_val = metrics.psnr(clean_gray, step_img)
        print(f"  {step_name:<20} PSNR: {psnr_val:.2f} dB")


interactive_restoration_demo()






print("KEY CONCEPTS GLOSSARY")

glossary = {
    "Image Restoration": "Process of recovering original image from degraded version",
    
    "Degradation Model": "Mathematical description of how images get corrupted",
    
    "Point Spread Function (PSF)": "Mathematical description of how a single point spreads in an imaging system",
    
    "Convolution": "Mathematical operation that describes how blur affects an image",
    
    "Inverse Filtering": "Direct inversion of the degradation process in frequency domain",
    
    "Wiener Filter": "Optimal filter that minimizes mean square error, accounts for noise",
    
    "Lucy-Richardson": "Iterative algorithm good for astronomical and medical images",
    
    "Total Variation": "Denoising method that preserves edges while smoothing",
    
    "PSNR": "Peak Signal-to-Noise Ratio - measures pixel-level accuracy",
    
    "SSIM": "Structural Similarity Index - measures perceptual quality"
}

for term, definition in glossary.items():
    print(f"• {term}: {definition}")






print("PRACTICAL DECISION GUIDE")
print("Choose the right restoration method")

decision_guide = {
    "Noise Type": {
        "Gaussian Noise": "Use Wiener Filter or Total Variation",
        "Salt & Pepper": "Use Median Filter",
        "Speckle Noise": "Use Bilateral Filter"
    },
    "Blur Type": {
        "Motion Blur": "Use Wiener Filter with estimated PSF",
        "Out of Focus": "Use Lucy-Richardson Deconvolution",
        "Unknown Blur": "Try blind deconvolution"
    },
    "Damage Type": {
        "Scratches": "Use Inpainting",
        "Fading": "Use Histogram Equalization",
        "JPEG Artifacts": "Use Median Filter + Denoising"
    }
}

for category, methods in decision_guide.items():
    print(f"\n{category}:")
    for condition, method in methods.items():
        print(f"   • {condition}: {method}")


print("RESTORATION GUIDE COMPLETE")
print("You now understand:")
print("• What image restoration is")
print("• Different types of degradation")
print("• Various restoration techniques")
print("• How to choose the right method")
print("• How to evaluate restoration quality")


cv2.imwrite('images/original_portrait.png', clean_image)
cv2.imwrite('images/restored_portrait.png', restored_photo)
print("\n Sample images saved to 'images/' folder")


def restoration_checklist(image, problem_description):
    """
    Decision tree for choosing restoration method
    """
    
    
    if "blurry" in problem_description.lower():
        if "motion" in problem_description.lower():
            return "Wiener Filter with motion PSF"
        else:
            return "Lucy-Richardson Deconvolution"
    
    elif "noisy" in problem_description.lower():
        
        unique_values = len(np.unique(image))
        if unique_values < 50:  
            return "Median Filter"
        else:
            return "Total Variation Denoising"
    
    elif "scratched" in problem_description.lower():
        return "Inpainting"
    
    elif "old" in problem_description.lower() or "faded" in problem_description.lower():
        return "Histogram Equalization + Sharpening"
    
    else:
        return "Try combination of methods"