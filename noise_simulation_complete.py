"""
File: noise_simulation_complete.py
Complete guide to simulating and understanding different noise types
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


print("COMPLETE GUIDE TO NOISE SIMULATION & FILTERING")
print("Understanding every type of noise and how to remove it")






def create_test_pattern():
    """
    Create a comprehensive test image with:
    - Smooth regions (for noise visibility)
    - Edges (for edge preservation testing)
    - Text (for detail preservation)
    - Gradients (for continuous tone testing)
    """
    height, width = 400, 400
    img = np.ones((height, width), dtype=np.uint8) * 128  
    
    
    for i in range(width):
        img[:, i] = int(128 + 50 * np.sin(i / 50))
    
    
    img[50:150, 50:150] = 200  
    img[200:300, 200:300] = 50   
    
    
    cv2.circle(img, (300, 100), 40, 200, -1)
    cv2.circle(img, (100, 300), 40, 50, -1)
    
    
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, 'NOISE', (120, 250), font, 1, 200, 2)
    cv2.putText(img, 'TEST', (220, 350), font, 1, 50, 2)
    
    
    for angle in range(0, 180, 30):
        x = 300 + int(50 * np.cos(np.radians(angle)))
        y = 300 + int(50 * np.sin(np.radians(angle)))
        cv2.line(img, (300, 300), (x, y), 150, 2)
    
    return img


test_image = create_test_pattern()
print("\n✅ Created comprehensive test pattern")


plt.figure(figsize=(10, 8))
plt.imshow(test_image, cmap='gray')
plt.title('Test Pattern for Noise Analysis', fontsize=14, fontweight='bold')
plt.colorbar(label='Pixel Intensity')
plt.axis('off')
plt.show()





class NoiseSimulator:
    """
    Complete library for simulating all types of image noise
    Each method includes real-world scenario explanation
    """
    
    def __init__(self):
        self.noise_types = {}
        
    def gaussian_noise(self, image, mean=0, sigma=25):
        """
        GAUSSIAN NOISE (Additive White Gaussian Noise - AWGN)
        
        Real-world sources:
        - Thermal noise in camera sensors
        - Low-light conditions
        - Electronic noise in circuits
        
        Mathematical model: n ~ N(μ, σ²)
        Characteristics: Affects all pixels, follows bell curve
        """
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8), noise
    
    def salt_pepper_noise(self, image, salt_prob=0.05, pepper_prob=0.05):
        """
        SALT & PEPPER NOISE (Impulse Noise)
        
        Real-world sources:
        - Faulty memory hardware
        - Transmission errors
        - Dust on sensor
        - Old film scratches
        
        Mathematical model: Random pixels set to min/max
        Characteristics: Random white and black pixels
        """
        noisy = image.copy()
        total_pixels = image.size
        
        
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy[salt_mask] = 255
        
        
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy[pepper_mask] = 0
        
        noise_map = np.zeros_like(image)
        noise_map[salt_mask] = 255
        noise_map[pepper_mask] = 128
        
        return noisy, noise_map
    
    def speckle_noise(self, image, variance=0.04):
        """
        SPECKLE NOISE (Multiplicative Noise)
        
        Real-world sources:
        - Ultrasound imaging
        - SAR radar imagery
        - Laser imaging
        - Coherent light systems
        
        Mathematical model: I_noisy = I + I * n, where n ~ N(0, σ²)
        Characteristics: Noise magnitude depends on signal strength
        """
        noise = np.random.randn(*image.shape) * np.sqrt(variance)
        noisy = image.astype(np.float32) + image.astype(np.float32) * noise
        return np.clip(noisy, 0, 255).astype(np.uint8), noise
    
    def poisson_noise(self, image, peak=100):
        """
        POISSON NOISE (Shot Noise)
        
        Real-world sources:
        - Photon counting in astronomy
        - Medical imaging (nuclear medicine)
        - Low-light photography
        - Quantum effects
        
        Mathematical model: Follows Poisson distribution
        Characteristics: Signal-dependent, appears in photon-limited imaging
        """
        
        scaled = image.astype(np.float32) / 255 * peak
        noisy = np.random.poisson(scaled).astype(np.float32)
        noisy = noisy / peak * 255
        noise = noisy - image
        
        return np.clip(noisy, 0, 255).astype(np.uint8), noise
    
    def periodic_noise(self, image, frequency=0.1, amplitude=50):
        """
        PERIODIC NOISE
        
        Real-world sources:
        - Electrical interference (50/60 Hz hum)
        - Camera sensor pattern noise
        - Scanner artifacts
        - RF interference
        
        Mathematical model: n = A * sin(2πfx + φ)
        Characteristics: Repeating pattern in frequency domain
        """
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        X, Y = np.meshgrid(x, y)
        
        
        noise_pattern = amplitude * np.sin(2 * np.pi * frequency * X)
        noise_pattern += amplitude * np.cos(2 * np.pi * frequency * Y)
        
        noisy = image.astype(np.float32) + noise_pattern
        return np.clip(noisy, 0, 255).astype(np.uint8), noise_pattern
    
    def uniform_noise(self, image, low=-50, high=50):
        """
        UNIFORM NOISE
        
        Real-world sources:
        - Quantization errors
        - A/D converter noise
        - Dithering artifacts
        
        Mathematical model: n ~ U(a, b)
        Characteristics: Equal probability across range
        """
        noise = np.random.uniform(low, high, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8), noise
    
    def rayleigh_noise(self, image, scale=30):
        """
        RAYLEIGH NOISE
        
        Real-world sources:
        - Range imaging (LIDAR)
        - Underwater acoustics
        - Fading channels
        
        Mathematical model: f(x) = (x/σ²)exp(-x²/2σ²)
        Characteristics: Skewed distribution, always positive
        """
        noise = np.random.rayleigh(scale, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8), noise
    
    def gamma_noise(self, image, shape=2, scale=15):
        """
        GAMMA NOISE
        
        Real-world sources:
        - Laser speckle
        - SAR imagery
        - Medical ultrasound
        
        Mathematical model: f(x) = x^(k-1)exp(-x/θ)/(θ^kΓ(k))
        Characteristics: Positively skewed, multiplicative-like
        """
        noise = np.random.gamma(shape, scale, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8), noise
    
    def mixed_noise(self, image, noise_types=['gaussian', 'salt_pepper']):
        """
        MIXED NOISE
        
        Real-world sources:
        - Real photographs (multiple noise sources)
        - Compressed and transmitted images
        - Old photographs with multiple defects
        
        Characteristics: Combination of different noise types
        """
        noisy = image.copy()
        noise_components = {}
        
        if 'gaussian' in noise_types:
            noisy, n_g = self.gaussian_noise(noisy, sigma=15)
            noise_components['gaussian'] = n_g
        
        if 'salt_pepper' in noise_types:
            noisy, n_sp = self.salt_pepper_noise(noisy, salt_prob=0.02, pepper_prob=0.02)
            noise_components['salt_pepper'] = n_sp
        
        if 'speckle' in noise_types:
            noisy, n_s = self.speckle_noise(noisy, variance=0.02)
            noise_components['speckle'] = n_s
        
        return noisy, noise_components


simulator = NoiseSimulator()






print("PART 3: VISUALIZING ALL NOISE TYPES")
print("Understanding how different noises affect images")



noise_methods = {
    'Gaussian Noise': simulator.gaussian_noise,
    'Salt & Pepper': simulator.salt_pepper_noise,
    'Speckle Noise': simulator.speckle_noise,
    'Poisson Noise': simulator.poisson_noise,
    'Periodic Noise': simulator.periodic_noise,
    'Uniform Noise': simulator.uniform_noise,
    'Rayleigh Noise': simulator.rayleigh_noise,
    'Gamma Noise': simulator.gamma_noise
}


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()


axes[0].imshow(test_image, cmap='gray')
axes[0].set_title('Original (Clean)', fontweight='bold')
axes[0].axis('off')


for idx, (noise_name, noise_func) in enumerate(noise_methods.items(), 1):
    if idx < 9:
        noisy_img, _ = noise_func(test_image)
        axes[idx].imshow(noisy_img, cmap='gray')
        axes[idx].set_title(noise_name, fontweight='bold')
        axes[idx].axis('off')

plt.suptitle('All Major Types of Image Noise', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()





def analyze_noise_distribution(clean_img, noisy_img, noise_name):
    """Analyze statistical properties of noise"""
    
    
    if len(clean_img.shape) == 2:
        noise = noisy_img.astype(float) - clean_img.astype(float)
    else:
        noise = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY).astype(float) - \
                cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY).astype(float)
    
    
    mean = np.mean(noise)
    std = np.std(noise)
    skewness = np.mean(((noise - mean)/std)**3)
    kurtosis = np.mean(((noise - mean)/std)**4) - 3
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': noise.min(),
        'max': noise.max()
    }

print("\n📊 Noise Distribution Analysis:")
print("-" * 60)
print(f"{'Noise Type':<20} {'Mean':<10} {'Std Dev':<10} {'Skewness':<10} {'Kurtosis':<10}")
print("-" * 60)

for noise_name, noise_func in noise_methods.items():
    noisy_img, _ = noise_func(test_image)
    stats = analyze_noise_distribution(test_image, noisy_img, noise_name)
    print(f"{noise_name:<20} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
          f"{stats['skewness']:<10.2f} {stats['kurtosis']:<10.2f}")






print("PART 5: COMPLETE FILTERING GUIDE")
print("Every filter explained with code and application")


class FilterLibrary:
    """
    Complete library of image filters
    Each filter includes mathematical explanation and best use case
    """
    
    @staticmethod
    def mean_filter(image, kernel_size=3):
        """
        MEAN FILTER (Averaging Filter)
        
        Formula: g(x,y) = (1/mn) * Σ f(x,y) over neighborhood
        
        Kernel: [1/9 1/9 1/9]
                [1/9 1/9 1/9]
                [1/9 1/9 1/9]
        
        Best for: Gaussian noise reduction
        Trade-off: Blurs edges, loses detail
        """
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def median_filter(image, kernel_size=3):
        """
        MEDIAN FILTER
        
        Formula: g(x,y) = median{ f(s,t) } over neighborhood
        
        Best for: Salt & pepper noise, impulse noise
        Trade-off: Good edge preservation, slower than mean
        """
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def gaussian_filter(image, kernel_size=3, sigma=1.0):
        """
        GAUSSIAN FILTER
        
        Formula: G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
        
        Kernel: Weighted average, center pixel has highest weight
        
        Best for: Gaussian noise, natural-looking blur
        Trade-off: Better than mean, still blurs edges
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """
        BILATERAL FILTER
        
        Formula: Combines domain and range filtering
        w = exp(-|x-y|²/2σ_d²) * exp(-|I(x)-I(y)|²/2σ_r²)
        
        Best for: Edge-preserving smoothing, speckle noise
        Trade-off: Slower, but preserves edges
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def wiener_filter(image, kernel_size=5, noise_var=None):
        """
        WIENER FILTER (Adaptive)
        
        Formula: F(u,v) = [H*(u,v)/|H(u,v)|²] * [|H(u,v)|²/(|H(u,v)|² + K)]
        
        Best for: Known blur + Gaussian noise
        Trade-off: Requires noise estimation, optimal in MSE sense
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        
        if noise_var is None:
            noise_var = np.var(gray[::10, ::10])  
        
        
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        mean = cv2.filter2D(gray.astype(float), -1, kernel)
        variance = cv2.filter2D(gray.astype(float)**2, -1, kernel) - mean**2
        
        
        result = mean + (variance / (variance + noise_var)) * (gray - mean)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def nlm_filter(image, h=10, template_size=7, search_size=21):
        """
        NON-LOCAL MEANS FILTER
        
        Formula: NL[I](i) = Σ w(i,j) I(j)
        w(i,j) = (1/Z(i)) * exp(-||I(Ni)-I(Nj)||²/h²)
        
        Best for: High-quality denoising, texture preservation
        Trade-off: Very slow, but excellent results
        """
        return cv2.fastNlMeansDenoising(image, None, h, template_size, search_size)
    
    @staticmethod
    def anisotropic_diffusion(image, iterations=10, kappa=50, gamma=0.1):
        """
        ANISOTROPIC DIFFUSION (Perona-Malik)
        
        Formula: ∂I/∂t = div(c(|∇I|) ∇I)
        c(x) = 1 / (1 + (x/kappa)²)
        
        Best for: Edge-preserving smoothing over multiple iterations
        Trade-off: Iterative, parameter sensitive
        """
        img = image.astype(np.float32)
        
        for _ in range(iterations):
            
            dx = np.roll(img, -1, axis=1) - img
            dy = np.roll(img, -1, axis=0) - img
            
            
            c_dx = 1 / (1 + (dx/kappa)**2)
            c_dy = 1 / (1 + (dy/kappa)**2)
            
            
            img += gamma * (c_dx * dx + c_dy * dy)
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def guided_filter(image, guide=None, radius=5, eps=0.01):
        """
        GUIDED FILTER
        
        Formula: q = a * I + b, where a and b are derived from guidance
        
        Best for: Edge-preserving smoothing with guidance image
        Trade-off: Can use another image as guide
        """
        if guide is None:
            guide = image
        
        
        mean_I = cv2.boxFilter(guide.astype(float), -1, (radius, radius))
        mean_p = cv2.boxFilter(image.astype(float), -1, (radius, radius))
        corr_I = cv2.boxFilter(guide.astype(float)**2, -1, (radius, radius))
        corr_Ip = cv2.boxFilter(guide.astype(float) * image.astype(float), -1, (radius, radius))
        
        var_I = corr_I - mean_I**2
        cov_Ip = corr_Ip - mean_I * mean_p
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        result = mean_a * guide + mean_b
        
        return np.clip(result, 0, 255).astype(np.uint8)


filters = FilterLibrary()





print("\n🔧 Testing filters on different noise types...")

def test_filters_on_noise(noisy_image, noise_type):
    """Apply all filters and compare results"""
    
    results = {}
    
    
    results['Mean Filter'] = filters.mean_filter(noisy_image, kernel_size=3)
    results['Median Filter'] = filters.median_filter(noisy_image, kernel_size=3)
    results['Gaussian Filter'] = filters.gaussian_filter(noisy_image, kernel_size=3, sigma=1)
    results['Bilateral Filter'] = filters.bilateral_filter(noisy_image)
    results['Wiener Filter'] = filters.wiener_filter(noisy_image)
    results['NLM Filter'] = filters.nlm_filter(noisy_image)
    
    return results


test_noises = {
    'Gaussian': simulator.gaussian_noise(test_image, sigma=30)[0],
    'Salt & Pepper': simulator.salt_pepper_noise(test_image, salt_prob=0.05, pepper_prob=0.05)[0],
    'Speckle': simulator.speckle_noise(test_image, variance=0.05)[0]
}


fig, axes = plt.subplots(3, 7, figsize=(20, 10))

for row, (noise_name, noisy) in enumerate(test_noises.items()):
    
    axes[row, 0].imshow(noisy, cmap='gray')
    axes[row, 0].set_title(f'Noisy\n{noise_name}', fontsize=9)
    axes[row, 0].axis('off')
    
    
    results = test_filters_on_noise(noisy, noise_name)
    
    for col, (filter_name, filtered) in enumerate(results.items(), 1):
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(filter_name, fontsize=9)
        axes[row, col].axis('off')

plt.suptitle('Filter Performance on Different Noise Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()





class FilterAnalyzer:
    """Analyze filter performance using multiple metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def psnr(self, original, filtered):
        """Peak Signal-to-Noise Ratio"""
        if original.shape != filtered.shape:
            filtered = cv2.resize(filtered, (original.shape[1], original.shape[0]))
        
        mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def mae(self, original, filtered):
        """Mean Absolute Error"""
        if original.shape != filtered.shape:
            filtered = cv2.resize(filtered, (original.shape[1], original.shape[0]))
        
        return np.mean(np.abs(original.astype(float) - filtered.astype(float)))
    
    def edge_preservation(self, original, filtered):
        """Measure how well edges are preserved"""
        
        sobel_orig = np.abs(cv2.Sobel(original, cv2.CV_64F, 1, 0)) + \
                     np.abs(cv2.Sobel(original, cv2.CV_64F, 0, 1))
        sobel_filt = np.abs(cv2.Sobel(filtered, cv2.CV_64F, 1, 0)) + \
                     np.abs(cv2.Sobel(filtered, cv2.CV_64F, 0, 1))
        
        
        correlation = np.corrcoef(sobel_orig.ravel(), sobel_filt.ravel())[0, 1]
        return correlation


analyzer = FilterAnalyzer()

print("\n📊 Filter Performance Analysis:")


for noise_name, noisy in test_noises.items():
    print(f"\n🔍 Noise Type: {noise_name}")
    print("-" * 60)
    print(f"{'Filter':<20} {'PSNR (dB)':<12} {'MAE':<10} {'Edge Preservation':<15}")
    print("-" * 60)
    
    results = test_filters_on_noise(noisy, noise_name)
    
    for filter_name, filtered in results.items():
        psnr_val = analyzer.psnr(test_image, filtered)
        mae_val = analyzer.mae(test_image, filtered)
        edge_val = analyzer.edge_preservation(test_image, filtered)
        
        print(f"{filter_name:<20} {psnr_val:<12.2f} {mae_val:<10.2f} {edge_val:<15.3f}")






print("PART 8: REAL-WORLD SCENARIO - MEDICAL ULTRASOUND")
print("Speckle noise reduction in medical imaging")


def create_ultrasound_simulation():
    """Create a simulated ultrasound image"""
    img = np.zeros((300, 300), dtype=np.uint8)
    
    
    cv2.rectangle(img, (50, 50), (250, 250), 150, -1)
    
    
    cv2.ellipse(img, (150, 150), (80, 60), 0, 0, 360, 100, 2)
    
    
    cv2.circle(img, (150, 150), 30, 50, -1)
    cv2.circle(img, (150, 150), 20, 80, -1)
    
    
    for _ in range(1000):
        x, y = np.random.randint(50, 250), np.random.randint(50, 250)
        delta = np.random.randint(-20, 20)
        pixel = np.int16(img[y, x])
        img[y, x] = np.clip(pixel + delta, 0, 255).astype(np.uint8)
    
    return img


ultrasound = create_ultrasound_simulation()


noisy_ultrasound, _ = simulator.speckle_noise(ultrasound, variance=0.1)


filters_to_test = {
    'Median': filters.median_filter(noisy_ultrasound, kernel_size=3),
    'Bilateral': filters.bilateral_filter(noisy_ultrasound),
    'NLM': filters.nlm_filter(noisy_ultrasound, h=15),
    'Anisotropic': filters.anisotropic_diffusion(noisy_ultrasound)
}


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(ultrasound, cmap='gray')
axes[0, 0].set_title('Clean Ultrasound (Simulated)', fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_ultrasound, cmap='gray')
axes[0, 1].set_title('With Speckle Noise', fontweight='bold')
axes[0, 1].axis('off')

for idx, (filter_name, filtered) in enumerate(filters_to_test.items(), 2):
    ax = axes[idx//3, idx%3]
    ax.imshow(filtered, cmap='gray')
    ax.set_title(f'Restored: {filter_name}', fontweight='bold')
    ax.axis('off')
    
    
    psnr_before = analyzer.psnr(ultrasound, noisy_ultrasound)
    psnr_after = analyzer.psnr(ultrasound, filtered)
    ax.text(10, 20, f'PSNR: {psnr_after:.1f}dB\n(+{psnr_after-psnr_before:.1f}dB)', 
            color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

plt.suptitle('Medical Ultrasound Denoising - Speckle Reduction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()





class AdaptiveFilter:
    """
    Filters that adapt to local image characteristics
    """
    
    @staticmethod
    def adaptive_median(image, max_window=7):
        """
        Adaptive Median Filter
        - Increases window size in noisy areas
        - Preserves detail in clean areas
        """
        result = image.copy()
        height, width = image.shape
        
        for i in range(height):
            for j in range(width):
                window_size = 3
                while window_size <= max_window:
                    half = window_size // 2
                    
                    i_min, i_max = max(0, i-half), min(height, i+half+1)
                    j_min, j_max = max(0, j-half), min(width, j+half+1)
                    window = image[i_min:i_max, j_min:j_max]
                    
                    
                    med = np.median(window)
                    mn = np.min(window)
                    mx = np.max(window)
                    
                    
                    if mn < med < mx:
                        if mn < image[i, j] < mx:
                            result[i, j] = image[i, j]
                        else:
                            result[i, j] = med
                        break
                    else:
                        window_size += 2
                
                if window_size > max_window:
                    result[i, j] = med
        
        return result
    
    @staticmethod
    def adaptive_wiener(image, window_size=5, noise_var=None):
        """
        Adaptive Wiener Filter
        - Adjusts strength based on local statistics
        """
        if noise_var is None:
            noise_var = np.var(image[::10, ::10])
        
        
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        local_mean = cv2.filter2D(image.astype(float), -1, kernel)
        local_var = cv2.filter2D(image.astype(float)**2, -1, kernel) - local_mean**2
        
        
        result = local_mean + (local_var - noise_var) / local_var * (image - local_mean)
        result[local_var < noise_var] = local_mean[local_var < noise_var]
        
        return np.clip(result, 0, 255).astype(np.uint8)


adaptive = AdaptiveFilter()

print("\n🔧 Testing Adaptive Filters...")


mixed_noisy = test_image.copy()
mixed_noisy, _ = simulator.gaussian_noise(mixed_noisy, sigma=20)
mixed_noisy, _ = simulator.salt_pepper_noise(mixed_noisy, salt_prob=0.02, pepper_prob=0.02)


adaptive_median_result = adaptive.adaptive_median(mixed_noisy)
adaptive_wiener_result = adaptive.adaptive_wiener(mixed_noisy)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(test_image, cmap='gray')
axes[0, 0].set_title('Original Clean', fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(mixed_noisy, cmap='gray')
axes[0, 1].set_title('Mixed Noise', fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(adaptive_median_result, cmap='gray')
axes[0, 2].set_title('Adaptive Median', fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(adaptive_wiener_result, cmap='gray')
axes[1, 0].set_title('Adaptive Wiener', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(filters.median_filter(mixed_noisy), cmap='gray')
axes[1, 1].set_title('Standard Median', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(filters.gaussian_filter(mixed_noisy), cmap='gray')
axes[1, 2].set_title('Standard Gaussian', fontweight='bold')
axes[1, 2].axis('off')

plt.suptitle('Adaptive vs Standard Filters Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()






print("PRACTICAL FILTER SELECTION GUIDE")
print("Choose the right filter for your noise type")


filter_guide = {
    "Gaussian Noise": {
        "Best Filter": "Wiener Filter or Gaussian Blur",
        "Why": "Mathematically optimal for Gaussian distribution",
        "Code": "cv2.GaussianBlur(img, (5,5), 1.5)"
    },
    "Salt & Pepper Noise": {
        "Best Filter": "Median Filter",
        "Why": "Removes outliers perfectly without blurring",
        "Code": "cv2.medianBlur(img, 5)"
    },
    "Speckle Noise": {
        "Best Filter": "Bilateral Filter or NLM",
        "Why": "Preserves edges while reducing multiplicative noise",
        "Code": "cv2.bilateralFilter(img, 9, 75, 75)"
    },
    "Mixed Noise": {
        "Best Filter": "Adaptive Median + Wiener",
        "Why": "Adapts to local noise characteristics",
        "Code": "adaptive = AdaptiveFilter(); adaptive.adaptive_median(img)"
    },
    "Periodic Noise": {
        "Best Filter": "Frequency domain filters",
        "Why": "Noise has distinct frequency signature",
        "Code": "Use FFT + notch filter"
    },
    "Poisson Noise": {
        "Best Filter": "Anscombe transform + Gaussian filter",
        "Why": "Stabilizes variance before filtering",
        "Code": "img_transformed = 2*sqrt(img + 3/8)"
    }
}

for noise_type, info in filter_guide.items():
    print(f"\n📌 {noise_type}:")
    print(f"   • Best Filter: {info['Best Filter']}")
    print(f"   • Why: {info['Why']}")
    print(f"   • Code: {info['Code']}")





def interactive_noise_demo():
    """
    Interactive demonstration - adjust noise parameters in real-time
    """
    
    print("INTERACTIVE NOISE SIMULATOR")
    print("Adjust parameters to see real-time effects")
    
    
    
    demo_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(demo_img, (50, 50), (150, 150), 200, -1)
    cv2.circle(demo_img, (100, 100), 30, 50, -1)
    
    
    noise_levels = [0, 10, 25, 50, 75, 100]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, sigma in enumerate(noise_levels):
        ax = axes[idx//3, idx%3]
        
        if sigma == 0:
            ax.imshow(demo_img, cmap='gray')
            ax.set_title('Original (No Noise)', fontweight='bold')
        else:
            noisy, _ = simulator.gaussian_noise(demo_img, sigma=sigma)
            ax.imshow(noisy, cmap='gray')
            ax.set_title(f'Gaussian Noise σ={sigma}', fontweight='bold')
            
            
            filtered = filters.median_filter(noisy, kernel_size=3)
            
            
            psnr_val = analyzer.psnr(demo_img, filtered)
            ax.text(10, 20, f'PSNR: {psnr_val:.1f}dB', 
                   color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        
        ax.axis('off')
    
    plt.suptitle('Effect of Increasing Noise Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


interactive_noise_demo()






print("KEY FORMULAS AND EQUATIONS")


formulas = {
    "Gaussian Noise": "p(z) = (1/σ√2π) * e^(-(z-μ)²/2σ²)",
    "Rayleigh Noise": "p(z) = (z/σ²) * e^(-z²/2σ²) for z ≥ 0",
    "Gamma Noise": "p(z) = [aᵇzᵇ⁻¹e^(-az)]/Γ(b) for z ≥ 0",
    "Exponential Noise": "p(z) = ae^(-az) for z ≥ 0",
    "Uniform Noise": "p(z) = 1/(b-a) for a ≤ z ≤ b",
    "Salt & Pepper": "p(z) = {P_s for z=255, P_p for z=0, 1-(P_s+P_p) otherwise}",
    "Mean Filter": "g(x,y) = (1/mn)∑f(x,y)",
    "Median Filter": "g(x,y) = median{f(s,t)}",
    "Gaussian Filter": "G(x,y) = (1/2πσ²)e^-(x²+y²)/2σ²",
    "Wiener Filter": "F(u,v) = [H*(u,v)/|H(u,v)|²] * [|H(u,v)|²/(|H(u,v)|² + S_n(u,v)/S_f(u,v))]"
}

for name, formula in formulas.items():
    print(f"\n• {name}:")
    print(f"  {formula}")






print("SUMMARY - NOISE SIMULATION & FILTERING")
print("Quick reference guide")


summary = """
┌─────────────────┬──────────────────────┬─────────────────────────┐
│ Noise Type      │ Best Filter          │ Real-World Application  │
├─────────────────┼──────────────────────┼─────────────────────────┤
│ Gaussian        │ Wiener / Gaussian    │ Low-light photos        │
│ Salt & Pepper   │ Median                │ Fax machines, old films │
│ Speckle         │ Bilateral / NLM       │ Ultrasound, radar       │
│ Poisson         │ Anscombe + Gaussian   │ Astronomy, microscopy   │
│ Periodic        │ Notch filter (FFT)    │ Electrical interference │
│ Mixed           │ Adaptive filters      │ Real photographs        │
│ Uniform         │ Mean filter           │ Quantization noise      │
│ Rayleigh        │ Specialized filters   │ LIDAR, sonar           │
└─────────────────┴──────────────────────┴─────────────────────────┘

FILTER SELECTION CRITERIA:
• Speed required? → Mean/Median (fast) vs NLM (slow)
• Edge preservation? → Bilateral/Anisotropic
• Noise known? → Use specific filter
• Noise unknown? → Adaptive/Median filters
• Real-time? → Gaussian/Median (optimized)

QUALITY METRICS REFERENCE:
• PSNR > 40 dB: Excellent restoration
• PSNR 30-40 dB: Good restoration
• PSNR 20-30 dB: Acceptable
• PSNR < 20 dB: Poor
"""

print(summary)


cv2.imwrite('images/test_pattern.png', test_image)
cv2.imwrite('images/ultrasound_sim.png', noisy_ultrasound)
print("\n💾 Example images saved to 'images/' folder")


print("✅ NOISE SIMULATION & FILTERING GUIDE COMPLETE")
print("You now understand:")
print("• 8+ types of noise and their characteristics")
print("• 10+ filtering techniques with mathematical foundations")
print("• How to choose the right filter for each situation")
print("• How to evaluate filter performance")
print("• Real-world applications in medical imaging, photography, etc.")
