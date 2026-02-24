"""
FREQUENCY DOMAIN PROCESSING - DECISION TREE
"""

def choose_frequency_method(problem_type, image_characteristics):
    """
    Decision guide for frequency domain processing
    """
    
    decisions = {
        'noise_reduction': {
            'gaussian_noise': 'Low-pass filter (Gaussian)',
            'salt_pepper': 'Median filter (spatial is better)',
            'periodic_noise': 'Notch filter',
            'mixed_noise': 'Wiener filter'
        },
        'enhancement': {
            'blurry': 'High-pass filter',
            'low_contrast': 'Homomorphic filter',
            'uneven_lighting': 'Homomorphic filter',
            'edges': 'High-pass filter'
        },
        'analysis': {
            'texture': 'Band-pass filter',
            'patterns': 'Frequency spectrum analysis',
            'defects': 'Subtract reference spectrum'
        },
        'compression': {
            'jpeg': 'DCT (Discrete Cosine Transform)',
            'wavelet': 'Wavelet transform'
        }
    }
    
    return decisions.get(problem_type, {}).get(image_characteristics, 'Analyze further')

# Example usage
print("📋 DECISION GUIDE EXAMPLES:")
print("-" * 50)

test_cases = [
    ('noise_reduction', 'periodic_noise', 'Remove scanner artifacts'),
    ('enhancement', 'uneven_lighting', 'Fix shadow on document'),
    ('analysis', 'texture', 'Classify fabric type'),
    ('noise_reduction', 'gaussian_noise', 'Clean low-light photo')
]

for problem, char, description in test_cases:
    method = choose_frequency_method(problem, char)
    print(f"• {description}: Use {method}")