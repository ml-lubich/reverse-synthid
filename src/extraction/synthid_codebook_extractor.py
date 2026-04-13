"""
SynthID Codebook Extractor

Based on analysis of 250 Gemini images, this script extracts and saves
the discovered SynthID watermark codebook for detection purposes.

KEY FINDINGS:
1. Carrier frequencies at specific locations (±14, ±14), (±126, ±14), etc.
2. High phase coherence (0.99+) at carrier frequencies
3. Noise correlation of ~0.21 between watermarked images
4. Noise structure ratio of ~1.32

The codebook consists of:
1. Reference noise pattern (average across all images)
2. Carrier frequency locations and expected phases
3. Detection thresholds
"""

import os
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
import pywt
import json
import pickle


def wavelet_denoise(channel, wavelet='db4', level=3):
    """Wavelet-based denoising."""
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    detail = coeffs[-1][0]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(channel.size))
    
    new_coeffs = [coeffs[0]]
    for details in coeffs[1:]:
        new_details = tuple(pywt.threshold(d, threshold, mode='soft') for d in details)
        new_coeffs.append(new_details)
    
    denoised = pywt.waverec2(new_coeffs, wavelet)
    return denoised[:channel.shape[0], :channel.shape[1]]


def extract_codebook(image_dir, output_path, max_images=250, size=None):
    """Extract SynthID codebook from a collection of watermarked images.
    
    If size is None, uses the native resolution of the first image found.
    """
    
    print(f"Loading images from {image_dir}...")
    
    # Load images
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    images = []
    native_h, native_w = None, None
    
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in extensions:
            path = os.path.join(image_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if native_h is None:
                    native_h, native_w = img.shape[:2]
                    if size is not None:
                        native_h = native_w = size
                    print(f"  Using resolution: {native_h}x{native_w}")
                img = cv2.resize(img, (native_w, native_h))
                images.append(img)
                if len(images) >= max_images:
                    break
    
    print(f"Loaded {len(images)} images")
    images = np.array(images)
    
    # ================================================================
    # 1. EXTRACT REFERENCE NOISE PATTERN
    # ================================================================
    print("Extracting reference noise pattern...")
    
    noise_sum = np.zeros((native_h, native_w, 3), dtype=np.float64)
    
    for img in images:
        img_f = img.astype(np.float32) / 255.0
        for c in range(3):
            denoised = wavelet_denoise(img_f[:, :, c])
            noise_sum[:, :, c] += img_f[:, :, c] - denoised
    
    reference_noise = noise_sum / len(images)
    
    # ================================================================
    # 2. EXTRACT CARRIER FREQUENCIES
    # ================================================================
    print("Extracting carrier frequencies...")
    
    magnitude_sum = None
    phase_sum = None
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = fft2(gray)
        fshift = fftshift(f)
        
        if magnitude_sum is None:
            magnitude_sum = np.abs(fshift)
            phase_sum = np.exp(1j * np.angle(fshift))
        else:
            magnitude_sum += np.abs(fshift)
            phase_sum += np.exp(1j * np.angle(fshift))
    
    avg_magnitude = magnitude_sum / len(images)
    phase_coherence = np.abs(phase_sum) / len(images)
    avg_phase = np.angle(phase_sum)
    
    # Find carrier frequencies (high coherence, significant magnitude)
    log_mag = np.log1p(avg_magnitude)
    combined_score = log_mag * phase_coherence
    
    # Get top carriers
    threshold = np.percentile(combined_score, 99.5)
    carrier_mask = combined_score > threshold
    carrier_locs = np.where(carrier_mask)
    
    center_y = native_h // 2
    center_x = native_w // 2
    carriers = []
    for y, x in zip(carrier_locs[0], carrier_locs[1]):
        freq_y, freq_x = y - center_y, x - center_x
        # Skip DC
        if abs(freq_y) < 5 and abs(freq_x) < 5:
            continue
        carriers.append({
            'position': (int(y), int(x)),
            'frequency': (int(freq_y), int(freq_x)),
            'magnitude': float(avg_magnitude[y, x]),
            'phase': float(avg_phase[y, x]),
            'coherence': float(phase_coherence[y, x])
        })
    
    carriers.sort(key=lambda c: c['coherence'] * np.log1p(c['magnitude']), reverse=True)
    carriers = carriers[:100]  # Top 100 carriers
    
    # ================================================================
    # 3. COMPUTE DETECTION THRESHOLDS
    # ================================================================
    print("Computing detection thresholds...")
    
    # Compute noise correlations for threshold calibration
    correlations = []
    for i in range(min(50, len(images))):
        for j in range(i+1, min(50, len(images))):
            img1 = images[i].astype(np.float32) / 255.0
            img2 = images[j].astype(np.float32) / 255.0
            
            noise1 = np.zeros((native_h, native_w, 3))
            noise2 = np.zeros((native_h, native_w, 3))
            
            for c in range(3):
                noise1[:, :, c] = img1[:, :, c] - wavelet_denoise(img1[:, :, c])
                noise2[:, :, c] = img2[:, :, c] - wavelet_denoise(img2[:, :, c])
            
            corr = np.corrcoef(noise1.ravel(), noise2.ravel())[0, 1]
            correlations.append(corr)
    
    correlation_mean = float(np.mean(correlations))
    correlation_std = float(np.std(correlations))
    
    # Detection threshold: if correlation > mean - 2*std, likely watermarked
    detection_threshold = correlation_mean - 2 * correlation_std
    
    # ================================================================
    # 4. CREATE CODEBOOK
    # ================================================================
    print("Creating codebook...")
    
    codebook = {
        'version': '1.0',
        'source': 'Gemini/SynthID',
        'n_images_analyzed': len(images),
        'image_size': (native_h, native_w),
        
        # Reference patterns
        'reference_noise': reference_noise,
        'reference_magnitude': avg_magnitude,
        'reference_phase': avg_phase,
        'phase_coherence': phase_coherence,
        
        # Carrier frequencies
        'carriers': carriers,
        'n_carriers': len(carriers),
        
        # Detection parameters
        'correlation_mean': correlation_mean,
        'correlation_std': correlation_std,
        'detection_threshold': detection_threshold,
        'noise_structure_ratio': 1.32,  # From previous analysis
        
        # Key carrier frequencies (simplified)
        'key_frequencies': [
            {'freq': (14, 14), 'coherence': 0.9996},
            {'freq': (-14, -14), 'coherence': 0.9996},
            {'freq': (126, 14), 'coherence': 0.9996},
            {'freq': (-126, -14), 'coherence': 0.9996},
            {'freq': (98, -14), 'coherence': 0.9994},
            {'freq': (-98, 14), 'coherence': 0.9994},
            {'freq': (128, 128), 'coherence': 0.9925},
            {'freq': (-128, -128), 'coherence': 0.9925},
        ]
    }
    
    # Save codebook
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save as pickle (includes numpy arrays, protocol=4 for compatibility)
    with open(output_path, 'wb') as f:
        pickle.dump(codebook, f, protocol=4)
    
    # Save metadata as JSON
    json_path = output_path.replace('.pkl', '_meta.json')
    meta = {
        'version': codebook['version'],
        'source': codebook['source'],
        'n_images_analyzed': codebook['n_images_analyzed'],
        'image_size': codebook['image_size'],
        'n_carriers': codebook['n_carriers'],
        'correlation_mean': codebook['correlation_mean'],
        'correlation_std': codebook['correlation_std'],
        'detection_threshold': codebook['detection_threshold'],
        'key_frequencies': codebook['key_frequencies'],
        'carriers': codebook['carriers'][:20]  # Top 20 for reference
    }
    
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nCodebook saved to {output_path}")
    print(f"Metadata saved to {json_path}")
    
    return codebook


def detect_synthid(image_path, codebook_path):
    """
    Detect SynthID watermark in an image using the extracted codebook.
    
    Returns:
        dict with detection results
    """
    # Load codebook
    with open(codebook_path, 'rb') as f:
        codebook = pickle.load(f)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not load image'}
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = codebook['image_size']
    if isinstance(img_size, (list, tuple)):
        target_h, target_w = int(img_size[0]), int(img_size[1])
    else:
        target_h = target_w = int(img_size)
    img = cv2.resize(img, (target_w, target_h))
    img_f = img.astype(np.float32) / 255.0
    
    # Extract noise pattern
    noise = np.zeros((target_h, target_w, 3))
    for c in range(3):
        noise[:, :, c] = img_f[:, :, c] - wavelet_denoise(img_f[:, :, c])
    
    # Method 1: Correlation with reference noise
    ref_noise = codebook['reference_noise']
    correlation = np.corrcoef(noise.ravel(), ref_noise.ravel())[0, 1]
    
    # Method 2: Check carrier frequencies
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    center_y = target_h // 2
    center_x = target_w // 2
    carrier_scores = []
    for carrier in codebook['carriers'][:20]:
        y, x = carrier['position']
        expected_phase = carrier['phase']
        actual_phase = phase[y, x]
        
        # Phase difference (accounting for wrap-around)
        phase_diff = np.abs(np.angle(np.exp(1j * (actual_phase - expected_phase))))
        phase_match = 1 - phase_diff / np.pi
        
        carrier_scores.append(phase_match)
    
    avg_phase_match = float(np.mean(carrier_scores))
    
    # Method 3: Noise structure ratio
    noise_gray = np.mean(noise, axis=2)
    structure_ratio = float(np.std(noise_gray) / (np.mean(np.abs(noise_gray)) + 1e-10))
    
    # Detection decision
    threshold = codebook['detection_threshold']
    is_watermarked = (
        correlation > threshold and
        avg_phase_match > 0.5 and
        0.8 < structure_ratio < 1.8
    )
    
    # Confidence score
    confidence = min(1.0, max(0.0, 
        (correlation - threshold) / (codebook['correlation_mean'] - threshold) * 0.4 +
        avg_phase_match * 0.4 +
        (1 - abs(structure_ratio - 1.32) / 0.5) * 0.2
    ))
    
    return {
        'is_watermarked': bool(is_watermarked),
        'confidence': float(confidence),
        'correlation': float(correlation),
        'phase_match': float(avg_phase_match),
        'structure_ratio': float(structure_ratio),
        'threshold': float(threshold),
        'reference_correlation_mean': float(codebook['correlation_mean'])
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SynthID Codebook Extractor and Detector')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract codebook from images')
    extract_parser.add_argument('image_dir', type=str, help='Directory with watermarked images')
    extract_parser.add_argument('--output', type=str, default='./synthid_codebook.pkl', help='Output path')
    extract_parser.add_argument('--max-images', type=int, default=250, help='Max images')
    extract_parser.add_argument('--size', type=int, default=None,
                                help='Image size (default: auto-detect from images)')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark in image')
    detect_parser.add_argument('image', type=str, help='Image to check')
    detect_parser.add_argument('--codebook', type=str, default='./synthid_codebook.pkl', help='Codebook path')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_codebook(args.image_dir, args.output, args.max_images, args.size)
    elif args.command == 'detect':
        result = detect_synthid(args.image, args.codebook)
        print("\nDetection Results:")
        print(f"  Watermarked: {result['is_watermarked']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Correlation: {result['correlation']:.4f}")
        print(f"  Phase Match: {result['phase_match']:.4f}")
        print(f"  Structure Ratio: {result['structure_ratio']:.4f}")
    else:
        parser.print_help()
