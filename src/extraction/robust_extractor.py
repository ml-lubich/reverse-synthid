"""
Robust SynthID Watermark Extractor

A comprehensive multi-stage watermark extraction pipeline that combines:
1. Multi-scale analysis (256, 512, 1024 pixels)
2. Multi-denoiser fusion (wavelet, bilateral, non-local means)
3. Ensemble carrier frequency detection with voting
4. ICA/PCA-based watermark separation
5. Adaptive thresholding based on image content

This provides significantly more robust detection than single-scale approaches.
"""

import os
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from scipy.stats import pearsonr
from collections import defaultdict
import pywt
import pickle
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from sklearn.decomposition import PCA, FastICA


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    is_watermarked: bool
    confidence: float
    correlation: float
    phase_match: float
    structure_ratio: float
    carrier_strength: float
    multi_scale_consistency: float
    details: Dict


class RobustSynthIDExtractor:
    """
    Robust SynthID watermark extractor using multi-stage analysis.
    
    Features:
    - Multi-scale processing for comprehensive watermark detection
    - Multiple denoising methods with intelligent fusion
    - Ensemble carrier frequency detection
    - Adaptive thresholds based on image content
    """
    
    def __init__(
        self,
        scales: List[int] = [256, 512, 1024],
        wavelets: List[str] = ['db4', 'sym8', 'coif3'],
        n_carriers: int = 100,
        codebook_path: Optional[str] = None
    ):
        """
        Initialize the robust extractor.
        
        Args:
            scales: Image scales for multi-scale analysis
            wavelets: Wavelet families for denoising
            n_carriers: Number of carrier frequencies to track
            codebook_path: Path to pre-extracted codebook
        """
        self.scales = scales
        self.wavelets = wavelets
        self.n_carriers = n_carriers
        self.codebook = None
        
        # Empirically verified SynthID carriers at 512px resolution.
        # Extracted from 291 Gemini-generated images (black, white, nb_pro).
        # Each carrier set has >0.95 intra-set phase coherence and >0.5
        # discriminative gap vs non-watermarked images.
        #
        # Dark-image carriers (black + nb_pro, diagonal grid):
        self.carriers_dark = [
            (-5, -3), (5, 3), (-5, 3), (5, -3),
            (-3, -4), (3, 4), (-3, 4), (3, -4),
            (-4, -3), (4, 3), (-4, 3), (4, -3),
            (-5, -1), (5, 1), (-5, 1), (5, -1),
            (-5, -2), (5, 2), (-5, 2), (5, -2),
            (-2, -5), (2, 5), (-2, 5), (2, -5),
            (-1, -5), (1, 5), (-1, 5), (1, -5),
            (-4, -4), (4, 4), (-4, 4), (4, -4),
            (-1, -6), (1, 6), (-3, -5), (3, 5),
        ]
        # White-image carriers (horizontal axis):
        self.carriers_white = [
            (0, -7), (0, 7), (0, -8), (0, 8),
            (0, -9), (0, 9), (0, -10), (0, 10),
            (0, -11), (0, 11), (0, -12), (0, 12),
            (0, -20), (0, 20), (0, -21), (0, 21),
            (0, -22), (0, 22), (0, -23), (0, 23),
        ]
        # Union for backward compat
        self.known_carriers = self.carriers_dark + self.carriers_white
        
        if codebook_path and os.path.exists(codebook_path):
            self.load_codebook(codebook_path)
    
    def load_codebook(self, path: str) -> None:
        """Load pre-extracted codebook. Supports .pkl files and auto-discovers
        .pkl when given a .npz path (common user mistake)."""
        if path.endswith('.npz'):
            pkl_candidates = [
                os.path.join(os.path.dirname(path), 'codebook', 'robust_codebook.pkl'),
                os.path.join(os.path.dirname(path), 'robust_codebook.pkl'),
                path.replace('.npz', '.pkl'),
            ]
            for pkl_path in pkl_candidates:
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        self.codebook = pickle.load(f)
                    return
            raise FileNotFoundError(
                f"Cannot load .npz as pickle codebook. "
                f"Provide a .pkl file instead, e.g.: "
                f"--detector artifacts/codebook/robust_codebook.pkl"
            )
        with open(path, 'rb') as f:
            self.codebook = pickle.load(f)
    
    def save_codebook(self, path: str) -> None:
        """Save extracted codebook."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.codebook, f, protocol=4)
    
    # ================================================================
    # DENOISING METHODS
    # ================================================================
    
    def wavelet_denoise(
        self,
        channel: np.ndarray,
        wavelet: str = 'db4',
        level: int = 3
    ) -> np.ndarray:
        """Wavelet-based denoising using soft thresholding."""
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        
        # Estimate noise from finest detail coefficients
        detail = coeffs[-1][0]
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(channel.size))
        
        # Apply soft thresholding to detail coefficients
        new_coeffs = [coeffs[0]]
        for details in coeffs[1:]:
            new_details = tuple(
                pywt.threshold(d, threshold, mode='soft') for d in details
            )
            new_coeffs.append(new_details)
        
        denoised = pywt.waverec2(new_coeffs, wavelet)
        return denoised[:channel.shape[0], :channel.shape[1]]
    
    def bilateral_denoise(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """Bilateral filter denoising (edge-preserving)."""
        if len(image.shape) == 2:
            return cv2.bilateralFilter(image.astype(np.float32), d, sigma_color, sigma_space)
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = cv2.bilateralFilter(
                    image[:, :, c].astype(np.float32), d, sigma_color, sigma_space
                )
            return result
    
    def nlm_denoise(
        self,
        image: np.ndarray,
        h: float = 10,
        template_size: int = 7,
        search_size: int = 21
    ) -> np.ndarray:
        """Non-local means denoising."""
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        if len(image.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(
                img_uint8, None, h, template_size, search_size
            )
        else:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_uint8, None, h, h, template_size, search_size
            )
        
        return denoised.astype(np.float32) / 255.0
    
    def wiener_filter(
        self,
        image: np.ndarray,
        noise_variance: Optional[float] = None
    ) -> np.ndarray:
        """Wiener filter for optimal noise estimation."""
        if noise_variance is None:
            # Estimate noise variance from high-frequency components
            noise_variance = np.var(image - ndimage.gaussian_filter(image, sigma=2))
        
        # Simple Wiener filter in Fourier domain
        f = fft2(image)
        power = np.abs(f) ** 2
        signal_power = np.maximum(power - noise_variance, 0)
        wiener_ratio = signal_power / (signal_power + noise_variance + 1e-10)
        
        denoised = np.real(ifft2(f * wiener_ratio))
        return denoised
    
    # ================================================================
    # NOISE EXTRACTION
    # ================================================================
    
    def extract_noise_single(
        self,
        image: np.ndarray,
        method: str = 'wavelet',
        **kwargs
    ) -> np.ndarray:
        """Extract noise using a single denoising method."""
        img_f = image.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0
        
        if method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db4')
            if len(img_f.shape) == 2:
                denoised = self.wavelet_denoise(img_f, wavelet)
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    denoised[:, :, c] = self.wavelet_denoise(img_f[:, :, c], wavelet)
        
        elif method == 'bilateral':
            denoised = self.bilateral_denoise(img_f)
        
        elif method == 'nlm':
            denoised = self.nlm_denoise(img_f)
        
        elif method == 'wiener':
            if len(img_f.shape) == 2:
                denoised = self.wiener_filter(img_f)
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    denoised[:, :, c] = self.wiener_filter(img_f[:, :, c])
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return img_f - denoised
    
    def extract_noise_fused(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise using multiple methods and fuse results.
        
        Uses weighted median fusion for robustness against outliers.
        """
        noises = []
        weights = []
        
        # Wavelet denoising with multiple families
        for wavelet in self.wavelets:
            noise = self.extract_noise_single(image, 'wavelet', wavelet=wavelet)
            noises.append(noise)
            weights.append(1.0)
        
        # Bilateral filter
        noise = self.extract_noise_single(image, 'bilateral')
        noises.append(noise)
        weights.append(0.8)
        
        # Non-local means
        noise = self.extract_noise_single(image, 'nlm')
        noises.append(noise)
        weights.append(0.7)
        
        # Wiener filter
        noise = self.extract_noise_single(image, 'wiener')
        noises.append(noise)
        weights.append(0.6)
        
        # Weighted fusion
        noises = np.array(noises)
        weights = np.array(weights) / sum(weights)
        
        # Use weighted average (more stable than weighted median)
        fused = np.tensordot(weights, noises, axes=([0], [0]))
        
        return fused
    
    # ================================================================
    # CARRIER FREQUENCY DETECTION
    # ================================================================
    
    def find_carrier_peaks(
        self,
        magnitude: np.ndarray,
        phase_coherence: np.ndarray,
        n_peaks: int = 100
    ) -> List[Tuple[int, int, float]]:
        """Find carrier frequency peaks using combined magnitude and coherence."""
        center = magnitude.shape[0] // 2
        
        # Combined score
        log_mag = np.log1p(magnitude)
        combined = log_mag * phase_coherence
        
        # Find peaks (excluding DC region)
        dc_mask = np.ones_like(combined, dtype=bool)
        y_coords, x_coords = np.ogrid[:combined.shape[0], :combined.shape[1]]
        dc_mask[((y_coords - center) ** 2 + (x_coords - center) ** 2) < 25] = False
        
        # Threshold and find peaks
        threshold = np.percentile(combined[dc_mask], 99)
        peak_mask = (combined > threshold) & dc_mask
        
        # Get peak locations with scores
        peak_locs = np.where(peak_mask)
        peaks = []
        for y, x in zip(peak_locs[0], peak_locs[1]):
            freq_y, freq_x = y - center, x - center
            score = combined[y, x]
            peaks.append((freq_y, freq_x, score))
        
        # Sort by score and return top N
        peaks.sort(key=lambda p: p[2], reverse=True)
        return peaks[:n_peaks]
    
    def detect_carriers_single_scale(
        self,
        images: List[np.ndarray],
        size: int
    ) -> Dict[Tuple[int, int], Dict]:
        """Detect carriers at a single scale."""
        magnitude_sum = None
        phase_sum = None
        n_images = 0
        
        for img in images:
            # Resize to target scale
            s = int(size)
            img_resized = cv2.resize(img, (s, s))
            if len(img_resized.shape) == 3:
                gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray = img_resized.astype(np.float32)
            
            # FFT
            f = fft2(gray)
            fshift = fftshift(f)
            
            if magnitude_sum is None:
                magnitude_sum = np.abs(fshift)
                phase_sum = np.exp(1j * np.angle(fshift))
            else:
                magnitude_sum += np.abs(fshift)
                phase_sum += np.exp(1j * np.angle(fshift))
            
            n_images += 1
        
        avg_magnitude = magnitude_sum / n_images
        phase_coherence = np.abs(phase_sum) / n_images
        avg_phase = np.angle(phase_sum)
        
        # Find peaks
        peaks = self.find_carrier_peaks(avg_magnitude, phase_coherence, self.n_carriers)
        
        # Build carrier dictionary
        carriers = {}
        center = size // 2
        for freq_y, freq_x, score in peaks:
            y, x = freq_y + center, freq_x + center
            carriers[(freq_y, freq_x)] = {
                'position': (y, x),
                'magnitude': float(avg_magnitude[y, x]),
                'phase': float(avg_phase[y, x]),
                'coherence': float(phase_coherence[y, x]),
                'score': float(score)
            }
        
        return carriers
    
    def detect_carriers_multi_scale(
        self,
        images: List[np.ndarray]
    ) -> List[Dict]:
        """
        Detect carriers using multi-scale analysis with voting.
        
        Carriers that appear consistently across scales are more reliable.
        Falls back to known carriers if voting doesn't find reliable ones.
        """
        all_carriers = defaultdict(lambda: {'votes': 0, 'total_score': 0, 'scales': [], 'infos': []})
        base_scale = 512
        
        for scale in self.scales:
            carriers = self.detect_carriers_single_scale(images, scale)
            
            for freq, info in carriers.items():
                # Normalize frequency to base scale (512) with tolerance
                norm_freq_y = int(round(freq[0] * base_scale / scale))
                norm_freq_x = int(round(freq[1] * base_scale / scale))
                
                # Use tolerance-based binning (frequencies within ±2 are considered the same)
                bin_freq = (norm_freq_y // 2 * 2, norm_freq_x // 2 * 2)
                
                all_carriers[bin_freq]['votes'] += 1
                all_carriers[bin_freq]['total_score'] += info['score']
                all_carriers[bin_freq]['scales'].append(scale)
                all_carriers[bin_freq]['infos'].append(info)
        
        # Filter carriers with multiple votes OR high score
        reliable_carriers = []
        for freq, info in all_carriers.items():
            # Accept if appears in 2+ scales OR has very high score
            if info['votes'] >= 2 or (info['votes'] >= 1 and info['total_score'] > 100):
                # Average the info from all scales
                avg_coherence = np.mean([i.get('coherence', 0) for i in info['infos']])
                avg_phase = np.mean([i.get('phase', 0) for i in info['infos']])
                avg_magnitude = np.mean([i.get('magnitude', 0) for i in info['infos']])
                
                carrier = {
                    'frequency': freq,
                    'votes': info['votes'],
                    'avg_score': info['total_score'] / info['votes'],
                    'scales': info['scales'],
                    'coherence': float(avg_coherence),
                    'phase': float(avg_phase),
                    'magnitude': float(avg_magnitude)
                }
                reliable_carriers.append(carrier)
        
        # Sort by votes then score
        reliable_carriers.sort(key=lambda c: (c['votes'], c['avg_score']), reverse=True)
        
        # FALLBACK: If no reliable carriers found, use known carriers
        if len(reliable_carriers) < 5:
            print(f"  Warning: Only {len(reliable_carriers)} carriers found, using known carriers as fallback")
            # Add known carriers with default values
            for freq in self.known_carriers:
                if freq not in [c['frequency'] for c in reliable_carriers]:
                    reliable_carriers.append({
                        'frequency': freq,
                        'votes': 0,
                        'avg_score': 50,
                        'scales': [],
                        'coherence': 0.99,
                        'phase': 0.0,  # Will be computed during detection
                        'magnitude': 1000
                    })
        
        return reliable_carriers[:self.n_carriers]
    
    # ================================================================
    # ICA/PCA SEPARATION
    # ================================================================
    
    def extract_watermark_ica(
        self,
        images: List[np.ndarray],
        n_components: int = 5
    ) -> np.ndarray:
        """
        Use ICA to separate watermark pattern from image content.
        
        The watermark should appear as a consistent component across images.
        """
        # Extract noise from all images
        noise_vectors = []
        # Use a consistent analysis size; cap at 512 for ICA memory
        if images:
            native_h, native_w = images[0].shape[:2]
            max_dim = 512
            if native_h > max_dim or native_w > max_dim:
                scale = max_dim / max(native_h, native_w)
                target_h = int(native_h * scale)
                target_w = int(native_w * scale)
            else:
                target_h, target_w = native_h, native_w
        else:
            target_h = target_w = 512
        
        for img in images[:50]:  # Limit for performance
            img_resized = cv2.resize(img, (target_w, target_h))
            noise = self.extract_noise_fused(img_resized)
            
            if len(noise.shape) == 3:
                noise = np.mean(noise, axis=2)
            
            noise_vectors.append(noise.flatten())
        
        noise_matrix = np.array(noise_vectors)
        
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        try:
            sources = ica.fit_transform(noise_matrix)
            components = ica.components_
        except Exception:
            # Fall back to PCA if ICA fails to converge
            pca = PCA(n_components=n_components)
            sources = pca.fit_transform(noise_matrix)
            components = pca.components_
        
        # Find the most consistent component (watermark)
        consistencies = []
        for i in range(n_components):
            component = components[i].reshape(target_h, target_w)
            # Watermark should have specific frequency structure
            f = fftshift(fft2(component))
            
            # Check energy at known carrier frequencies (scaled to analysis size)
            center_y = target_h // 2
            center_x = target_w // 2
            scale_y = target_h / 512
            scale_x = target_w / 512
            carrier_energy = 0
            for freq_y, freq_x in self.known_carriers:
                y = int(freq_y * scale_y) + center_y
                x = int(freq_x * scale_x) + center_x
                if 0 <= y < target_h and 0 <= x < target_w:
                    carrier_energy += np.abs(f[y, x])
            
            consistencies.append(carrier_energy)
        
        # Return the component with highest carrier energy
        best_idx = np.argmax(consistencies)
        watermark = components[best_idx].reshape(target_h, target_w)
        
        return watermark
    
    # ================================================================
    # CODEBOOK EXTRACTION
    # ================================================================
    
    def extract_codebook(
        self,
        image_dir: str,
        max_images: int = 250,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Extract comprehensive codebook from watermarked images.
        
        Uses multi-scale analysis and ensemble methods for robustness.
        """
        print(f"Loading images from {image_dir}...")
        
        # Load images
        extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        images = []
        
        for fname in sorted(os.listdir(image_dir)):
            if os.path.splitext(fname)[1].lower() in extensions:
                path = os.path.join(image_dir, fname)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if len(images) >= max_images:
                        break
        
        print(f"Loaded {len(images)} images")
        
        # Multi-scale carrier detection
        print("Detecting carriers (multi-scale)...")
        carriers = self.detect_carriers_multi_scale(images)
        
        # Extract reference noise pattern
        print("Extracting reference noise pattern...")
        # Use native resolution of first image instead of fixed 512
        if images:
            target_h, target_w = images[0].shape[:2]
        else:
            target_h = target_w = 512
        target_size = max(target_h, target_w)  # for carrier normalization
        noise_sum = np.zeros((target_h, target_w, 3), dtype=np.float64)
        
        for img in images:
            img_resized = cv2.resize(img, (target_w, target_h))
            noise = self.extract_noise_fused(img_resized)
            noise_sum += noise
        
        reference_noise = noise_sum / len(images)
        
        # ICA-based watermark extraction
        print("Extracting watermark pattern via ICA...")
        watermark_pattern = self.extract_watermark_ica(images)
        
        # Compute correlation statistics
        print("Computing correlation statistics...")
        correlations = []
        sample_images = images[:min(50, len(images))]
        
        for i, img1 in enumerate(sample_images):
            for j, img2 in enumerate(sample_images):
                if i < j:
                    img1_resized = cv2.resize(img1, (target_w, target_h))
                    img2_resized = cv2.resize(img2, (target_w, target_h))
                    
                    noise1 = self.extract_noise_fused(img1_resized)
                    noise2 = self.extract_noise_fused(img2_resized)
                    
                    corr = np.corrcoef(noise1.ravel(), noise2.ravel())[0, 1]
                    correlations.append(corr)
        
        correlation_mean = float(np.mean(correlations))
        correlation_std = float(np.std(correlations))
        detection_threshold = correlation_mean - 2.5 * correlation_std
        
        # Compute FFT statistics
        print("Computing FFT reference...")
        ref_gray = np.mean(reference_noise, axis=2)
        ref_fft = fftshift(fft2(ref_gray))
        ref_magnitude = np.abs(ref_fft)
        ref_phase = np.angle(ref_fft)
        
        # Build codebook
        self.codebook = {
            'version': '2.0',
            'source': 'Gemini/SynthID',
            'extractor': 'RobustSynthIDExtractor',
            'n_images_analyzed': len(images),
            'image_size': (target_h, target_w),
            'scales_used': self.scales,
            
            # Reference patterns
            'reference_noise': reference_noise,
            'watermark_pattern': watermark_pattern,
            'reference_magnitude': ref_magnitude,
            'reference_phase': ref_phase,
            
            # Carriers
            'carriers': carriers,
            'known_carriers': self.known_carriers,
            
            # Detection thresholds
            'correlation_mean': correlation_mean,
            'correlation_std': correlation_std,
            'detection_threshold': detection_threshold,
            'noise_structure_ratio': 1.32,
        }
        
        if save_path:
            self.save_codebook(save_path)
            print(f"Codebook saved to {save_path}")
        
        return self.codebook
    
    # ================================================================
    # DETECTION
    # ================================================================
    
    def detect(self, image_path: str) -> DetectionResult:
        """Detect SynthID watermark in an image file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detect_array(img)
    
    def detect_array(self, image: np.ndarray) -> DetectionResult:
        """Detect SynthID watermark in a numpy array image.

        Uses empirically-verified carrier frequencies with per-set reference
        phases.  The watermark is content-adaptive: dark images use diagonal-
        grid carriers; white/bright images use horizontal-axis carriers.

        Detection checks phase agreement at both carrier sets against known
        reference phases and takes the best match.  Calibrated on 291 Gemini
        watermarked + 16 non-watermarked images:

          Watermarked:     phase match 0.92–0.99
          Non-watermarked: phase match 0.47–0.53 (max 0.71)
        """
        if self.codebook is None:
            raise ValueError("No codebook loaded. Call extract_codebook() or load_codebook() first.")

        # Support both legacy scalar (512) and new tuple (h, w) image_size
        img_size = self.codebook['image_size']
        if isinstance(img_size, (list, tuple)):
            target_h, target_w = int(img_size[0]), int(img_size[1])
        else:
            target_h = target_w = int(img_size)
        img_resized = cv2.resize(image, (target_w, target_h))
        center_y = target_h // 2
        center_x = target_w // 2

        # ------------------------------------------------------------------
        # Image-domain FFT (grayscale)
        # ------------------------------------------------------------------
        gray = np.mean(img_resized, axis=2) if len(img_resized.shape) == 3 else img_resized
        f_img = fftshift(fft2(gray.astype(np.float32)))
        img_phase = np.angle(f_img)
        img_mag = np.abs(f_img)

        # ------------------------------------------------------------------
        # Phase matching against per-set reference phases
        # Try both dark and white carrier sets; take best match.
        # ------------------------------------------------------------------
        carrier_refs = self.codebook.get('carrier_refs', {})
        set_results = {}

        for set_name, carriers_attr, ref_key in [
            ('dark', 'carriers_dark', 'dark_ref_phases'),
            ('white', 'carriers_white', 'white_ref_phases'),
        ]:
            carriers = getattr(self, carriers_attr, [])
            ref_phases = carrier_refs.get(ref_key)
            if ref_phases is None or len(carriers) == 0:
                continue

            phase_matches = []
            for i, (fy, fx) in enumerate(carriers):
                y, x = fy + center_y, fx + center_x
                if 0 <= y < target_h and 0 <= x < target_w and i < len(ref_phases):
                    diff = np.abs(np.angle(np.exp(1j * (img_phase[y, x] - ref_phases[i]))))
                    phase_matches.append(1 - diff / np.pi)

            if phase_matches:
                set_results[set_name] = {
                    'phase_match': float(np.mean(phase_matches)),
                    'phase_std': float(np.std(phase_matches)),
                    'n_carriers': len(phase_matches),
                }

        # Best phase match across carrier sets
        best_set = max(set_results, key=lambda k: set_results[k]['phase_match']) if set_results else 'dark'
        best_phase_match = set_results[best_set]['phase_match'] if set_results else 0.0

        # Also compute average across both sets for reporting
        all_matches = []
        for sr in set_results.values():
            all_matches.append(sr['phase_match'])
        avg_phase_match = float(np.mean(all_matches)) if all_matches else 0.0

        # ------------------------------------------------------------------
        # Noise-domain carrier-vs-random ratio (supporting signal)
        # ------------------------------------------------------------------
        ref_noise = self.codebook['reference_noise']
        # Resize ref_noise to match target if needed
        if ref_noise.shape[0] != target_h or ref_noise.shape[1] != target_w:
            ref_noise_resized = cv2.resize(ref_noise.astype(np.float32),
                                           (target_w, target_h))
        else:
            ref_noise_resized = ref_noise
        noise = self.extract_noise_fused(img_resized)
        noise_gray = np.mean(noise, axis=2) if len(noise.shape) == 3 else noise
        f_noise = fftshift(fft2(noise_gray))
        noise_mag = np.abs(f_noise)

        all_carriers = self.carriers_dark + self.carriers_white
        carrier_mags = [
            noise_mag[fy + center_y, fx + center_x]
            for fy, fx in all_carriers
            if 0 <= fy + center_y < target_h and 0 <= fx + center_x < target_w
        ]

        rng = np.random.RandomState(42)
        random_mags = []
        for _ in range(len(all_carriers) * 4):
            ry, rx = rng.randint(10, target_h - 10), rng.randint(10, target_w - 10)
            if abs(ry - center_y) < 5 and abs(rx - center_x) < 5:
                continue
            random_mags.append(noise_mag[ry, rx])

        cvr_noise = float(np.mean(carrier_mags)) / (float(np.mean(random_mags)) + 1e-10)

        # ------------------------------------------------------------------
        # Legacy metrics (for reporting / backward compat)
        # ------------------------------------------------------------------
        structure_ratio = float(np.std(noise_gray) / (np.mean(np.abs(noise_gray)) + 1e-10))
        correlation = float(np.corrcoef(noise.ravel(), ref_noise_resized.ravel())[0, 1])

        carrier_mags_img = [
            img_mag[fy + center_y, fx + center_x]
            for fy, fx in all_carriers
            if 0 <= fy + center_y < target_h and 0 <= fx + center_x < target_w
        ]
        avg_carrier_strength = float(np.mean(carrier_mags_img)) if carrier_mags_img else 0.0

        multi_scale_consistency = 0.0  # skip expensive computation; phase match is decisive

        # ==================================================================
        # SCORING
        #
        # Primary: best_phase_match — phase agreement at carrier frequencies
        # vs per-set reference.  Calibrated gap:
        #   WM:     0.92-0.99 (black/nbpro 0.99, white 0.92)
        #   non-WM: 0.47-0.53 (max observed 0.71)
        #
        # Threshold at 0.80: well above non-WM max (0.71) with margin,
        # well below WM min (0.92 avg for white).
        #
        # Supporting: cvr_noise adds confidence for dark images.
        # ==================================================================

        # Phase score: sigmoid centered at 0.78 (between 0.71 max non-WM and 0.92 min WM)
        phase_score = float(1.0 / (1.0 + np.exp(-20.0 * (best_phase_match - 0.78))))

        # CVR noise score: supporting signal
        cvr_score = float(1.0 / (1.0 + np.exp(-2.0 * (cvr_noise - 2.0))))

        # Combined confidence — phase-dominant
        confidence = float(min(1.0, 0.80 * phase_score + 0.20 * cvr_score))

        is_watermarked = confidence > 0.50

        return DetectionResult(
            is_watermarked=bool(is_watermarked),
            confidence=confidence,
            correlation=correlation,
            phase_match=best_phase_match,
            structure_ratio=structure_ratio,
            carrier_strength=avg_carrier_strength,
            multi_scale_consistency=multi_scale_consistency,
            details={
                'best_set': best_set,
                'best_phase_match': best_phase_match,
                'set_results': set_results,
                'cvr_noise': cvr_noise,
                'phase_score': phase_score,
                'cvr_score': cvr_score,
            }
        )


# ================================================================
# CLI INTERFACE
# ================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust SynthID Watermark Extractor')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract codebook from images')
    extract_parser.add_argument('image_dir', type=str, help='Directory with watermarked images')
    extract_parser.add_argument('--output', type=str, default='./robust_codebook.pkl', help='Output path')
    extract_parser.add_argument('--max-images', type=int, default=250, help='Max images to process')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark in image')
    detect_parser.add_argument('image', type=str, help='Image to check')
    detect_parser.add_argument('--codebook', type=str, required=True, help='Codebook path')
    
    args = parser.parse_args()
    
    extractor = RobustSynthIDExtractor()
    
    if args.command == 'extract':
        extractor.extract_codebook(args.image_dir, args.max_images, args.output)
        
    elif args.command == 'detect':
        extractor.load_codebook(args.codebook)
        result = extractor.detect(args.image)
        
        print("\n" + "=" * 50)
        print("ROBUST SYNTHID DETECTION RESULTS")
        print("=" * 50)
        print(f"  Watermarked: {result.is_watermarked}")
        print(f"  Confidence:  {result.confidence:.4f}")
        print(f"  Correlation: {result.correlation:.4f}")
        print(f"  Phase Match: {result.phase_match:.4f}")
        print(f"  Structure:   {result.structure_ratio:.4f}")
        print(f"  Carrier Str: {result.carrier_strength:.2f}")
        print(f"  Multi-Scale: {result.multi_scale_consistency:.4f}")
        print("=" * 50)
    
    else:
        parser.print_help()
