"""
SynthID Bypass - Non-DL Watermark Removal

Implements watermark removal techniques inspired by diffusion-based bypass,
but using pure signal processing approaches (no deep learning required).

Key techniques:
1. Noise replacement (mimics low-denoise regeneration)
2. Frequency domain disruption (phase scrambling at carrier frequencies)
3. JPEG degradation (quality cycling, chroma subsampling)
4. Bit manipulation (LSB randomization, bit-depth reduction)
5. Structure-preserving reconstruction (edge-guided blending)

Based on insights from:
- Synthid-Bypass ComfyUI workflow (diffusion regeneration approach)
- SynthID-Image paper (watermark embedding mechanism)
"""

import os
import sys
import io
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage

# Ensure same-directory modules are importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class BypassResult:
    """Result of watermark bypass."""
    success: bool
    cleaned_image: np.ndarray
    psnr: float
    ssim: float
    detection_before: Optional[Dict]
    detection_after: Optional[Dict]
    stages_applied: List[str]
    details: Dict


class SynthIDBypass:
    """
    Remove SynthID watermarks using signal processing techniques.
    
    This class implements a multi-stage bypass pipeline that mimics
    the effect of diffusion-based regeneration without requiring
    deep learning models.
    """
    
    # SynthID carrier frequencies extracted from 288 Gemini reference images.
    # Carriers sit on a (48, 88) grid in frequency space at 512x512 resolution.
    KNOWN_CARRIERS = [
        # Axis-aligned (strong)
        (48, 0), (-48, 0),
        (96, 0), (-96, 0),
        (192, 0), (-192, 0),
        (210, 0), (-210, 0),
        (238, 0), (-238, 0),
        (0, 88), (0, -88),
        (0, 176), (0, -176),
        (0, 192), (0, -192),
        # Off-axis (grid intersections)
        (48, 88), (-48, -88),
        (48, -88), (-48, 88),
        (96, 88), (-96, -88),
        (96, -88), (-96, 88),
        (96, 176), (-96, -176),
        (96, -176), (-96, 176),
    ]
    
    def __init__(
        self,
        iterations: int = 3,
        extractor=None
    ):
        """
        Initialize the bypass.
        
        Args:
            iterations: Number of bypass passes
            extractor: Optional RobustSynthIDExtractor for verification
        """
        self.iterations = iterations
        self.extractor = extractor
    
    # ================================================================
    # STAGE 1: NOISE REPLACEMENT
    # Mimics low-denoise regeneration - replace watermark noise with new noise
    # ================================================================
    
    def add_calibrated_noise(
        self,
        image: np.ndarray,
        sigma: float = 3.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add calibrated Gaussian noise to disrupt watermark.
        
        The noise level is carefully chosen to be strong enough to
        disrupt the watermark but weak enough to preserve image quality.
        """
        if seed is not None:
            np.random.seed(seed)
        
        noise = np.random.normal(0, sigma / 255.0, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def denoise_bilateral(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """Edge-preserving bilateral filter denoising."""
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)

        return denoised.astype(np.float32) / 255.0
    
    def denoise_nlm(
        self,
        image: np.ndarray,
        h: float = 5,
        template_size: int = 7,
        search_size: int = 21
    ) -> np.ndarray:
        """Non-local means denoising."""
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_uint8, None, h, h, template_size, search_size
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                img_uint8, None, h, template_size, search_size
            )
        
        return denoised.astype(np.float32) / 255.0
    
    def noise_replacement_pass(
        self,
        image: np.ndarray,
        noise_sigma: float = 5.0,
        denoise_strength: float = 8.0
    ) -> np.ndarray:
        """
        Single pass of noise replacement.
        
        This mimics the diffusion model's low-denoise regeneration:
        1. Add noise to disrupt existing patterns
        2. Denoise to recover structure (but with different noise)
        """
        # Add calibrated noise
        noisy = self.add_calibrated_noise(image, sigma=noise_sigma)
        
        # Denoise with edge-preserving filter
        denoised = self.denoise_bilateral(noisy, d=9, sigma_color=denoise_strength * 10, sigma_space=75)
        
        # Blend with original to preserve some structure
        result = denoised * 0.7 + image * 0.3
        
        return result
    
    def apply_noise_replacement(
        self,
        image: np.ndarray,
        passes: int = 2,
        noise_sigma: float = 5.0
    ) -> np.ndarray:
        """
        Apply multiple noise replacement passes.
        
        Similar to multiple KSampler passes in the diffusion workflow.
        """
        current = image.copy()

        for i in range(passes):
            # Decrease noise sigma slightly each pass, clamp to avoid negative
            sigma = noise_sigma * max(0, 1 - i * 0.2)
            if sigma <= 0:
                break
            current = self.noise_replacement_pass(current, noise_sigma=sigma)

        return current
    
    # ================================================================
    # STAGE 2: FREQUENCY DOMAIN DISRUPTION
    # Scramble phases at known carrier frequencies
    # ================================================================
    
    def scramble_carrier_phases(
        self,
        image: np.ndarray,
        carriers: Optional[List[Tuple[int, int]]] = None,
        scramble_radius: int = 3,
        scramble_strength: float = 0.8
    ) -> np.ndarray:
        """
        Randomize phases at carrier frequencies to break watermark coherence.
        """
        if carriers is None:
            carriers = self.KNOWN_CARRIERS
        
        img_f = image.astype(np.float32)
        h, w = img_f.shape[:2]
        center = (h // 2, w // 2)
        
        # Scale carriers to image size (carriers are for 512px)
        scale_y = h / 512
        scale_x = w / 512
        
        if len(img_f.shape) == 3:
            result = np.zeros_like(img_f)
            for c in range(img_f.shape[2]):
                result[:, :, c] = self._scramble_channel(
                    img_f[:, :, c], carriers, center, scale_y, scale_x,
                    scramble_radius, scramble_strength
                )
        else:
            result = self._scramble_channel(
                img_f, carriers, center, scale_y, scale_x,
                scramble_radius, scramble_strength
            )
        
        return result
    
    def _scramble_channel(
        self,
        channel: np.ndarray,
        carriers: List[Tuple[int, int]],
        center: Tuple[int, int],
        scale_y: float,
        scale_x: float,
        radius: int,
        strength: float
    ) -> np.ndarray:
        """Scramble phases in a single channel."""
        f = fftshift(fft2(channel))
        h, w = channel.shape
        
        for freq_y, freq_x in carriers:
            # Scale to image size
            y = int(freq_y * scale_y) + center[0]
            x = int(freq_x * scale_x) + center[1]
            
            # Scramble region around carrier
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        # Randomize phase while preserving magnitude
                        mag = np.abs(f[ny, nx])
                        random_phase = np.random.uniform(0, 2 * np.pi)
                        original_phase = np.angle(f[ny, nx])
                        new_phase = original_phase * (1 - strength) + random_phase * strength
                        f[ny, nx] = mag * np.exp(1j * new_phase)
                        
                        # Also scramble conjugate
                        cny, cnx = h - ny, w - nx
                        if 0 <= cny < h and 0 <= cnx < w:
                            cmag = np.abs(f[cny, cnx])
                            f[cny, cnx] = cmag * np.exp(-1j * new_phase)
        
        result = np.real(ifft2(ifftshift(f)))
        return np.clip(result, 0, 1)
    
    def inject_bandpass_noise(
        self,
        image: np.ndarray,
        freq_range: Tuple[float, float] = (0.02, 0.15),
        noise_strength: float = 0.02
    ) -> np.ndarray:
        """
        Inject noise in specific frequency bands where watermark lives.
        """
        img_f = image.astype(np.float32)
        h, w = img_f.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Create bandpass mask
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
        norm_dist = dist / max_dist
        
        bandpass = ((norm_dist > freq_range[0]) & (norm_dist < freq_range[1])).astype(np.float32)
        
        if len(img_f.shape) == 3:
            result = np.zeros_like(img_f)
            for c in range(img_f.shape[2]):
                result[:, :, c] = self._inject_bandpass_channel(
                    img_f[:, :, c], bandpass, noise_strength
                )
        else:
            result = self._inject_bandpass_channel(img_f, bandpass, noise_strength)
        
        return result
    
    def _inject_bandpass_channel(
        self,
        channel: np.ndarray,
        bandpass: np.ndarray,
        noise_strength: float
    ) -> np.ndarray:
        """Inject bandpass noise in a single channel."""
        f = fftshift(fft2(channel))
        
        # Generate random phase noise
        phase_noise = np.random.uniform(0, 2 * np.pi, f.shape)
        
        # Add noise only in bandpass region
        noise_complex = noise_strength * np.exp(1j * phase_noise) * bandpass
        f_noisy = f + noise_complex * np.abs(f).mean()
        
        result = np.real(ifft2(ifftshift(f_noisy)))
        return np.clip(result, 0, 1)
    
    # ================================================================
    # STAGE 3: JPEG DEGRADATION
    # JPEG compression breaks watermark coherence
    # ================================================================
    
    def jpeg_compress(
        self,
        image: np.ndarray,
        quality: int = 85
    ) -> np.ndarray:
        """Apply JPEG compression/decompression cycle."""
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to PIL for JPEG encoding
        if len(image.shape) == 3:
            pil_img = Image.fromarray(img_uint8, mode='RGB')
        else:
            pil_img = Image.fromarray(img_uint8, mode='L')
        
        # Compress to JPEG in memory
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Decompress
        pil_img = Image.open(buffer)
        result = np.array(pil_img).astype(np.float32) / 255.0
        
        return result
    
    def jpeg_quality_cycle(
        self,
        image: np.ndarray,
        qualities: List[int] = [70, 80, 92]
    ) -> np.ndarray:
        """
        Apply multiple JPEG compression cycles at varying qualities.
        
        This disrupts the watermark through quantization artifacts
        while the varying qualities prevent adaptation.
        """
        current = image.copy()
        
        for q in qualities:
            current = self.jpeg_compress(current, quality=q)
        
        return current
    
    def chroma_subsample(
        self,
        image: np.ndarray,
        factor: int = 2
    ) -> np.ndarray:
        """
        Subsample and upsample chroma channels.
        
        This is similar to what JPEG does but more aggressive,
        disrupting watermark in color channels.
        """
        if len(image.shape) != 3:
            return image
        
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Subsample chroma
        h, w = cr.shape
        cr_small = cv2.resize(cr, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
        cb_small = cv2.resize(cb, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
        
        # Upsample back
        cr_up = cv2.resize(cr_small, (w, h), interpolation=cv2.INTER_LINEAR)
        cb_up = cv2.resize(cb_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Merge and convert back
        ycrcb_new = cv2.merge([y, cr_up, cb_up])
        rgb = cv2.cvtColor(ycrcb_new, cv2.COLOR_YCrCb2RGB)
        
        return rgb.astype(np.float32) / 255.0
    
    # ================================================================
    # STAGE 4: BIT MANIPULATION
    # Modify LSBs where watermark often resides
    # ================================================================
    
    def randomize_lsb(
        self,
        image: np.ndarray,
        n_bits: int = 2,
        probability: float = 0.5
    ) -> np.ndarray:
        """
        Randomize least significant bits.
        
        LSBs often carry watermark info; randomizing them disrupts
        the watermark while having minimal visual impact.
        """
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Create mask for bits to randomize
        mask = np.uint8((1 << n_bits) - 1)  # e.g., n_bits=2 -> mask=0b11
        inv_mask = np.uint8(~mask)  # Properly handle uint8 inversion
        
        # Random bits
        random_bits = np.random.randint(0, int(mask) + 1, img_uint8.shape, dtype=np.uint8)
        
        # Random selection of pixels to modify
        modify_mask = np.random.random(img_uint8.shape) < probability
        
        # Clear LSBs and add random bits
        result = img_uint8.copy()
        result[modify_mask] = (result[modify_mask] & inv_mask) | random_bits[modify_mask]
        
        return result.astype(np.float32) / 255.0
    
    def reduce_bit_depth(
        self,
        image: np.ndarray,
        bits: int = 6
    ) -> np.ndarray:
        """
        Reduce and expand bit depth.
        
        Quantizes to fewer bits then expands, effectively
        removing fine-grained watermark patterns.
        """
        levels = 2 ** bits
        
        # Quantize
        quantized = np.round(image * (levels - 1))
        
        # Expand back
        result = quantized / (levels - 1)
        
        return result.astype(np.float32)
    
    def color_jitter(
        self,
        image: np.ndarray,
        brightness: float = 0.02,
        contrast: float = 0.02,
        saturation: float = 0.02
    ) -> np.ndarray:
        """
        Apply small random color adjustments.
        
        Subtle color variations break watermark coherence.
        """
        result = image.copy()
        
        # Brightness
        b_factor = 1 + np.random.uniform(-brightness, brightness)
        result = result * b_factor
        
        # Contrast
        c_factor = 1 + np.random.uniform(-contrast, contrast)
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = (result - mean) * c_factor + mean
        
        # Saturation (for color images)
        if len(result.shape) == 3:
            s_factor = 1 + np.random.uniform(-saturation, saturation)
            gray = np.mean(result, axis=2, keepdims=True)
            result = gray + (result - gray) * s_factor
        
        return np.clip(result, 0, 1)
    
    # ================================================================
    # STAGE 5: STRUCTURE-PRESERVING RECONSTRUCTION
    # Use edges to guide reconstruction like ControlNet
    # ================================================================
    
    def extract_structure(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract structural information (edges and texture).
        
        This is similar to how Canny edges are used in ControlNet
        to preserve structure during regeneration.
        """
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8
        
        # Edge detection (like Canny in ControlNet)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        
        # Gradient magnitude (texture measure)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient = gradient / (gradient.max() + 1e-10)
        
        return edges, gradient.astype(np.float32)
    
    def guided_filter(
        self,
        image: np.ndarray,
        guide: np.ndarray,
        radius: int = 8,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Apply guided filter for edge-preserving smoothing."""
        def box_filter(img, r):
            return ndimage.uniform_filter(img, size=r * 2 + 1)
        
        if len(image.shape) == 3 and len(guide.shape) == 2:
            guide = np.stack([guide] * image.shape[2], axis=2)
        
        if len(image.shape) == 2:
            mean_i = box_filter(guide, radius)
            mean_p = box_filter(image, radius)
            mean_ip = box_filter(guide * image, radius)
            cov_ip = mean_ip - mean_i * mean_p
            
            mean_ii = box_filter(guide * guide, radius)
            var_i = mean_ii - mean_i * mean_i
            
            a = cov_ip / (var_i + epsilon)
            b = mean_p - a * mean_i
            
            mean_a = box_filter(a, radius)
            mean_b = box_filter(b, radius)
            
            return mean_a * guide + mean_b
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self.guided_filter(
                    image[:, :, c], guide[:, :, c], radius, epsilon
                )
            return result
    
    def reconstruct_with_structure(
        self,
        processed: np.ndarray,
        original: np.ndarray,
        edges: np.ndarray,
        blend_factor: float = 0.2
    ) -> np.ndarray:
        """
        Blend processed image with original guided by structure.
        
        Preserves edges and structure while keeping watermark disruption.
        """
        # Use original as guide for edge-preserving filtering
        filtered = self.guided_filter(processed, original, radius=5)
        
        # Stronger preservation near edges
        if len(original.shape) == 3:
            edge_map = np.stack([edges] * 3, axis=2)
        else:
            edge_map = edges
        
        # Near edges: blend more with original
        # Away from edges: keep more of processed
        result = filtered * (1 - edge_map * 0.3) + original * (edge_map * 0.3)
        
        # Final blend
        result = result * (1 - blend_factor) + original * blend_factor
        
        return np.clip(result, 0, 1)
    
    # ================================================================
    # QUALITY METRICS
    # ================================================================
    
    def compute_psnr(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        return float(10 * np.log10(1.0 / mse))
    
    def compute_ssim(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Compute SSIM using vectorized block-based approach (no loop)."""
        img_o = original.astype(np.float64)
        img_m = modified.astype(np.float64)
        if img_o.max() > 1.5:
            img_o = img_o / 255.0
            img_m = img_m / 255.0
        
        # Convert to grayscale using luminance weights
        if img_o.ndim == 3:
            gray_o = 0.299 * img_o[:,:,0] + 0.587 * img_o[:,:,1] + 0.114 * img_o[:,:,2]
            gray_m = 0.299 * img_m[:,:,0] + 0.587 * img_m[:,:,1] + 0.114 * img_m[:,:,2]
        else:
            gray_o = img_o
            gray_m = img_m
        
        blk = 8
        rows, cols = gray_o.shape
        rc = (rows // blk) * blk
        cc = (cols // blk) * blk
        
        # Vectorized block reshape
        a = gray_o[:rc, :cc].reshape(rc // blk, blk, cc // blk, blk)
        a = a.transpose(0, 2, 1, 3).reshape(-1, blk, blk)
        b = gray_m[:rc, :cc].reshape(rc // blk, blk, cc // blk, blk)
        b = b.transpose(0, 2, 1, 3).reshape(-1, blk, blk)
        
        mu_a = a.mean(axis=(1, 2))
        mu_b = b.mean(axis=(1, 2))
        var_a = a.var(axis=(1, 2))
        var_b = b.var(axis=(1, 2))
        cov_ab = ((a - mu_a[:, None, None]) * (b - mu_b[:, None, None])).mean(axis=(1, 2))
        
        k1_sq = 0.0001  # (0.01)^2
        k2_sq = 0.0009  # (0.03)^2
        
        num = (2.0 * mu_a * mu_b + k1_sq) * (2.0 * cov_ab + k2_sq)
        den = (mu_a * mu_a + mu_b * mu_b + k1_sq) * (var_a + var_b + k2_sq)
        
        return float(np.mean(num / den))
    
    # ================================================================
    # MAIN BYPASS PIPELINE
    # ================================================================
    
    def bypass_simple(
        self,
        image: np.ndarray,
        jpeg_quality: int = 50,
        verify: bool = True
    ) -> BypassResult:
        """
        Simple, effective bypass using just JPEG compression.
        
        Testing showed JPEG Q50 is the most effective single technique,
        achieving ~11% phase match reduction with excellent quality (PSNR 37dB).
        Other techniques (noise, frequency manipulation) are less effective
        and hurt quality more than they help.
        
        Args:
            image: Input image (RGB, uint8 or float)
            jpeg_quality: JPEG quality (50 is optimal for SynthID)
            verify: Whether to verify removal with detection
            
        Returns:
            BypassResult with cleaned image and metrics
        """
        img_f = image.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0
        
        # Initial detection
        detection_before = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array((img_f * 255).astype(np.uint8))
            detection_before = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Apply JPEG compression
        cleaned = self.jpeg_compress(img_f, quality=jpeg_quality)
        
        # Compute quality metrics
        psnr = self.compute_psnr(img_f, cleaned)
        ssim = self.compute_ssim(img_f, cleaned)
        
        # Final detection
        detection_after = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array((cleaned * 255).astype(np.uint8))
            detection_after = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Determine success
        success = psnr > 30
        if detection_before and detection_after:
            phase_drop = detection_before['phase_match'] - detection_after['phase_match']
            success = success and phase_drop > 0.05
        
        cleaned_uint8 = (cleaned * 255).clip(0, 255).astype(np.uint8)
        
        return BypassResult(
            success=success,
            cleaned_image=cleaned_uint8,
            psnr=psnr,
            ssim=ssim,
            detection_before=detection_before,
            detection_after=detection_after,
            stages_applied=['jpeg_compress'],
            details={'method': 'simple', 'jpeg_quality': jpeg_quality}
        )
    
    def bypass(
        self,
        image: np.ndarray,
        mode: str = 'balanced',
        verify: bool = True
    ) -> BypassResult:
        """
        Main bypass pipeline - remove SynthID watermark.
        
        Args:
            image: Input image (RGB, uint8 or float)
            mode: 'light', 'balanced', or 'aggressive'
            verify: Whether to verify removal with detection
            
        Returns:
            BypassResult with cleaned image and metrics
        """
        img_f = image.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0
        
        # Set parameters based on mode
        params = self._get_mode_params(mode)
        
        # Initial detection
        detection_before = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array((img_f * 255).astype(np.uint8))
            detection_before = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Extract structure for preservation
        edges, gradient = self.extract_structure(img_f)
        
        current = img_f.copy()
        stages_applied = []
        
        # Apply bypass iterations
        for iteration in range(params['iterations']):
            # Stage 1: Noise replacement
            if params['noise_replacement']:
                current = self.apply_noise_replacement(
                    current,
                    passes=params['noise_passes'],
                    noise_sigma=params['noise_sigma']
                )
                stages_applied.append(f'noise_replacement_{iteration}')
            
            # Stage 2: Frequency disruption
            if params['frequency_disruption']:
                current = self.scramble_carrier_phases(
                    current,
                    scramble_radius=params['scramble_radius'],
                    scramble_strength=params['scramble_strength']
                )
                current = self.inject_bandpass_noise(
                    current,
                    noise_strength=params['bandpass_noise']
                )
                stages_applied.append(f'frequency_disruption_{iteration}')
            
            # Stage 3: JPEG degradation
            if params['jpeg_degradation']:
                current = self.jpeg_quality_cycle(current, params['jpeg_qualities'])
                if params['chroma_subsample']:
                    current = self.chroma_subsample(current)
                stages_applied.append(f'jpeg_degradation_{iteration}')
            
            # Stage 4: Bit manipulation
            if params['bit_manipulation']:
                current = self.randomize_lsb(
                    current, n_bits=params['lsb_bits'],
                    probability=params['lsb_probability']
                )
                current = self.color_jitter(current)
                stages_applied.append(f'bit_manipulation_{iteration}')
            
            # Stage 5: Structure preservation
            current = self.reconstruct_with_structure(
                current, img_f, edges,
                blend_factor=params['structure_blend']
            )
            stages_applied.append(f'structure_preservation_{iteration}')
        
        # Compute quality metrics
        psnr = self.compute_psnr(img_f, current)
        ssim = self.compute_ssim(img_f, current)
        
        # Final detection
        detection_after = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array((current * 255).astype(np.uint8))
            detection_after = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Determine success
        success = psnr > 28 and ssim > 0.9
        if detection_before and detection_after:
            phase_drop = detection_before['phase_match'] - detection_after['phase_match']
            success = success and (phase_drop > 0.05 or not detection_after['is_watermarked'])
        
        # Convert to uint8
        cleaned_uint8 = (current * 255).clip(0, 255).astype(np.uint8)
        
        return BypassResult(
            success=success,
            cleaned_image=cleaned_uint8,
            psnr=psnr,
            ssim=ssim,
            detection_before=detection_before,
            detection_after=detection_after,
            stages_applied=stages_applied,
            details={'mode': mode, 'params': params}
        )
    
    def _get_mode_params(self, mode: str) -> Dict:
        """Get parameters for each mode."""
        if mode == 'light':
            return {
                'iterations': 1,
                'noise_replacement': True,
                'noise_passes': 1,
                'noise_sigma': 3.0,
                'frequency_disruption': True,
                'scramble_radius': 2,
                'scramble_strength': 0.5,
                'bandpass_noise': 0.01,
                'jpeg_degradation': True,
                'jpeg_qualities': [88, 95],
                'chroma_subsample': False,
                'bit_manipulation': False,
                'lsb_bits': 1,
                'lsb_probability': 0.3,
                'structure_blend': 0.3
            }
        elif mode == 'aggressive':
            return {
                'iterations': 3,
                'noise_replacement': True,
                'noise_passes': 2,
                'noise_sigma': 8.0,
                'frequency_disruption': True,
                'scramble_radius': 5,
                'scramble_strength': 0.9,
                'bandpass_noise': 0.03,
                'jpeg_degradation': True,
                'jpeg_qualities': [65, 75, 88],
                'chroma_subsample': True,
                'bit_manipulation': True,
                'lsb_bits': 2,
                'lsb_probability': 0.6,
                'structure_blend': 0.15
            }
        elif mode == 'maximum':
            # Maximum bypass - prioritizes watermark removal over quality
            # Based on empirical testing: JPEG Q50 + Noise(25) are most effective
            return {
                'iterations': 3,
                'noise_replacement': True,
                'noise_passes': 3,
                'noise_sigma': 25.0,  # Heavy noise injection
                'frequency_disruption': True,
                'scramble_radius': 8,
                'scramble_strength': 1.0,  # Full phase randomization
                'bandpass_noise': 0.05,
                'jpeg_degradation': True,
                'jpeg_qualities': [50, 60, 75],  # Low quality JPEG
                'chroma_subsample': True,
                'bit_manipulation': True,
                'lsb_bits': 3,  # More LSB randomization
                'lsb_probability': 0.8,
                'structure_blend': 0.05  # Minimal blending to avoid restoring watermark
            }
        else:  # balanced
            return {
                'iterations': 2,
                'noise_replacement': True,
                'noise_passes': 2,
                'noise_sigma': 5.0,
                'frequency_disruption': True,
                'scramble_radius': 3,
                'scramble_strength': 0.7,
                'bandpass_noise': 0.02,
                'jpeg_degradation': True,
                'jpeg_qualities': [75, 85, 92],
                'chroma_subsample': True,
                'bit_manipulation': True,
                'lsb_bits': 2,
                'lsb_probability': 0.5,
                'structure_blend': 0.2
            }
    
    def bypass_file(
        self,
        input_path: str,
        output_path: str,
        mode: str = 'balanced',
        verify: bool = True
    ) -> BypassResult:
        """
        Bypass watermark in image file and save result.
        """
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = self.bypass(img_rgb, mode=mode, verify=verify)
        
        # Save result
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR))
        
        return result

    # ================================================================
    # V2: COMBINED WORST-CASE BYPASS PIPELINE
    # Targets SynthID's documented weakness against stacked
    # multi-category transforms (combination worst TPR ~84%)
    # ================================================================

    def _spatial_disruption(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Spatial transforms — SynthID's weakest category (52% TPR worst-case).
        
        Applies random affine + crop-resize + perspective warp to break
        spatial coherence of the watermark pattern.
        """
        h, w = image.shape[:2]
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Random affine: rotation, scale, translation (very subtle)
        angle = np.random.uniform(-0.3, 0.3) * strength
        scale = 1.0 + np.random.uniform(-0.005, 0.005) * strength
        tx = np.random.uniform(-1, 1) * strength
        ty = np.random.uniform(-1, 1) * strength
        
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        result = cv2.warpAffine(img_uint8, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT_101)
        
        # Random crop and resize back (0.3-0.8%)
        crop_frac = 0.003 + 0.005 * strength
        cx = int(w * crop_frac * np.random.uniform(0.3, 1.0))
        cy = int(h * crop_frac * np.random.uniform(0.3, 1.0))
        cx = max(1, cx)
        cy = max(1, cy)
        cropped = result[cy:h-cy, cx:w-cx]
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Subtle perspective warp
        if strength > 0.5:
            offset = max(1, int(1.5 * strength))
            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst_pts = np.float32([
                [np.random.randint(0, offset+1), np.random.randint(0, offset+1)],
                [w - np.random.randint(0, offset+1), np.random.randint(0, offset+1)],
                [np.random.randint(0, offset+1), h - np.random.randint(0, offset+1)],
                [w - np.random.randint(0, offset+1), h - np.random.randint(0, offset+1)]
            ])
            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            result = cv2.warpPerspective(result, M_persp, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REFLECT_101)
        
        return result.astype(np.float32) / 255.0

    def _quality_degradation(
        self,
        image: np.ndarray,
        jpeg_quality: int = 40,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Quality transforms: JPEG/WebP cycling + resize cycling.
        
        Forces requantization across different codec bases (DCT vs wavelet)
        to destroy watermark coherence that survives any single codec.
        """
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        
        # Step 1: JPEG compression
        pil_img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=jpeg_quality)
        buf.seek(0)
        result = np.array(Image.open(buf).convert('RGB'))
        
        # Step 2: WebP compression (different transform basis than JPEG DCT)
        webp_q = max(30, jpeg_quality - 5)
        pil_img2 = Image.fromarray(result)
        buf2 = io.BytesIO()
        pil_img2.save(buf2, format='WEBP', quality=webp_q)
        buf2.seek(0)
        result = np.array(Image.open(buf2).convert('RGB'))
        
        # Step 3: Second JPEG at slightly different quality
        pil_img3 = Image.fromarray(result)
        buf3 = io.BytesIO()
        pil_img3.save(buf3, format='JPEG', quality=jpeg_quality + 15)
        buf3.seek(0)
        result = np.array(Image.open(buf3).convert('RGB'))
        
        # Step 4: Downscale + upscale (destroys sub-pixel watermark info)
        if strength > 0.3:
            down_factor = 0.875 - 0.05 * strength  # 82-87% downscale
            small_h = max(64, int(h * down_factor))
            small_w = max(64, int(w * down_factor))
            small = cv2.resize(result, (small_w, small_h), interpolation=cv2.INTER_AREA)
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return result.astype(np.float32) / 255.0

    def _noise_disruption(
        self,
        image: np.ndarray,
        sigma: float = 10.0,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Noise injection + denoising: replaces watermark noise with random noise.
        
        Uses both Gaussian and Poisson noise (different distributions)
        followed by edge-preserving denoising.
        """
        result = image.copy()
        
        # Gaussian noise (moderate: enough to displace watermark bits)
        noise_std = (sigma / 255.0) * strength
        noise = np.random.normal(0, noise_std, result.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 1)
        
        # Poisson noise (shot noise — different distribution, subtle)
        if strength > 0.5:
            lam = 200 / strength  # Higher lambda = less noise
            noisy = np.random.poisson(np.maximum(result * lam, 0)) / lam
            result = result * 0.9 + noisy.astype(np.float32) * 0.1
            result = np.clip(result, 0, 1)
        
        # Edge-preserving bilateral denoising (moderate)
        img_uint8 = (result * 255).clip(0, 255).astype(np.uint8)
        d = 5
        sigma_c = 30 + 15 * strength
        sigma_s = 30 + 15 * strength
        denoised = cv2.bilateralFilter(img_uint8, d, sigma_c, sigma_s)
        
        # NLM denoising for a second pass (light)
        if strength > 0.7:
            h_param = 3 + 3 * strength
            denoised = cv2.fastNlMeansDenoisingColored(
                denoised, None, h_param, h_param, 7, 21
            )
        
        return denoised.astype(np.float32) / 255.0

    def _color_disruption(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Color channel manipulation to destroy per-channel watermark coherence.
        
        SynthID embeds differently per channel (G most, then R, then B).
        Disrupting color space breaks cross-channel correlations.
        """
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        
        # YCrCb chroma subsampling (like aggressive JPEG chroma)
        ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
        h, w = ycrcb.shape[:2]
        factor = 2 + int(strength)
        # Subsample chroma channels
        for c in [1, 2]:
            ch = ycrcb[:, :, c]
            small = cv2.resize(ch, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
            ycrcb[:, :, c] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        # Hue shift in HSV space
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = np.random.uniform(-3, 3) * strength
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        # Saturation adjustment
        sat_factor = 1.0 + np.random.uniform(-0.05, 0.05) * strength
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Gamma correction
        gamma = 1.0 + np.random.uniform(-0.06, 0.06) * strength
        result_f = (result.astype(np.float32) / 255.0) ** gamma
        
        return np.clip(result_f, 0, 1)

    def _overlay_disruption(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Overlay-type disruptions: artifact injection and dithering.
        
        These add structured patterns that interfere with watermark detection.
        """
        result = image.copy()
        
        # JPEG artifact overlay: encode at very low quality, compute diff
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=max(10, int(15 / strength)))
        buf.seek(0)
        heavy_jpeg = np.array(Image.open(buf).convert('RGB')).astype(np.float32) / 255.0
        
        # Add a fraction of the JPEG artifacts
        artifacts = heavy_jpeg - image
        artifact_strength = 0.08 + 0.07 * strength  # 8-15% of artifacts
        result = result + artifacts * artifact_strength
        result = np.clip(result, 0, 1)
        
        # Floyd-Steinberg-style dithering then smoothing
        if strength > 0.4:
            n_levels = max(32, int(64 / strength))
            quantized = np.round(result * (n_levels - 1)) / (n_levels - 1)
            # Smooth the quantized image to remove banding
            q_uint8 = (quantized * 255).clip(0, 255).astype(np.uint8)
            smoothed = cv2.GaussianBlur(q_uint8, (3, 3), 0.5)
            smoothed_f = smoothed.astype(np.float32) / 255.0
            # Blend: mostly smoothed, a bit of original for detail
            blend = 0.15 + 0.1 * strength
            result = smoothed_f * (1 - blend) + result * blend
            result = np.clip(result, 0, 1)
        
        return result

    def _final_reconstruction(
        self,
        processed: np.ndarray,
        original: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Final reconstruction step: restore detail while keeping disruption.
        
        Light edge-aware smoothing + selective sharpening.
        Does NOT use original for content guidance to avoid re-introducing watermark.
        """
        proc_uint8 = (processed * 255).clip(0, 255).astype(np.uint8)
        
        # Edge-aware bilateral filter to smooth processing artifacts
        result = cv2.bilateralFilter(proc_uint8, 5, 25, 25)
        
        # Selective sharpening to restore detail lost during processing
        sharp_amount = 0.25 + 0.15 * (1.0 - strength)
        blurred = cv2.GaussianBlur(result, (3, 3), 0.8)
        sharpened = cv2.addWeighted(result, 1.0 + sharp_amount, blurred, -sharp_amount, 0)
        
        return sharpened.astype(np.float32) / 255.0

    def bypass_v2(
        self,
        image: np.ndarray,
        strength: str = 'aggressive',
        iterations: int = 2,
        verify: bool = True
    ) -> BypassResult:
        """
        V2 Combined Worst-Case Bypass Pipeline.
        
        Stacks transforms from 6 DIFFERENT categories to maximize
        the attack surface against SynthID's robustness training.
        Per the SynthID paper (Table 1), combination worst-case
        drops TPR to ~84% (vs 99%+ for individual categories).
        
        Args:
            image: Input image (RGB, uint8 or float)
            strength: 'moderate', 'aggressive', or 'maximum'
            iterations: Number of full pipeline passes (2-3 recommended)
            verify: Whether to verify removal with detection
            
        Returns:
            BypassResult with cleaned image and metrics
        """
        img_f = image.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0
        
        # Strength parameters — single pass with appropriate strength
        strength_map = {
            'moderate': {'base': 0.5, 'jpeg_q': 60, 'noise_sigma': 6},
            'aggressive': {'base': 0.85, 'jpeg_q': 45, 'noise_sigma': 10},
            'maximum': {'base': 1.0, 'jpeg_q': 35, 'noise_sigma': 15},
        }
        params = strength_map.get(strength, strength_map['aggressive'])
        
        # Initial detection
        detection_before = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array((img_f * 255).astype(np.uint8))
            detection_before = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        current = img_f.copy()
        stages_applied = []
        s = params['base']

        for iteration in range(iterations):
            # Diminish strength slightly on subsequent iterations
            iter_s = s * max(0.5, 1.0 - iteration * 0.15)
            iter_jpeg_q = min(95, params['jpeg_q'] + iteration * 5)

            # Pass through transform categories
            # Each attacks a different dimension of the watermark embedding

            # Stage 1: Spatial disruption — only in 'maximum' mode
            # (causes significant pixel misalignment affecting SSIM, but
            #  targets SynthID's weakest category at 52% TPR worst-case)
            if strength == 'maximum':
                current = self._spatial_disruption(current, strength=iter_s)
                stages_applied.append(f'spatial_{iteration}')

            # Stage 2: Quality degradation (JPEG/WebP/resize cycling)
            current = self._quality_degradation(
                current, jpeg_quality=iter_jpeg_q, strength=iter_s
            )
            stages_applied.append(f'quality_{iteration}')

            # Stage 3: Noise injection + denoising
            current = self._noise_disruption(
                current, sigma=params['noise_sigma'], strength=iter_s
            )
            stages_applied.append(f'noise_{iteration}')

            # Stage 4: Color manipulation
            current = self._color_disruption(current, strength=iter_s)
            stages_applied.append(f'color_{iteration}')

            # Stage 5: Overlay disruption
            current = self._overlay_disruption(current, strength=iter_s)
            stages_applied.append(f'overlay_{iteration}')

        # Clamp output to valid [0,1] range
        current = np.clip(current, 0, 1)

        # Quantize to uint8 for consistent quality metrics
        cleaned_uint8 = (current * 255).clip(0, 255).astype(np.uint8)
        original_uint8 = (img_f * 255).clip(0, 255).astype(np.uint8)
        cleaned_qf = cleaned_uint8.astype(np.float64) / 255.0
        original_qf = original_uint8.astype(np.float64) / 255.0
        
        # PSNR
        _mse = np.mean((original_qf - cleaned_qf) ** 2)
        psnr = float('inf') if _mse == 0 else float(10 * np.log10(1.0 / _mse))
        
        # Inline SSIM computation (avoids Python 3.14 method dispatch issue)
        _go = 0.299 * original_qf[:,:,0] + 0.587 * original_qf[:,:,1] + 0.114 * original_qf[:,:,2]
        _gm = 0.299 * cleaned_qf[:,:,0] + 0.587 * cleaned_qf[:,:,1] + 0.114 * cleaned_qf[:,:,2]
        _blk = 8
        _rc = (_go.shape[0] // _blk) * _blk
        _cc = (_go.shape[1] // _blk) * _blk
        _a = _go[:_rc, :_cc].reshape(_rc // _blk, _blk, _cc // _blk, _blk).transpose(0, 2, 1, 3).reshape(-1, _blk, _blk)
        _b = _gm[:_rc, :_cc].reshape(_rc // _blk, _blk, _cc // _blk, _blk).transpose(0, 2, 1, 3).reshape(-1, _blk, _blk)
        _ma = _a.mean(axis=(1, 2))
        _mb = _b.mean(axis=(1, 2))
        _va = _a.var(axis=(1, 2))
        _vb = _b.var(axis=(1, 2))
        _cv = ((_a - _ma[:, None, None]) * (_b - _mb[:, None, None])).mean(axis=(1, 2))
        _c1 = 0.0001
        _c2 = 0.0009
        _num = (2.0 * _ma * _mb + _c1) * (2.0 * _cv + _c2)
        _den = (_ma * _ma + _mb * _mb + _c1) * (_va + _vb + _c2)
        ssim = float(np.mean(_num / _den))
        
        # Final detection
        detection_after = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array(cleaned_uint8)
            detection_after = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Determine success
        # Rely on PSNR > 28 dB as primary quality gate; SSIM is computed
        # for reporting but heavy multi-pass transforms can depress it below
        # the 0.90 threshold even when visual quality is acceptable.
        success = psnr > 28
        if detection_before and detection_after:
            conf_drop = detection_before['confidence'] - detection_after['confidence']
            success = success and (conf_drop > 0.15 or not detection_after['is_watermarked'])
        
        return BypassResult(
            success=success,
            cleaned_image=cleaned_uint8,
            psnr=psnr,
            ssim=ssim,
            detection_before=detection_before,
            detection_after=detection_after,
            stages_applied=stages_applied,
            details={
                'method': 'combined_worst_case_v2',
                'strength': strength,
                'iterations': iterations,
                'params': params
            }
        )

    def bypass_v2_file(
        self,
        input_path: str,
        output_path: str,
        strength: str = 'aggressive',
        iterations: int = None,
        verify: bool = True
    ) -> BypassResult:
        """Bypass watermark in image file using v2 pipeline."""
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = self.bypass_v2(img_rgb, strength=strength,
                                iterations=iterations, verify=verify)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR))
        
        return result
    
    def bypass_v3(
        self,
        image: np.ndarray,
        codebook: 'SpectralCodebook',
        strength: str = 'moderate',
        passes: int = 0,
        verify: bool = True,
    ) -> BypassResult:
        """
        V3 Spectral Bypass — multi-resolution codebook subtraction.

        Automatically selects the best-matching profile from the codebook
        for the input image's resolution.  If an exact resolution match
        exists, operates entirely in the FFT domain (fast path).
        Otherwise, constructs the watermark in the spatial domain at the
        profile's native resolution, resizes to the target, and subtracts.
        """
        if passes <= 0:
            passes = {'gentle': 1, 'moderate': 2, 'aggressive': 3,
                      'maximum': 3}.get(strength, 2)

        original_uint8 = (np.clip(image, 0, 255).astype(np.uint8)
                          if image.dtype != np.uint8 else image.copy())

        if image.dtype == np.uint8:
            work = image.astype(np.float64)
        elif np.max(image) > 1.5:
            work = image.astype(np.float64)
        else:
            work = image.astype(np.float64) * 255.0

        h, w = work.shape[:2]
        avg_luminance = float(np.mean(work)) / 255.0

        profile, (prof_h, prof_w), exact = codebook.get_profile(h, w)

        stages = []
        str_sequence = {
            'gentle':     ['gentle'],
            'moderate':   ['moderate', 'gentle'],
            'aggressive': ['aggressive', 'moderate', 'gentle'],
            'maximum':    ['maximum', 'aggressive', 'moderate'],
        }.get(strength, ['moderate'])
        while len(str_sequence) < passes:
            str_sequence.append(str_sequence[-1])
        str_sequence = str_sequence[:passes]

        for p_idx, p_str in enumerate(str_sequence):
            if exact:
                cleaned_chs = []
                for ch in range(3):
                    fft_ch = np.fft.fft2(work[:, :, ch])
                    wm_est = codebook.estimate_watermark_fft(
                        fft_ch, ch, strength=p_str,
                        image_luminance=avg_luminance,
                        profile=profile, ref_shape=(prof_h, prof_w),
                    )
                    cleaned_chs.append(
                        np.real(np.fft.ifft2(fft_ch - wm_est)))
                work = np.clip(np.stack(cleaned_chs, axis=-1), 0, 255)
            else:
                cleaned_chs = []
                for ch in range(3):
                    wm_cb = codebook.watermark_spatial(
                        ch, strength=p_str,
                        image_luminance=avg_luminance,
                        profile=profile, ref_shape=(prof_h, prof_w),
                    )
                    wm_resized = cv2.resize(
                        wm_cb, (w, h),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    cleaned_chs.append(work[:, :, ch] - wm_resized)
                work = np.clip(np.stack(cleaned_chs, axis=-1), 0, 255)

            stages.append(f'pass_{p_idx}({p_str})')

        work = cv2.GaussianBlur(work, (3, 3), 0.4)
        stages.append('anti_alias')

        cleaned_uint8 = np.clip(work, 0, 255).astype(np.uint8)

        oq = original_uint8.astype(np.float64) / 255.0
        cq = cleaned_uint8.astype(np.float64) / 255.0
        mse = float(np.mean((oq - cq) ** 2))
        psnr = float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))

        _go = 0.299*oq[:,:,0] + 0.587*oq[:,:,1] + 0.114*oq[:,:,2]
        _gm = 0.299*cq[:,:,0] + 0.587*cq[:,:,1] + 0.114*cq[:,:,2]
        _b = 8
        _rc = (_go.shape[0]//_b)*_b; _cc = (_go.shape[1]//_b)*_b
        _ao = _go[:_rc,:_cc].reshape(_rc//_b,_b,_cc//_b,_b).transpose(0,2,1,3).reshape(-1,_b,_b)
        _am = _gm[:_rc,:_cc].reshape(_rc//_b,_b,_cc//_b,_b).transpose(0,2,1,3).reshape(-1,_b,_b)
        _ma=_ao.mean(axis=(1,2)); _mb=_am.mean(axis=(1,2))
        _va=_ao.var(axis=(1,2));  _vb=_am.var(axis=(1,2))
        _cv=((_ao-_ma[:,None,None])*(_am-_mb[:,None,None])).mean(axis=(1,2))
        ssim = float(np.mean(
            (2*_ma*_mb+1e-4)*(2*_cv+9e-4) /
            ((_ma**2+_mb**2+1e-4)*(_va+_vb+9e-4))
        ))

        detection_before = detection_after = None
        if verify and self.extractor is not None:
            try:
                rb = self.extractor.detect_array(original_uint8)
                detection_before = dict(is_watermarked=rb.is_watermarked,
                                        confidence=rb.confidence,
                                        phase_match=rb.phase_match)
            except Exception:
                pass
            try:
                ra = self.extractor.detect_array(cleaned_uint8)
                detection_after = dict(is_watermarked=ra.is_watermarked,
                                       confidence=ra.confidence,
                                       phase_match=ra.phase_match)
            except Exception:
                pass

        nb = profile['n_black_refs']
        nw = profile['n_white_refs']
        nr = profile['n_random_refs']
        success = psnr > 28 and ssim > 0.88
        if detection_before and detection_after:
            cd = detection_before['confidence'] - detection_after['confidence']
            success = success and (cd > 0.10 or not detection_after['is_watermarked'])

        return BypassResult(
            success=success,
            cleaned_image=cleaned_uint8,
            psnr=psnr,
            ssim=ssim,
            detection_before=detection_before,
            detection_after=detection_after,
            stages_applied=stages,
            details={
                'version': 'v3_multi_res',
                'strength': strength,
                'passes': passes,
                'pass_schedule': str_sequence,
                'avg_luminance': avg_luminance,
                'profile_resolution': f'{prof_h}x{prof_w}',
                'exact_match': exact,
                'codebook_refs': f'{nb}b+{nw}w+{nr}r',
            },
        )
    
    def bypass_v3_file(
        self,
        input_path: str,
        output_path: str,
        codebook: 'SpectralCodebook',
        strength: str = 'moderate',
        verify: bool = True
    ) -> BypassResult:
        """Bypass watermark using v3 spectral pipeline and save result."""
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load: {input_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.bypass_v3(img_rgb, codebook, strength=strength, verify=verify)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR))
        
        return result


# ================================================================
# SPECTRAL CODEBOOK — Multi-Resolution V3 Frequency-Domain Profile
# ================================================================

class SpectralCodebook:
    """
    Multi-resolution watermark profile for SynthID bypass.

    Stores per-resolution profiles so a single codebook file handles
    images of any size.  Each profile captures the full spectral
    envelope at one (H, W) resolution:

        magnitude_profile   — avg |FFT| of the watermark signal
        phase_template      — circular-mean phase angle per bin
        phase_consistency   — coherence (0..1) per bin
        content_magnitude_baseline — avg |FFT| of content (Wiener)
        white_* / black_white_agreement — cross-validation data

    Build profiles with:
        extract_from_references()  — from pure-black / pure-white refs
        build_from_watermarked()   — from diverse watermarked content
    Both detect native resolution automatically and add a profile
    keyed by (H, W).  Call both to cover multiple resolutions.
    """

    CHANNEL_WEIGHTS = np.array([0.85, 1.0, 0.70])

    _PROFILE_ARRAYS = [
        'magnitude_profile', 'phase_template', 'phase_consistency',
        'content_magnitude_baseline', 'white_magnitude_profile',
        'white_phase_template', 'white_phase_consistency',
        'black_white_agreement',
    ]
    _PROFILE_SCALARS = ['n_black_refs', 'n_white_refs', 'n_random_refs']

    def __init__(self):
        self.profiles = {}  # {(H, W): profile_dict}

    @property
    def resolutions(self):
        return list(self.profiles.keys())

    @property
    def ref_shape(self):
        """Primary resolution (first added).  Backward-compat helper."""
        if not self.profiles:
            return None
        return next(iter(self.profiles))

    def get_profile(self, h, w):
        """Best-matching profile for target (H, W).

        Returns (profile_dict, (prof_H, prof_W), exact_match: bool).
        Prefers exact resolution match, else closest aspect ratio.
        """
        if (h, w) in self.profiles:
            return self.profiles[(h, w)], (h, w), True
        if not self.profiles:
            raise ValueError("Codebook has no profiles")
        target_ar = h / (w + 1e-9)
        best_key, best_score = None, float('inf')
        for kh, kw in self.profiles:
            ar_diff = abs(kh / (kw + 1e-9) - target_ar) / (target_ar + 1e-9)
            px_diff = abs(kh * kw - h * w) / (h * w + 1e-9)
            score = ar_diff * 2 + px_diff
            if score < best_score:
                best_score, best_key = score, (kh, kw)
        return self.profiles[best_key], best_key, False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_reference_images(directory: str, max_images: int = None) -> list:
        """Collect image files, excluding unwatermarked baselines."""
        import glob

        excl = {
            'black.jpg', 'black.jpeg', 'black.png',
            'white.jpg', 'white.jpeg', 'white.png',
        }
        paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
            paths.extend(glob.glob(os.path.join(directory, ext)))
            paths.extend(glob.glob(os.path.join(directory, ext.upper())))
        seen = set()
        out = []
        for p in sorted(paths):
            if os.path.basename(p).lower() in excl:
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
        if max_images:
            out = out[:max_images]
        return out

    @staticmethod
    def _load_image(fpath, target_shape=None):
        """Load image, optionally resizing to target_shape=(H, W)."""
        img = cv2.imread(fpath)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        if target_shape and (rgb.shape[0], rgb.shape[1]) != target_shape:
            rgb = cv2.resize(rgb, (target_shape[1], target_shape[0]),
                             interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
        return rgb

    @staticmethod
    def _image_shape(fpath):
        """Read image dimensions without loading full array."""
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return (img.shape[0], img.shape[1])

    @staticmethod
    def _accumulate_fft(img_rgb: np.ndarray):
        """Return (magnitude[3], phase_unit[3]) stacks for one image."""
        mags, units = [], []
        for ch in range(3):
            fft_r = np.fft.fft2(img_rgb[:, :, ch])
            mags.append(np.abs(fft_r))
            units.append(np.exp(1j * np.angle(fft_r)))
        return np.stack(mags, axis=-1), np.stack(units, axis=-1)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_from_references(
        self,
        black_dir: str,
        white_dir: str = None,
        random_dir: str = None,
        max_images: int = None,
    ):
        """
        Build a profile from black / white / random reference directories.

        The profile is stored at the native resolution of the black images.
        """
        black_files = self._list_reference_images(black_dir, max_images)
        if not black_files:
            raise ValueError(f"No images in {black_dir} (baselines excluded)")

        build_shape = self._image_shape(black_files[0])
        print(f"[black] {len(black_files)} images  resolution={build_shape}")

        mag_sum = phase_unit_sum = None
        n = 0
        for i, fp in enumerate(black_files):
            img = self._load_image(fp, target_shape=build_shape)
            if img is None:
                continue
            m, u = self._accumulate_fft(img)
            if mag_sum is None:
                mag_sum, phase_unit_sum = m.copy(), u.copy()
            else:
                mag_sum += m
                phase_unit_sum += u
            n += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(black_files)}")
            del m, u, img

        avg_mag = mag_sum / n
        pv = phase_unit_sum / n
        del mag_sum, phase_unit_sum

        profile = {
            'magnitude_profile': avg_mag,
            'phase_template': np.angle(pv),
            'phase_consistency': np.abs(pv),
            'content_magnitude_baseline': None,
            'white_magnitude_profile': None,
            'white_phase_template': None,
            'white_phase_consistency': None,
            'black_white_agreement': None,
            'n_black_refs': n,
            'n_white_refs': 0,
            'n_random_refs': 0,
        }
        print(f"  mean consistency (G)="
              f"{np.mean(profile['phase_consistency'][:,:,1]):.4f}")

        # ---- White refs ----
        if white_dir:
            white_files = self._list_reference_images(white_dir, max_images)
            if white_files:
                print(f"[white] {len(white_files)} images")
                wm_sum = wu_sum = None
                nw = 0
                for fp in white_files:
                    img = self._load_image(fp, target_shape=build_shape)
                    if img is None:
                        continue
                    m, u = self._accumulate_fft(255.0 - img)
                    if wm_sum is None:
                        wm_sum, wu_sum = m.copy(), u.copy()
                    else:
                        wm_sum += m
                        wu_sum += u
                    nw += 1
                    del m, u, img
                profile['white_magnitude_profile'] = wm_sum / nw
                wpv = wu_sum / nw
                profile['white_phase_template'] = np.angle(wpv)
                profile['white_phase_consistency'] = np.abs(wpv)
                profile['n_white_refs'] = nw
                profile['black_white_agreement'] = np.abs(np.cos(
                    profile['phase_template'] - profile['white_phase_template']
                ))
                agree_g = profile['black_white_agreement'][:, :, 1]
                print(f"  cross-validated bins (|cos|>0.90, G): "
                      f"{int(np.sum(agree_g > 0.90))}")

        # ---- Random / content baseline ----
        if random_dir:
            rand_files = self._list_reference_images(random_dir, max_images)
            if rand_files:
                print(f"[random] {len(rand_files)} images (content baseline)")
                rm_sum = None
                nr = 0
                for fp in rand_files:
                    img = self._load_image(fp, target_shape=build_shape)
                    if img is None:
                        continue
                    m, _ = self._accumulate_fft(img)
                    if rm_sum is None:
                        rm_sum = m.copy()
                    else:
                        rm_sum += m
                    nr += 1
                    del m, img
                profile['content_magnitude_baseline'] = rm_sum / nr
                profile['n_random_refs'] = nr

        self.profiles[build_shape] = profile
        self._print_top_carriers(profile, build_shape)
        nb = profile['n_black_refs']
        nw = profile['n_white_refs']
        nr = profile['n_random_refs']
        print(f"\nProfile added: {build_shape[0]}x{build_shape[1]}  "
              f"({nb}b+{nw}w+{nr}r refs)")

    def build_from_watermarked(
        self,
        image_dir: str,
        max_images: int = None,
        group_by_resolution: bool = False,
    ):
        """
        Build a profile from diverse watermarked images at native resolution.

        Content averages out across images; the fixed watermark signal
        survives in phase coherence.  Watermark magnitude is estimated as
        ``avg_magnitude × coherence²``.

        The profile is stored at the native resolution of the images.

        When *group_by_resolution* is True, images of different resolutions
        within the same directory are automatically grouped and a separate
        profile is built for each resolution.
        """
        files = self._list_reference_images(image_dir, max_images)
        if not files:
            raise ValueError(f"No images in {image_dir}")

        if group_by_resolution:
            groups = {}
            for fp in files:
                shape = self._image_shape(fp)
                if shape is not None:
                    groups.setdefault(shape, []).append(fp)
            print(f"[watermarked] {len(files)} images across "
                  f"{len(groups)} resolution(s)")
            for shape, group_files in sorted(groups.items()):
                self._build_watermarked_profile(shape, group_files)
        else:
            build_shape = self._image_shape(files[0])
            print(f"[watermarked] {len(files)} images  "
                  f"resolution={build_shape}")
            self._build_watermarked_profile(build_shape, files)

    def _build_watermarked_profile(self, build_shape, files):
        """Build a single watermarked-image profile at *build_shape*."""
        mag_sum = phase_unit_sum = None
        n = 0
        for i, fp in enumerate(files):
            img = self._load_image(fp, target_shape=build_shape)
            if img is None:
                continue
            m, u = self._accumulate_fft(img)
            if mag_sum is None:
                mag_sum, phase_unit_sum = m.copy(), u.copy()
            else:
                mag_sum += m
                phase_unit_sum += u
            n += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(files)}")
            del m, u, img

        if n == 0:
            print(f"  WARNING: No valid images for {build_shape}")
            return

        avg_mag = mag_sum / n
        pv = phase_unit_sum / n
        coherence = np.abs(pv)
        del mag_sum, phase_unit_sum

        profile = {
            'magnitude_profile': avg_mag * (coherence ** 2),
            'phase_template': np.angle(pv),
            'phase_consistency': coherence,
            'content_magnitude_baseline': avg_mag,
            'white_magnitude_profile': None,
            'white_phase_template': None,
            'white_phase_consistency': None,
            'black_white_agreement': None,
            'n_black_refs': 0,
            'n_white_refs': 0,
            'n_random_refs': n,
        }

        self.profiles[build_shape] = profile
        self._print_top_carriers(profile, build_shape)
        print(f"\nProfile added: {build_shape[0]}x{build_shape[1]}  "
              f"({n} watermarked refs)")

    @staticmethod
    def _print_top_carriers(profile, ref_shape):
        h, w = ref_shape
        cg = profile['phase_consistency'][:, :, 1].copy()
        cg[0, 0] = 0
        top = np.unravel_index(np.argsort(cg.ravel())[-10:], cg.shape)
        print("  Top-10 phase-consistent carriers (G):")
        bwa = profile.get('black_white_agreement')
        for fy, fx in zip(top[0][::-1], top[1][::-1]):
            fy_s = fy if fy <= h // 2 else fy - h
            fx_s = fx if fx <= w // 2 else fx - w
            mg = profile['magnitude_profile'][fy, fx, 1]
            cs = cg[fy, fx]
            xv = (f"  agree={bwa[fy, fx, 1]:.3f}" if bwa is not None else "")
            print(f"    ({fy_s:+4d},{fx_s:+4d})  mag={mg:9.0f}  "
                  f"cons={cs:.4f}{xv}")

    # ------------------------------------------------------------------
    # Watermark estimation (Wiener-style)
    # ------------------------------------------------------------------

    def estimate_watermark_fft(
        self,
        image_fft: np.ndarray,
        channel: int,
        strength: str = 'moderate',
        image_luminance: float = 0.5,
        profile: dict = None,
        ref_shape: tuple = None,
    ) -> np.ndarray:
        """
        Estimate watermark FFT for subtraction.

        If *profile* / *ref_shape* are not supplied, auto-selects the
        best-matching profile from ``image_fft.shape``.
        """
        if profile is None:
            profile, ref_shape, _ = self.get_profile(*image_fft.shape)
        if ref_shape is None:
            ref_shape = image_fft.shape

        cfg = {
            'gentle':     {'removal': 0.60, 'cons_floor': 0.70, 'mag_cap': 0.50,
                           'dc_radius': 30},
            'moderate':   {'removal': 0.80, 'cons_floor': 0.50, 'mag_cap': 0.70,
                           'dc_radius': 25},
            'aggressive': {'removal': 0.95, 'cons_floor': 0.30, 'mag_cap': 0.90,
                           'dc_radius': 20},
            'maximum':    {'removal': 1.00, 'cons_floor': 0.15, 'mag_cap': 0.95,
                           'dc_radius': 15},
        }.get(strength, {'removal': 0.80, 'cons_floor': 0.50, 'mag_cap': 0.70,
                         'dc_radius': 25})

        ch_w = float(self.CHANNEL_WEIGHTS[channel])
        H, W = ref_shape

        ref_mag = profile['magnitude_profile'][:, :, channel]
        ref_phase = profile['phase_template'][:, :, channel]
        consistency = profile['phase_consistency'][:, :, channel]

        w_mp = profile.get('white_magnitude_profile')
        if w_mp is not None:
            wm_mag = (ref_mag * (1.0 - image_luminance)
                      + w_mp[:, :, channel] * image_luminance)
        else:
            wm_mag = ref_mag.copy()

        bwa = profile.get('black_white_agreement')
        if bwa is not None:
            wm_mag = wm_mag * bwa[:, :, channel]

        dc_r = cfg['dc_radius']
        fy = np.arange(H).reshape(-1, 1).astype(np.float64)
        fx = np.arange(W).reshape(1, -1).astype(np.float64)
        fy = np.where(fy > H / 2, fy - H, fy)
        fx = np.where(fx > W / 2, fx - W, fx)
        dc_ramp = np.clip(np.sqrt(fy**2 + fx**2) / dc_r, 0, 1)
        wm_mag = wm_mag * dc_ramp

        cons_weight = np.clip(
            (consistency - cfg['cons_floor']) / (1.0 - cfg['cons_floor'] + 1e-9),
            0, 1,
        )

        subtract_mag = wm_mag * cons_weight * cfg['removal'] * ch_w
        subtract_mag = np.minimum(subtract_mag, np.abs(image_fft) * cfg['mag_cap'])

        return subtract_mag * np.exp(1j * ref_phase)

    def watermark_spatial(
        self,
        channel: int,
        strength: str = 'moderate',
        image_luminance: float = 0.5,
        profile: dict = None,
        ref_shape: tuple = None,
    ) -> np.ndarray:
        """
        Estimated watermark in spatial domain at a profile's native resolution.

        The caller can ``cv2.resize`` the result to any target resolution.
        """
        if profile is None:
            if not self.profiles:
                raise ValueError("No profiles")
            ref_shape = next(iter(self.profiles))
            profile = self.profiles[ref_shape]
        if ref_shape is None:
            ref_shape = next(iter(self.profiles))

        cb = profile.get('content_magnitude_baseline')
        pt = profile['phase_template'][:, :, channel]
        if cb is not None:
            synth_fft = cb[:, :, channel] * np.exp(1j * pt)
        else:
            synth_fft = (profile['magnitude_profile'][:, :, channel] * 10
                         * np.exp(1j * pt))

        wm_fft = self.estimate_watermark_fft(
            synth_fft, channel, strength=strength,
            image_luminance=image_luminance,
            profile=profile, ref_shape=ref_shape,
        )
        return np.real(np.fft.ifft2(wm_fft))

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # rfft symmetry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rfft_to_full_sym(rfft_half, H, W):
        """(H, W//2+1, C) → (H, W, C) via conjugate symmetry."""
        rw = W // 2 + 1
        full = np.zeros((H, W) + rfft_half.shape[2:], dtype=rfft_half.dtype)
        full[:, :rw] = rfft_half
        if W > 2:
            sky = (H - np.arange(H)) % H
            skx = W - np.arange(rw, W)
            full[:, rw:] = full[sky[:, None], skx[None, :]]
        return full

    @staticmethod
    def _rfft_to_full_anti(rfft_half, H, W):
        """Anti-symmetric variant (for phase)."""
        rw = W // 2 + 1
        full = np.zeros((H, W) + rfft_half.shape[2:], dtype=rfft_half.dtype)
        full[:, :rw] = rfft_half
        if W > 2:
            sky = (H - np.arange(H)) % H
            skx = W - np.arange(rw, W)
            full[:, rw:] = -full[sky[:, None], skx[None, :]]
        return full

    # ------------------------------------------------------------------
    # Save / Load — compact format
    #
    # Reduces codebook size ~25x through:
    #   1. rfft2 half-spectrum (conjugate symmetry → 2x)
    #   2. Drop derivable arrays (content_baseline, white_phase_*)
    #   3. float16 for magnitudes (log2-encoded) and phase
    #   4. uint8 for consistency and agreement ([0,1] → 0..255)
    #   5. Sparse storage for profiles where <50% of bins are active
    #      (bins with consistency < 0.15 have zero bypass contribution)
    # ------------------------------------------------------------------

    _CONS_THRESHOLD = 0.15  # minimum cons_floor across all strength levels

    def save(self, path: str):
        """Save in compact format (~20 MB for typical dual-resolution codebook)."""
        data = {
            'format_version': np.array(2),
            'resolutions': np.array(list(self.profiles.keys())),
        }

        for (h, w), prof in self.profiles.items():
            pfx = f'{h}x{w}/'
            rw = w // 2 + 1

            for key in self._PROFILE_SCALARS:
                data[pfx + key] = np.array(prof.get(key, 0))

            mag = prof['magnitude_profile']
            phase = prof['phase_template']
            cons = prof['phase_consistency']

            mag_r = mag[:, :rw, :]
            phase_r = phase[:, :rw, :]
            cons_r = cons[:, :rw, :]

            active_frac = float(np.mean(cons_r > self._CONS_THRESHOLD))
            use_sparse = active_frac < 0.50
            data[pfx + 'sparse'] = np.array(int(use_sparse))

            if use_sparse:
                for ch in range(3):
                    mask = cons_r[:, :, ch] >= self._CONS_THRESHOLD
                    idx = np.where(mask.ravel())[0].astype(np.uint32)
                    vals_m = mag_r[:, :, ch].ravel()[idx]
                    vals_p = phase_r[:, :, ch].ravel()[idx]
                    vals_c = cons_r[:, :, ch].ravel()[idx]
                    data[pfx + f'idx_{ch}'] = idx
                    data[pfx + f'mag_{ch}'] = np.log2(1.0 + vals_m).astype(np.float16)
                    data[pfx + f'phase_{ch}'] = vals_p.astype(np.float16)
                    data[pfx + f'cons_{ch}'] = np.round(vals_c * 255).clip(0, 255).astype(np.uint8)
            else:
                data[pfx + 'mag'] = np.log2(1.0 + mag_r).astype(np.float16)
                data[pfx + 'phase'] = phase_r.astype(np.float16)
                data[pfx + 'cons'] = np.round(cons_r * 255).clip(0, 255).astype(np.uint8)

                wmag = prof.get('white_magnitude_profile')
                if wmag is not None:
                    data[pfx + 'wmag'] = np.log2(1.0 + wmag[:, :rw, :]).astype(np.float16)

                agree = prof.get('black_white_agreement')
                if agree is not None:
                    data[pfx + 'agree'] = np.round(agree[:, :rw, :] * 255).clip(0, 255).astype(np.uint8)

        np.savez(path, **data)
        sz = os.path.getsize(path) / 1e6
        res_str = ', '.join(f'{h}x{w}' for h, w in self.profiles)
        print(f"Codebook saved → {path}  [{res_str}]  {sz:.1f} MB")

    def load(self, path: str):
        """Load codebook (auto-detects format version)."""
        d = np.load(path)
        fmt = int(d['format_version']) if 'format_version' in d else 0

        if fmt >= 2:
            self._load_compact(d)
        elif 'resolutions' in d:
            self._load_v1(d)
        else:
            self._load_legacy(d)

        res_str = ', '.join(f'{h}x{w}' for h, w in self.profiles)
        print(f"Codebook loaded: [{res_str}]")
        for (h, w), prof in self.profiles.items():
            nb, nw, nr = prof['n_black_refs'], prof['n_white_refs'], prof['n_random_refs']
            print(f"  {h}x{w}: {nb}b+{nw}w+{nr}r")

    def _load_compact(self, d):
        """Load format_version >= 2 (rfft + mixed precision + optional sparse)."""
        for res in d['resolutions']:
            h, w = int(res[0]), int(res[1])
            pfx = f'{h}x{w}/'
            rw = w // 2 + 1
            use_sparse = bool(int(d[pfx + 'sparse']))

            if use_sparse:
                mag_r = np.zeros((h, rw, 3), dtype=np.float64)
                phase_r = np.zeros((h, rw, 3), dtype=np.float64)
                cons_r = np.zeros((h, rw, 3), dtype=np.float64)
                for ch in range(3):
                    idx = d[pfx + f'idx_{ch}']
                    rows, cols = np.unravel_index(idx, (h, rw))
                    mag_r[rows, cols, ch] = np.power(2.0, d[pfx + f'mag_{ch}'].astype(np.float64)) - 1.0
                    phase_r[rows, cols, ch] = d[pfx + f'phase_{ch}'].astype(np.float64)
                    cons_r[rows, cols, ch] = d[pfx + f'cons_{ch}'].astype(np.float64) / 255.0
            else:
                mag_r = np.power(2.0, d[pfx + 'mag'].astype(np.float64)) - 1.0
                phase_r = d[pfx + 'phase'].astype(np.float64)
                cons_r = d[pfx + 'cons'].astype(np.float64) / 255.0

            mag_full = self._rfft_to_full_sym(mag_r, h, w)
            phase_full = self._rfft_to_full_anti(phase_r, h, w)
            cons_full = self._rfft_to_full_sym(cons_r, h, w)

            wmag_full = None
            if pfx + 'wmag' in d:
                wmag_r = np.power(2.0, d[pfx + 'wmag'].astype(np.float64)) - 1.0
                wmag_full = self._rfft_to_full_sym(wmag_r, h, w)

            agree_full = None
            if pfx + 'agree' in d:
                agree_r = d[pfx + 'agree'].astype(np.float64) / 255.0
                agree_full = self._rfft_to_full_sym(agree_r, h, w)

            # Reconstruct content_baseline for watermarked-only profiles
            content_base = None
            nb = int(d.get(pfx + 'n_black_refs', 0))
            if nb == 0:
                safe = np.maximum(cons_full ** 2, 1e-10)
                content_base = mag_full / safe

            self.profiles[(h, w)] = {
                'magnitude_profile': mag_full,
                'phase_template': phase_full,
                'phase_consistency': cons_full,
                'content_magnitude_baseline': content_base,
                'white_magnitude_profile': wmag_full,
                'white_phase_template': None,
                'white_phase_consistency': None,
                'black_white_agreement': agree_full,
                'n_black_refs': nb,
                'n_white_refs': int(d.get(pfx + 'n_white_refs', 0)),
                'n_random_refs': int(d.get(pfx + 'n_random_refs', 0)),
            }

    def _load_v1(self, d):
        """Load v1 multi-resolution format (full-precision arrays)."""
        for res in d['resolutions']:
            h, w = int(res[0]), int(res[1])
            pfx = f'{h}x{w}/'
            prof = {}
            for key in self._PROFILE_ARRAYS:
                fk = pfx + key
                prof[key] = d[fk] if fk in d else None
            for key in self._PROFILE_SCALARS:
                fk = pfx + key
                prof[key] = int(d[fk]) if fk in d else 0
            self.profiles[(h, w)] = prof

    def _load_legacy(self, d):
        """Load original single-resolution format."""
        h, w = int(d['ref_shape'][0]), int(d['ref_shape'][1])
        prof = {}
        for key in self._PROFILE_ARRAYS:
            prof[key] = d[key] if key in d else None
        prof['n_black_refs'] = int(d['n_black_refs']) if 'n_black_refs' in d else 0
        prof['n_white_refs'] = int(d['n_white_refs']) if 'n_white_refs' in d else 0
        prof['n_random_refs'] = int(d['n_random_refs']) if 'n_random_refs' in d else 0
        self.profiles[(h, w)] = prof


# ================================================================
# CLI INTERFACE
# ================================================================

def _print_bypass_result(result, mode_str):
    print("\n" + "=" * 60)
    print("SYNTHID BYPASS RESULTS")
    print("=" * 60)
    print(f"  Pipeline: {mode_str}")
    print(f"  Success: {result.success}")
    print(f"  PSNR: {result.psnr:.2f} dB")
    print(f"  SSIM: {result.ssim:.4f}")
    print(f"  Stages: {', '.join(result.stages_applied)}")
    if result.details:
        for k, v in result.details.items():
            print(f"  {k}: {v}")
    if result.detection_before:
        print("\n  Before Bypass:")
        for k, v in result.detection_before.items():
            print(f"    {k}: {v}")
    if result.detection_after:
        print("\n  After Bypass:")
        for k, v in result.detection_after.items():
            print(f"    {k}: {v}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SynthID Watermark Bypass & Codebook Builder')
    sub = parser.add_subparsers(dest='command')

    # --- build-codebook ---
    bp = sub.add_parser('build-codebook',
                        help='Build a multi-resolution spectral codebook')
    bp.add_argument('--black', help='Black reference images directory')
    bp.add_argument('--white', help='White reference images directory')
    bp.add_argument('--watermarked', nargs='+', default=[],
                    help='Watermarked image dirs (one profile per dir)')
    bp.add_argument('--group-by-resolution', action='store_true',
                    help='Auto-group images by resolution within each dir')
    bp.add_argument('--output', required=True, help='Output .npz path')

    # --- bypass ---
    byp = sub.add_parser('bypass', help='Remove SynthID watermark from image')
    byp.add_argument('input', help='Input image path')
    byp.add_argument('output', help='Output image path')
    byp.add_argument('--version', choices=['v1', 'v2', 'v3'], default='v3')
    byp.add_argument('--strength', default='aggressive',
                     choices=['gentle', 'moderate', 'aggressive', 'maximum'])
    byp.add_argument('--codebook', help='Spectral codebook .npz (V3)')
    byp.add_argument('--detector', help='Detector codebook .pkl')
    byp.add_argument('--no-verify', action='store_true')

    # --- legacy positional (backward compat) ---
    byp_legacy = sub.add_parser('legacy', help=argparse.SUPPRESS)
    byp_legacy.add_argument('input', help='Input image path')
    byp_legacy.add_argument('output', help='Output image path')
    byp_legacy.add_argument('--v2', action='store_true')
    byp_legacy.add_argument('--mode', default='balanced')
    byp_legacy.add_argument('--strength', default='aggressive')
    byp_legacy.add_argument('--codebook', default=None)
    byp_legacy.add_argument('--no-verify', action='store_true')

    args = parser.parse_args()

    if args.command == 'build-codebook':
        codebook = SpectralCodebook()
        if args.black:
            codebook.extract_from_references(
                args.black, white_dir=args.white)
        for d in args.watermarked:
            codebook.build_from_watermarked(
                d, group_by_resolution=args.group_by_resolution)
        if not codebook.profiles:
            parser.error("Provide --black and/or --watermarked directories")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        codebook.save(args.output)

    elif args.command in ('bypass', 'legacy'):
        extractor = None
        det_path = getattr(args, 'detector', None) or getattr(args, 'codebook', None)
        if det_path and not args.no_verify:
            try:
                from robust_extractor import RobustSynthIDExtractor
                extractor = RobustSynthIDExtractor()
                extractor.load_codebook(det_path)
            except Exception as e:
                print(f"Warning: Could not load extractor: {e}")

        bypass = SynthIDBypass(extractor=extractor)
        ver = getattr(args, 'version', None)
        if ver == 'v3' or (args.command == 'bypass' and ver is None):
            ver = 'v3'
        if getattr(args, 'v2', False):
            ver = 'v2'

        if ver == 'v3':
            cb_path = args.codebook
            if not cb_path:
                parser.error("V3 requires --codebook path")
            codebook = SpectralCodebook()
            codebook.load(cb_path)
            result = bypass.bypass_v3_file(
                args.input, args.output, codebook,
                strength=args.strength, verify=not args.no_verify)
            _print_bypass_result(result, f"v3/{args.strength}")
        elif ver == 'v2':
            result = bypass.bypass_v2_file(
                args.input, args.output,
                strength=args.strength, verify=not args.no_verify)
            _print_bypass_result(result, f"v2/{args.strength}")
        else:
            mode = getattr(args, 'mode', 'balanced')
            result = bypass.bypass_file(
                args.input, args.output,
                mode=mode, verify=not args.no_verify)
            _print_bypass_result(result, f"v1/{mode}")
        print(f"Saved to: {args.output}")

    else:
        parser.print_help()
