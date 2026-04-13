"""
Generate pure-black and pure-white reference images via the Gemini API.

These images carry SynthID watermarks but have no content signal,
making them ideal for extracting watermark carrier frequencies.

Usage:
    export GEMINI_API_KEY="your-key-here"
    pip install google-genai

    python generate_references.py --color black --count 50
    python generate_references.py --color white --count 50
    python generate_references.py --color both  --count 50

Output is saved to gemini_black_nb_pro/ and gemini_white_nb_pro/.
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini API setup
# ---------------------------------------------------------------------------

def get_client():
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Source image creation (pure black / pure white)
# ---------------------------------------------------------------------------

def make_source_image(color: str, size: int = 512) -> bytes:
    """Create a pure-color PNG in memory to attach as input."""
    value = 0 if color == "black" else 255
    img = Image.new("RGB", (size, size), (value, value, value))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

ASPECT_RATIOS = {
    "1:1":   (1024, 1024),  # square
    "9:16":  (1344, 768),   # portrait phone
    "16:9":  (768, 1344),   # landscape phone
    "4:3":   (864, 1184),   # classic photo landscape
    "3:4":   (1184, 864),   # classic photo portrait
    "2:3":   (1536, 1024),  # portrait standard
    "3:2":   (1024, 1536),  # landscape standard
    "1:2":   (2048, 1024),  # tall portrait
    "2:1":   (1024, 2048),  # wide landscape
}


def generate_single_image(client, color: str, source_bytes: bytes,
                          aspect_ratio: str = None, max_retries: int = 5):
    """Generate one watermarked reference image via Gemini.

    Args:
        client: google.genai.Client instance
        color: "black" or "white"
        source_bytes: PNG bytes of the pure-color source image
        aspect_ratio: e.g. "9:16", "4:3", "3:4" or None for default
        max_retries: retries on rate-limit (429) errors

    Returns:
        PIL.Image or None if generation failed
    """
    from google.genai import types

    source_part = types.Part.from_bytes(data=source_bytes, mime_type="image/png")
    image_config = None
    if aspect_ratio:
        image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=["Recreate this image exactly as it is", source_part],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=image_config,
                ),
            )
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        raw = part.inline_data.data
                        return Image.open(io.BytesIO(raw))
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 2 ** attempt
                print(f"    rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    return None


def run(color: str, count: int, delay: float, ratios: list[str]):
    client = get_client()
    colors = ["black", "white"] if color == "both" else [color]

    for c in colors:
        for ratio in ratios:
            expected_h, expected_w = ASPECT_RATIOS[ratio]
            tag = f"{expected_h}x{expected_w}"
            out_dir = Path(f"gemini_{c}_nb_pro") / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            existing = list(out_dir.glob("*.png"))
            start_idx = len(existing)

            source_bytes = make_source_image(c)
            print(f"\n[{c} / {ratio} / {tag}] Generating {count} images -> {out_dir}/")

            success = 0
            for i in range(count):
                idx = start_idx + i
                try:
                    img = generate_single_image(client, c, source_bytes,
                                                aspect_ratio=ratio)
                    if img is None:
                        print(f"  [{idx}] skipped (no image in response)")
                        continue

                    fname = out_dir / f"ref_{c}_{tag}_{idx:04d}.png"
                    img.save(fname, format="PNG")
                    success += 1
                    print(f"  [{idx}] saved {fname.name}  "
                          f"({img.size[0]}x{img.size[1]})")

                except Exception as e:
                    print(f"  [{idx}] error: {e}")

                if delay > 0:
                    time.sleep(delay)

            print(f"  Done: {success}/{count}")

    print("\nAll done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_RATIOS = list(ASPECT_RATIOS.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SynthID reference images via Gemini API")
    parser.add_argument("--color", choices=["black", "white", "both"],
                        default="both", help="Which color to generate")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of images per color per ratio (default: 50)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API calls (rate limiting)")
    parser.add_argument("--ratios", nargs="+", choices=ALL_RATIOS,
                        default=ALL_RATIOS,
                        help="Aspect ratios to generate (default: all)")
    args = parser.parse_args()
    run(args.color, args.count, args.delay, args.ratios)
