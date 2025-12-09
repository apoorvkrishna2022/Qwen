#!/usr/bin/env python3
"""
Download Qwen3-VL-8B MLX Model

Downloads the MLX-optimized version of Qwen3-VL for Apple Silicon.
This version is 3-5x faster than the PyTorch version!

Usage:
    python download_model_mlx.py                    # Download 4-bit (fastest, recommended)
    python download_model_mlx.py --quantization 5bit # Download 5-bit (balanced)
    python download_model_mlx.py --quantization 8bit # Download 8-bit (best quality)
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


QUANTIZATION_OPTIONS = {
    "4bit": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit",
    "5bit": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit",
    "8bit": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit",
}

QUANTIZATION_SIZES = {
    "4bit": "~2.5GB",
    "5bit": "~3.5GB",
    "8bit": "~5.5GB",
}


def download_model(quantization: str, cache_dir: Path):
    """Download the MLX model with progress tracking."""

    model_id = QUANTIZATION_OPTIONS[quantization]
    size = QUANTIZATION_SIZES[quantization]

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name for cache path check
    model_cache_name = model_id.replace("/", "--models--")
    model_cache_path = cache_dir / model_cache_name

    if model_cache_path.exists():
        print(f"✓ Model already downloaded!")
        print(f"  Location: {model_cache_path}")
        print(f"  Quantization: {quantization}")
        print(f"\nYou can now run the model with:")
        print(f"  uv run python main_mlx.py --interactive")
        return

    print("="*60)
    print("Downloading Qwen3-VL-8B MLX Model")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print(f"Quantization: {quantization}")
    print(f"Size: {size}")
    print(f"Optimized for: Apple Silicon (M1/M2/M3/M4)")
    print("="*60 + "\n")

    try:
        # Download with progress bars
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False,
        )

        print("\n" + "="*60)
        print("✓ Download Complete!")
        print("="*60)
        print(f"Model saved to: {model_path}")
        print(f"Quantization: {quantization}")
        print(f"\nThis MLX model is 3-5x faster than PyTorch!")
        print(f"\nYou can now run the model with:")
        print(f"  uv run python main_mlx.py --interactive")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Don't worry - the download will resume from where it left off next time!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print(f"  2. Make sure you have ~{size} of free disk space")
        print("  3. Try running the script again (download will resume)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download MLX-optimized Qwen3-VL-8B model for Apple Silicon"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "5bit", "8bit"],
        default="4bit",
        help="Quantization level (default: 4bit for fastest inference)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models_mlx",
        help="Directory to cache the model (default: ./models_mlx)"
    )

    args = parser.parse_args()

    print("\nQuantization options:")
    print("  4-bit: ~2.5GB, fastest inference (recommended)")
    print("  5-bit: ~3.5GB, balanced speed/quality")
    print("  8-bit: ~5.5GB, best quality")
    print(f"\nSelected: {args.quantization}\n")

    cache_dir = Path(args.cache_dir)
    download_model(args.quantization, cache_dir)


if __name__ == "__main__":
    main()
