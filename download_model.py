#!/usr/bin/env python3
"""
Download Qwen3-VL-8B-Instruct Model

This script downloads the Qwen3-VL model and caches it locally.
Run this first before using main.py to run the model.

Usage:
    python download_model.py
    python download_model.py --cache-dir /custom/path
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(model_id: str, cache_dir: Path):
    """Download the model with progress tracking."""

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    model_cache_path = cache_dir / "models--Qwen--Qwen3-VL-8B-Instruct"

    if model_cache_path.exists():
        print(f"✓ Model already downloaded!")
        print(f"  Location: {model_cache_path}")
        print(f"\nYou can now run the model with:")
        print(f"  uv run python main.py --interactive")
        return

    print("="*60)
    print("Downloading Qwen3-VL-8B-Instruct Model")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print(f"Size: ~18GB")
    print(f"This may take a while depending on your internet speed...")
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
        print(f"\nYou can now run the model with:")
        print(f"  uv run python main.py --interactive")
        print(f"  uv run python main.py --text 'Your question here'")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Don't worry - the download will resume from where it left off next time!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Make sure you have ~18GB of free disk space")
        print("  3. Try running the script again (download will resume)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-VL-8B-Instruct model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID to download (default: Qwen/Qwen3-VL-8B-Instruct)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Directory to cache the model (default: ./models)"
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    download_model(args.model, cache_dir)


if __name__ == "__main__":
    main()
