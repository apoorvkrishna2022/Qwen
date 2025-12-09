#!/usr/bin/env python3
"""
Qwen3-VL-8B MLX Runner (Fast!)

This script runs the MLX-optimized version of Qwen3-VL on Apple Silicon.
3-5x faster than the PyTorch version!

NOTE: Run download_model_mlx.py first to download the MLX model!

Usage:
    python download_model_mlx.py  # Run this first!
    python main_mlx.py --interactive
    python main_mlx.py --text "Explain quantum physics"
    python main_mlx.py --image photo.jpg --text "What's in this image?"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


QUANTIZATION_OPTIONS = {
    "4bit": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit",
    "5bit": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit",
    "8bit": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit",
}


def load_model(quantization: str = "4bit", cache_dir: Optional[Path] = None):
    """Load the MLX model and processor."""

    if cache_dir is None:
        cache_dir = Path("./models_mlx")

    model_id = QUANTIZATION_OPTIONS[quantization]

    print(f"Loading MLX model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print("⚡ Using MLX (optimized for Apple Silicon)")

    # Check if model is cached
    model_cache_name = model_id.replace("/", "--models--")
    model_cache_path = cache_dir / model_cache_name
    if not model_cache_path.exists():
        print("\n" + "="*60)
        print("ERROR: MLX model not found!")
        print("="*60)
        print("Please download the MLX model first by running:")
        print(f"  uv run python download_model_mlx.py --quantization {quantization}")
        print("="*60 + "\n")
        sys.exit(1)

    print("✓ Model found in cache!")

    # Load MLX model
    print("\nLoading model into memory...")

    try:
        # Load from cache directory
        cache_path_str = str(cache_dir)
        model, processor = load(model_id, tokenizer_config={"cache_dir": cache_path_str})

        print(f"\n✓ MLX model loaded successfully!")
        print(f"  Quantization: {quantization}")
        print(f"  Ready for fast inference!")

        return model, processor

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTry re-downloading the model:")
        print(f"  uv run python download_model_mlx.py --quantization {quantization}")
        sys.exit(1)


def generate_response(
    model,
    processor,
    text: str,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7
):
    """Generate a response from the MLX model."""

    # Prepare the prompt
    if image_path:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>{text}"
        image = image_path
    elif video_path:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        # For video, mlx-vlm typically processes frames as images
        prompt = f"<|vision_start|><|video_pad|><|vision_end|>{text}"
        image = video_path
    else:
        prompt = text
        image = None

    # Generate response
    print("\nGenerating response...")

    try:
        output = generate(
            model,
            processor,
            image=image,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False
        )

        return output

    except Exception as e:
        raise Exception(f"Error during generation: {e}")


def interactive_mode(model, processor, max_tokens: int = 512):
    """Run the model in interactive chat mode."""
    print("\n" + "="*60)
    print("Interactive Chat Mode (MLX - Fast!)")
    print("="*60)
    print("Commands:")
    print("  /image <path>  - Add an image to the next message")
    print("  /video <path>  - Add a video to the next message")
    print("  /clear         - Clear image/video")
    print("  /quit or /exit - Exit interactive mode")
    print("="*60 + "\n")

    current_image = None
    current_video = None

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if user_input in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break
                elif user_input.startswith("/image "):
                    current_image = user_input[7:].strip()
                    if Path(current_image).exists():
                        print(f"Image set: {current_image}")
                    else:
                        print(f"Warning: Image not found: {current_image}")
                        current_image = None
                    continue
                elif user_input.startswith("/video "):
                    current_video = user_input[7:].strip()
                    if Path(current_video).exists():
                        print(f"Video set: {current_video}")
                    else:
                        print(f"Warning: Video not found: {current_video}")
                        current_video = None
                    continue
                elif user_input == "/clear":
                    current_image = None
                    current_video = None
                    print("Image/video cleared")
                    continue
                else:
                    print(f"Unknown command: {user_input}")
                    continue

            # Generate response
            response = generate_response(
                model,
                processor,
                user_input,
                image_path=current_image,
                video_path=current_video,
                max_tokens=max_tokens
            )

            print(f"\nQwen: {response}\n")

            # Clear media after use
            if current_image or current_video:
                current_image = None
                current_video = None
                print("(Image/video cleared after use)")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-8B with MLX (fast on Apple Silicon)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text prompt/question"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to image file"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "5bit", "8bit"],
        default="4bit",
        help="Which quantization to use (default: 4bit)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models_mlx",
        help="Directory where model is cached (default: ./models_mlx)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )

    args = parser.parse_args()

    # Load model
    try:
        cache_dir = Path(args.cache_dir)
        model, processor = load_model(args.quantization, cache_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, processor, args.max_tokens)
    # Single query mode
    elif args.text:
        try:
            response = generate_response(
                model,
                processor,
                args.text,
                image_path=args.image,
                video_path=args.video,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(f"\n{response}\n")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nFirst time setup:")
        print("  uv run python download_model_mlx.py  # Download MLX model (~2.5GB)")
        print("\nExamples:")
        print("  uv run python main_mlx.py --interactive")
        print("  uv run python main_mlx.py --text 'Explain quantum computing'")
        print("  uv run python main_mlx.py --image photo.jpg --text 'Describe this image'")


if __name__ == "__main__":
    main()
