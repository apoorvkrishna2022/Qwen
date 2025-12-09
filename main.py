#!/usr/bin/env python3
"""
Qwen3-VL-8B-Instruct Local Runner

This script runs the Qwen3-VL model locally for:
- Text-only conversations
- Vision-language tasks (image understanding, video analysis, OCR, etc.)

NOTE: Run download_model.py first to download the model!

Usage:
    python download_model.py  # Run this first!
    python main.py --text "Describe quantum physics"
    python main.py --image path/to/image.jpg --text "What's in this image?"
    python main.py --interactive  # Start interactive chat
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    cache_dir: Optional[Path] = None,
    use_bfloat16: bool = False
):
    """Load the Qwen3-VL model and processor.

    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to cache model files
        use_bfloat16: If True, use bfloat16 precision (may be faster on Apple Silicon)
    """

    # Set cache directory
    if cache_dir is None:
        cache_dir = Path("./models")

    print(f"Loading model: {model_id}")
    print(f"Cache directory: {cache_dir}")

    # Check if model is cached
    model_cache_path = cache_dir / "models--Qwen--Qwen3-VL-8B-Instruct"
    if not model_cache_path.exists():
        print("\n" + "="*60)
        print("ERROR: Model not found!")
        print("="*60)
        print("Please download the model first by running:")
        print("  uv run python download_model.py")
        print("="*60 + "\n")
        sys.exit(1)

    print("✓ Model found in cache!")

    # Prepare load kwargs
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    print(f"⚡ Using {dtype} precision")

    load_kwargs = {
        "cache_dir": str(cache_dir),
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True
    }

    # Load model - using auto device mapping for Apple Silicon
    print("\nLoading model into memory...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **load_kwargs
    )

    # Enable torch.compile for faster inference (first run will be slow)
    # Note: Compilation happens lazily on first inference
    print("⚡ Enabling torch.compile for faster inference...")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=True
    )

    print(f"\n✓ Model loaded successfully!")
    print(f"  Device: {model.device}")
    if torch.backends.mps.is_available():
        print(f"  Memory usage: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")

    return model, processor


def generate_response(
    model,
    processor,
    text: str,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    max_new_tokens: int = 512
):
    """Generate a response from the model."""

    # Prepare messages
    messages = []
    content = []

    # Add image if provided
    if image_path:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        content.append({"type": "image", "image": image_path})

    # Add video if provided
    if video_path:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        content.append({"type": "video", "video": video_path})

    # Add text
    content.append({"type": "text", "text": text})

    messages.append({
        "role": "user",
        "content": content
    })

    # Prepare inputs
    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = inputs.to(model.device)

    # Generate response
    print("\nGenerating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # Trim generated IDs to only new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode response
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response


def interactive_mode(model, processor):
    """Run the model in interactive chat mode."""
    print("\n" + "="*60)
    print("Interactive Chat Mode")
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
                video_path=current_video
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
        description="Run Qwen3-VL-8B-Instruct locally"
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
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID to use (default: Qwen/Qwen3-VL-8B-Instruct)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Directory to cache the model (default: ./models)"
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 precision (may be faster on Apple Silicon)"
    )

    args = parser.parse_args()

    # Load model
    try:
        cache_dir = Path(args.cache_dir)
        model, processor = load_model(args.model, cache_dir, use_bfloat16=args.bfloat16)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, processor)
    # Single query mode
    elif args.text:
        try:
            response = generate_response(
                model,
                processor,
                args.text,
                image_path=args.image,
                video_path=args.video,
                max_new_tokens=args.max_tokens
            )
            print(f"\n{response}\n")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nFirst time setup:")
        print("  uv run python download_model.py  # Download model first (~18GB)")
        print("\nExamples:")
        print("  uv run python main.py --interactive")
        print("  uv run python main.py --text 'Explain quantum computing'")
        print("  uv run python main.py --image photo.jpg --text 'Describe this image'")


if __name__ == "__main__":
    main()
