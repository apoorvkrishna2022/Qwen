#!/usr/bin/env python3
"""
Batch process invoice images and convert them to markdown.
Uses the official Qwen3-VL API from HuggingFace.
"""

import sys
from pathlib import Path
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def load_model(cache_dir: Path = None):
    """Load the Qwen3-VL model and processor."""

    if cache_dir is None:
        cache_dir = Path("./models")

    print(f"Loading model from {cache_dir}...")

    # Check if model is cached
    model_cache_path = cache_dir / "models--Qwen--Qwen3-VL-8B-Instruct"
    if not model_cache_path.exists():
        print("\nERROR: Model not found!")
        print("Please download the model first by running:")
        print("  uv run python download_model.py")
        sys.exit(1)

    # Load model using official API
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir=str(cache_dir),
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir=str(cache_dir),
        trust_remote_code=True
    )

    print("✓ Model loaded successfully!\n")
    return model, processor


def process_invoice_to_markdown(model, processor, image_path: Path, max_tokens: int = 2048):
    """Process a single invoice image and convert to markdown."""

    prompt = """Convert this invoice/receipt image to well-structured markdown format.

Extract and organize the following information:
- Business/vendor name and contact details
- Invoice/receipt number and date
- Itemized list of products/services with prices
- Subtotals, taxes, and total amount
- Payment information if available

Use proper markdown formatting with headers, tables, and lists."""

    # Format messages following official API
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs using official method
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate response with recommended parameters for vision-language tasks
    print(f"Processing: {image_path.name}...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )

    # Trim to only new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode response
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


def main():
    # Configuration
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [
        f for f in samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {samples_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} invoice images to process\n")
    print("="*60)

    # Load model once
    model, processor = load_model()

    # Process each image
    for image_path in image_files:
        try:
            # Generate markdown
            markdown_content = process_invoice_to_markdown(model, processor, image_path)

            # Save to file
            output_file = output_dir / f"{image_path.stem}.md"
            output_file.write_text(markdown_content)

            print(f"✓ Saved to: {output_file}")
            print("-"*60)

        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            print("-"*60)

    print("\n" + "="*60)
    print(f"✓ Processing complete! Check the '{output_dir}' directory for results.")
    print("="*60)


if __name__ == "__main__":
    main()
