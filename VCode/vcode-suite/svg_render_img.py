from pathlib import Path
import sys
import cairosvg
from PIL import Image
import io
import argparse

import concurrent.futures

def process_single_svg(args):
    """Worker function to render a single SVG"""
    svg_file, input_path, output_path, ref_path = args
    
    relative_path = svg_file.relative_to(input_path)
    output_dir = output_path / relative_path.parent
    
    # We might have race condition on creating dirs, but valid in threaded mode usually if exist_ok=True
    output_dir.mkdir(parents=True, exist_ok=True)
    name = svg_file.stem

    out_ext = ".png"
    if (ref_path / relative_path.with_suffix('.jpg')).exists() or \
       (ref_path / relative_path.with_suffix('.jpeg')).exists():
        out_ext = ".jpg"
    
    output_file = output_dir / f"{name}{out_ext}"

    try:
        # Render SVG to high-resolution PNG bytes in memory (try 3x, fallback to 2x/1x)
        png_bytes = None
        for scale in [3.0, 2.0, 1.0]:
            try:
                png_bytes = cairosvg.svg2png(url=str(svg_file), scale=scale)
                break
            except Exception as e:
                if scale == 1.0:
                    raise e
                # print(f"  ⚠️ Scale {scale}x failed for {relative_path}, retrying...")

        if out_ext == ".png":
            with open(output_file, "wb") as f:
                f.write(png_bytes)
        else:
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            image.save(output_file, "JPEG", quality=95, optimize=True)

        try:
            shown_output = output_file.relative_to(Path.cwd())
        except ValueError:
            shown_output = output_file

        return (True, str(relative_path), str(shown_output))
    except Exception as e:
        error_msg = str(e).lower()
        if "no element found" in error_msg or "premature end of file" in error_msg:
            try:
                svg_file.unlink()
                return (False, str(relative_path), "Empty or invalid file detected, deleted.")
            except Exception as delete_error:
                return (False, str(relative_path), f"Failed to delete invalid file - {delete_error}")
        else:
            return (False, str(relative_path), f"Conversion failed - {e}")

def render_svgs_to_images(svg_input_dir, image_output_dir, reference_dir, max_workers=1):
    """
    Converts SVG files to raster images parallelly.
    """
    input_path = Path(svg_input_dir)
    output_path = Path(image_output_dir)
    ref_path = Path(reference_dir)
    
    if not input_path.exists():
        print(f"❌ Error: Input SVG directory not found: {input_path}")
        return
    
    if not ref_path.exists():
        print(f"❌ Error: Reference directory not found: {ref_path}")
        return

    svg_files = sorted(list(input_path.rglob("*.svg")))
    
    if not svg_files:
        print(f"⚠️ No SVG files found in '{input_path}'.")
        return

    print(f"🔍 Found {len(svg_files)} SVG files. Starting conversion with {max_workers} threads...")
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = []
    for svg_file in svg_files:
        tasks.append((svg_file, input_path, output_path, ref_path))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_svg, task): task for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            success, rel_path, msg = future.result()
            
            if success:
                print(f"[{i}/{len(svg_files)}] ✅ {rel_path} → {msg}")
            else:
                print(f"[{i}/{len(svg_files)}] ❌ {rel_path}: {msg}")

    print("✅ Conversion complete!")

def main():
    """Parses command-line arguments and starts the rendering process."""
    parser = argparse.ArgumentParser(
        description="Converts a directory of SVG files to high-resolution raster images (PNG or JPG)."
    )
    
    parser.add_argument(
        "--svg-input-dir",
        required=True,
        help="The source directory containing SVG files to be converted."
    )
    parser.add_argument(
        "--image-output-dir",
        required=True,
        help="The destination directory where rendered images will be saved."
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Path to the directory of original images, used to match the output format (e.g., .jpg or .png)."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of threads for parallel rendering."
    )
    
    args = parser.parse_args()
    
    render_svgs_to_images(
        args.svg_input_dir, 
        args.image_output_dir, 
        args.reference_dir,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()