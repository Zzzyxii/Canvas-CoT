import os
import re
import argparse
from typing import Optional, List, Tuple


def _extract_last_complete_svg_block(content: str) -> Optional[str]:
    """Extract the last complete top-level <svg>...</svg> block.

    Why: SVGs may legally contain nested <svg>. A naive "last <svg> before last
    </svg>" approach can accidentally return an inner SVG fragment.
    """
    token_pattern = re.compile(r"<svg\b[^>]*>|</svg>", re.IGNORECASE)
    stack: List[int] = []
    blocks: List[Tuple[int, int]] = []

    for match in token_pattern.finditer(content):
        token = match.group(0).lower()
        if token.startswith("<svg"):
            stack.append(match.start())
        else:
            if not stack:
                continue
            start = stack.pop()
            if not stack:
                blocks.append((start, match.end()))

    if not blocks:
        return None

    start, end = blocks[-1]
    return content[start:end]

def clean_svg_file(file_path):
    """
    Cleans a single SVG file by keeping only the last valid <svg>...</svg> block
    and escaping standalone '&' characters.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error: Could not read file {file_path}: {e}")
        return

    cleaned_content = _extract_last_complete_svg_block(content)
    if not cleaned_content:
        return

    cleaned_content = re.sub(r'&(?!amp;|lt;|gt;|quot;|#)', '&amp;', cleaned_content)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
    except Exception as e:
        print(f"Error: Could not write to file {file_path}: {e}")

def process_svg_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: The provided path '{folder_path}' is not a valid directory.")
        return
    
    print(f"Starting to process folder: {folder_path}")
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith('.svg'):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, folder_path)
                print(f"  Processing: {relative_path}")
                clean_svg_file(file_path)
                file_count += 1
    
    print(f"\nProcessing complete. Total files handled: {file_count}.")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively cleans all SVG files in a directory by keeping only the last SVG block and fixing entities."
    )
    
    parser.add_argument(
        "--svg-folder",
        required=True,
        help="Path to the target directory containing SVG files."
    )
    
    args = parser.parse_args()
    
    process_svg_folder(args.svg_folder)

if __name__ == "__main__":
    main()