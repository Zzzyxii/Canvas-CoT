"""Multi-round per-file SVG revision using the slz_reason blackboard agent.

This script is intentionally separate from revision.py.

Core idea:
- For each SVG file:
  - Render current SVG to an image (via cairosvg).
  - Compare original image vs rendered image.
  - Ask blackboard agent to produce a revised SVG ONLY.
  - Repeat for N rounds.

Outputs:
- Optimized SVGs are written to --output-folder, preserving relative paths.
- Per-round logs are written to --analysis-folder for debugging.

Notes:
- Requires system cairo libs for cairosvg.
- Requires OPENAI-compatible endpoint; pass via --base-url/--api-key or env vars.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import mimetypes
import traceback
from pathlib import Path
from typing import Optional, Tuple
from bs4 import BeautifulSoup

import cairosvg


def get_image_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type

    ext = Path(image_path).suffix.lower()
    ext_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return ext_to_mime.get(ext, "image/png")


def _encode_file_as_data_url(file_path: str) -> str:
    mime_type = get_image_mime_type(file_path)
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def find_matching_image(base_folder: str, relative_svg_path: Path) -> Optional[Path]:
    base_folder_path = Path(base_folder)
    svg_stem = relative_svg_path.stem

    search_dir = base_folder_path / relative_svg_path.parent
    if not search_dir.is_dir():
        return None

    for ext in [".png", ".jpg", ".jpeg"]:
        potential_path = search_dir / f"{svg_stem}{ext}"
        if potential_path.exists():
            return potential_path
    return None


def _try_import_blackboard_model():
    """Best-effort import of the blackboard agent model."""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from slz_reason.openai_model_slz_black_v6_2 import OpenAIModel as BlackboardOpenAIModel  # type: ignore
        from slz_reason.openai_model_slz_black_v6_2 import parse_response as blackboard_parse_response  # type: ignore

        return BlackboardOpenAIModel, blackboard_parse_response
    except Exception:
        return None, None


def _extract_svg(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("<svg")
    end = text.rfind("</svg>")
    if start != -1 and end != -1 and end > start:
        return text[start : end + len("</svg>")].strip()
    return None


def render_svg_text_to_png(svg_text: str, output_png_path: Path, scale: float = 3.0) -> None:
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), scale=scale)
    output_png_path.write_bytes(png_bytes)


def _ensure_svg_id(svg_text: str, default_id: str = "main_svg") -> str:
    """Ensures the root <svg> tag has an ID. Returns modified SVG text."""
    try:
        soup = BeautifulSoup(svg_text, "xml") # Use xml parser for SVG
        svg_tag = soup.find("svg")
        if svg_tag:
            if not svg_tag.get("id"):
                svg_tag["id"] = default_id
            return str(soup)
    except Exception as e:
        print(f"Warning: Failed to parse/add ID to SVG: {e}")
    return svg_text


def blackboard_revise_svg_once(
    *,
    original_image_path: Path,
    rendered_image_path: Path,
    current_svg_code: str,
    api_config: dict,
    pid: int,
    analysis_dir: Optional[Path] = None,
) -> Tuple[Optional[str], str]:
    """Return (revised_svg_or_none, raw_model_output)."""

    # Ensure the SVG has a known ID so the agent can reference it
    current_svg_code = _ensure_svg_id(current_svg_code, "main_svg")

    BlackboardOpenAIModel, blackboard_parse_response = _try_import_blackboard_model()
    if BlackboardOpenAIModel is None or blackboard_parse_response is None:
        raise RuntimeError("Blackboard model unavailable (import error or missing deps).")

    os.environ.setdefault("OPENAI_API_KEY", api_config["api_key"])
    os.environ.setdefault("BASE_URL", api_config["base_url"])

    prompt = (
        "You are an SVG code specialist.\n\n"
        "Task: Compare the ORIGINAL image (first) with the CURRENT SVG RENDER (second). \n\n"
        "Then modify the SVG code to better match the original.\n\n"
        
        "**CRITICAL INSTRUCTION ON VISUAL FIDELITY**:\n"
        "- Focus on the global SIMILARITY of the entire image.\n"
        "- If the original image contains **bounding boxes, selection rectangles, or highlights**, treat them as **part of the visual content** to be drawn.\n"
        "- Do NOT interpret a bounding box as a command to 'zoom in' or 'focus only on this area'.\n"
        "- Do NOT ignore the box itself. If there is a red box in the original, your SVG must have a red box in the same location.\n"
        "- Your goal is to make the SVG look **exactly like the original image**, including any annotations or markers present in the original.\n"
        "- **Occluded Objects**: If objects in the original image are occluded or partially visible, reproduce them EXACTLY as they appear (i.e., partially visible).\n\n"
        
        # "**OBSERVATION CHECKLIST**:\n"
        # "When optimize the svg to better match the original image, explicitly check and maintain:\n"
        # "1. **Object Count**: make sure main items' count matches.\n"
        # "2. **Relative Position**: make sure objects in the correct relative positions (left/right, above/below, front/back, far/near)?\n"
        # "3. **Real-world Spatial Relationships**: Maintain the real-world spatial relationships (perspective, depth, scale) depicted in the image.\n"
        # "4. **Text preservation**: If present, ensure text message in original image (like mathematical formulas and so on) are preserved and rendered correctly.\n"
        # "5. **Information Preservation**: Ensure all key information from the original image is represented in the SVG.\n"
        # "...\n\n"
        
        "Process:\n"
        "1. **Analyze & Understand**: First, use <think>...</think> to analyze the visual differences (missing/redundant elements, wrong shapes, incorrect colors, and relative spatial positions/alignment), and understand what each visual element represents in the original image.\n"
        "2. **Plan for Fidelity**: Decide how to represent these better use SVG."
        "3. **Execute Changes**: Use the blackboard tools to MODIFY the initialized SVG on canvas, including but not limited to visually draft the corrected shapes , verify spatial constraints, verify item deletions or additions on initialized canvas. Don't write reasoning text on the canvas.\n"
        "4. **Iterative Refinement**: Perform multi-round iteration to correct the SVG image, verifying each change visually.\n"
        "5. Finally, output the COMPLETE revised SVG code wrapped in <answer>...</answer> tags.\n\n"
        "Rules:\n"
        "- The SVG inside <answer> MUST start with <svg and end with </svg>.\n"
        "- Do NOT change the canvas/viewBox/overall size unless necessary for correctness.\n"
        "- **Modify the content on the CURRENT canvas. Do NOT create a new SVG element. Modify the existing one.**\n"
        "- **When using `insert_element`, ALWAYS use `rootId='main_svg'` to insert inside the existing SVG.**\n\n"
        "CURRENT SVG CODE:\n"
        f"{current_svg_code}\n\n"
        "Now, start your reasoning and revision:" 
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _encode_file_as_data_url(str(original_image_path)), "detail": "high"}},
                {"type": "image_url", "image_url": {"url": _encode_file_as_data_url(str(rendered_image_path)), "detail": "high"}},
            ],
        }
    ]

    model = BlackboardOpenAIModel(
        model_name=api_config["model"],
        temperature=0.2,
        top_p=None,
        api_sleep=0,
        image_detail="high",
    )

    resp = model.generate_roll(
        inputs={"messages": messages},
        data_ori={
            "pid": pid, 
            "question": "svg multiround optimize", 
            "image_list": [str(original_image_path)],
            "save_dir": str(analysis_dir) if analysis_dir else None,
            "initial_svg": current_svg_code
        },
    )

    raw_output = (resp or {}).get("output") or ""
    history = (resp or {}).get("history") or []
    
    # Save full history if available
    if history and analysis_dir:
        import json
        def _json_serial(obj):
            if isinstance(obj, (Path, bytes)):
                return str(obj)
            return str(obj)
            
        try:
            (analysis_dir / f"round_{pid % 100}_history.json").write_text(
                json.dumps(history, default=_json_serial, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
        except Exception as e:
            print(f"  ⚠️ Failed to save history: {e}")

    clean = blackboard_parse_response(raw_output).get("content") or ""
    revised_svg = _extract_svg(clean) or _extract_svg(raw_output) or None

    # Fallback: If model didn't output <answer> with SVG, try to get it from the blackboard state
    if not revised_svg and hasattr(model, 'blackboard') and model.blackboard:
        try:
            # Blackboard state is HTML, so we parse it to find the SVG
            soup_state = BeautifulSoup(model.blackboard.state, "html.parser")
            svg_tag = soup_state.find("svg")
            if svg_tag:
                revised_svg = str(svg_tag)
                print("  [Info] Recovered SVG from Blackboard state.")
        except Exception as e:
            print(f"  [Warning] Failed to recover SVG from Blackboard state: {e}")

    return revised_svg, raw_output


def optimize_folder_multiround(
    *,
    svg_folder: str,
    original_folder: str,
    output_folder: str,
    analysis_folder: str,
    rounds: int,
    api_config: dict,
    sleep_seconds: float,
) -> None:
    svg_root = Path(svg_folder)
    out_root = Path(output_folder)
    analysis_root = Path(analysis_folder)

    out_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    svg_files = list(svg_root.rglob("*.svg"))
    if not svg_files:
        print(f"No SVG files found in {svg_root}")
        return

    print(f"🚀 Multi-round blackboard optimization for {len(svg_files)} SVG files (rounds={rounds})")

    successful = 0
    failed = 0

    for index, svg_file in enumerate(svg_files, 1):
        relative_path = svg_file.relative_to(svg_root)
        original_path = find_matching_image(original_folder, relative_path)

        print(f"\n[{index}/{len(svg_files)}] {relative_path}")
        if not original_path or not original_path.exists():
            print("  ❌ Original image not found")
            failed += 1
            continue

        try:
            current_svg = svg_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ❌ Failed to read SVG: {e}")
            failed += 1
            continue

        # Check if already optimized
        out_path = out_root / relative_path.parent / f"{relative_path.stem}_optimized.svg"
        if out_path.exists():
            print(f"  ✅ Already optimized, skipping: {out_path}")
            successful += 1
            continue

        # Per-file temp render dir under analysis for inspection
        per_file_analysis_dir = analysis_root / relative_path.parent / relative_path.stem
        per_file_analysis_dir.mkdir(parents=True, exist_ok=True)

        last_svg = current_svg
        for r in range(1, rounds + 1):
            rendered_png_path = per_file_analysis_dir / f"round_{r}_render.png"
            try:
                render_svg_text_to_png(last_svg, rendered_png_path, scale=3.0)
            except Exception as e:
                print(f"  ❌ Render failed at round {r}: {e}")
                break

            try:
                revised_svg, raw_output = blackboard_revise_svg_once(
                    original_image_path=original_path,
                    rendered_image_path=rendered_png_path,
                    current_svg_code=last_svg,
                    api_config=api_config,
                    pid=index * 100 + r,
                    analysis_dir=per_file_analysis_dir,
                )
            except Exception as e:
                err_text = traceback.format_exc()
                print(f"  ❌ Blackboard call failed at round {r}: {repr(e)}")
                try:
                    (per_file_analysis_dir / f"round_{r}_error.txt").write_text(err_text, encoding="utf-8")
                except Exception:
                    pass
                break

            (per_file_analysis_dir / f"round_{r}_raw.txt").write_text(raw_output, encoding="utf-8")

            if not revised_svg:
                print(f"  ❌ No SVG extracted at round {r}")
                break

            (per_file_analysis_dir / f"round_{r}_revised.svg").write_text(revised_svg, encoding="utf-8")

            if revised_svg.strip() == last_svg.strip():
                print(f"  ✓ Converged (no change) at round {r}")
                last_svg = revised_svg
                break

            print(f"  ✓ Round {r}: revised")
            last_svg = revised_svg

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        # Save final result
        if _extract_svg(last_svg):
            out_path = out_root / relative_path.parent / f"{relative_path.stem}_optimized.svg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(last_svg, encoding="utf-8")
            print(f"  ✅ Saved: {out_path}")
            successful += 1
        else:
            print("  ❌ Final SVG invalid")
            failed += 1

    print("\n🎯 MULTI-ROUND OPTIMIZATION COMPLETED")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-round per-file SVG revision using blackboard agent.")

    parser.add_argument("--svg-folder", required=True, help="Root directory of SVG files to process.")
    parser.add_argument("--original-folder", required=True, help="Root directory of original reference images.")
    parser.add_argument("--output-folder", required=True, help="Directory to save optimized SVGs.")
    parser.add_argument("--analysis-folder", required=True, help="Directory to save per-round logs/renders.")

    parser.add_argument("--rounds", type=int, default=3, help="Max revision rounds per SVG.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between rounds (rate limiting).")

    parser.add_argument("--base-url", type=str, default=os.getenv("BASE_URL", "http://123.129.219.111:3000/v1"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "gpt-5"))
    parser.add_argument("--max-tokens", type=int, default=16384)

    args = parser.parse_args()

    if not args.api_key:
        parser.error("API key is required. Provide it via --api-key or set OPENAI_API_KEY environment variable.")

    # Set environment variables for the model to use
    os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["BASE_URL"] = args.base_url

    api_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "max_tokens": args.max_tokens,
    }

    optimize_folder_multiround(
        svg_folder=args.svg_folder,
        original_folder=args.original_folder,
        output_folder=args.output_folder,
        analysis_folder=args.analysis_folder,
        rounds=max(1, args.rounds),
        api_config=api_config,
        sleep_seconds=max(0.0, args.sleep_seconds),
    )


if __name__ == "__main__":
    main()
