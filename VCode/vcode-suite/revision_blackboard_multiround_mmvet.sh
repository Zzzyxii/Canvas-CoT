#!/usr/bin/env bash

# ==============================================================================
#      🚀 AUTOMATED SVG OPTIMIZATION PIPELINE (BLACKBOARD MULTI-ROUND) - MM-Vet
# ==============================================================================
#
# Pipeline:
#   1) Multi-round blackboard revision of SVGs against MM-Vet reference images
#   2) Filter optimized SVGs
#   3) Render optimized SVGs to raster images (png/jpg) matching reference ext

set -euo pipefail

########################################
#                CONFIG                #
########################################

# --- Executable ---
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

# --- Script Paths ---
REVISION_SCRIPT="vcode-suite/revision_blackboard_multiround.py"
FILTER_SCRIPT="vcode-suite/filter.py"
RENDER_SCRIPT="vcode-suite/svg_render_img.py"



# --- Pipeline Control ---
MAX_LOOPS=5
ROUNDS_PER_LOOP=1

# --- Inputs ---
# SVGs to optimize (example output folder; switch to your model folder if needed)
INITIAL_SVG_FOLDER="results-mmvet-gpt5/temp_initial_svgs"
# Reference MM-Vet images (used for both optimization + render extension matching)
ORIGINAL_IMAGES_FOLDER="results-mmvet-gpt5/temp_images_subset"
# MM-Vet metadata
MMVET_METADATA="data/mm-vet/mm-vet.json"

# --- Outputs (all under one root) ---
RESULT_ROOT="results-mmvet-gpt5/mm-vet"
WORKSPACE_DIR="$RESULT_ROOT/revision_workspace"

FINAL_OPTIMIZED_SVGS_FOLDER="$RESULT_ROOT/optimized_svgs_blackboard"
FINAL_ANALYSIS_FOLDER="$RESULT_ROOT/visual_analysis_blackboard"
FINAL_RENDERED_IMGS_FOLDER="$RESULT_ROOT/optimized_imgs_blackboard"

# --- API Configuration ---
# You can override any of these via environment variables when running the script.
# Example:
#   BASE_URL=... MODEL=... API_KEY=... bash revision_blackboard_multiround_mmvet.sh
API_KEY=""
BASE_URL=""
MODEL="${MODEL:-gpt-5}"

# Inference API key (required)
API_KEY_INFERENCE="${API_KEY_INFERENCE:-${OPENAI_API_KEY:-$API_KEY}}"

########################################
#              VALIDATION              #
########################################

if [[ -z "$API_KEY_INFERENCE" ]]; then
  echo "Error: missing inference API key. Set API_KEY_INFERENCE or OPENAI_API_KEY." >&2
  exit 1
fi

if [[ ! -d "$INITIAL_SVG_FOLDER" ]]; then
  echo "Error: INITIAL_SVG_FOLDER not found: $INITIAL_SVG_FOLDER" >&2
  exit 1
fi

if [[ ! -d "$ORIGINAL_IMAGES_FOLDER" ]]; then
  echo "Error: ORIGINAL_IMAGES_FOLDER not found: $ORIGINAL_IMAGES_FOLDER" >&2
  exit 1
fi

if [[ ! -f "$MMVET_METADATA" ]]; then
  echo "Error: MMVET_METADATA not found: $MMVET_METADATA" >&2
  exit 1
fi

########################################
#              PIPELINE                #
########################################

echo "Preparing a clean workspace at '$WORKSPACE_DIR' ..."
# rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

echo "Copying initial SVGs from $INITIAL_SVG_FOLDER ..."
# Sync missing files: Copy from INITIAL to WORKSPACE if not in FINAL and not already in WORKSPACE
find "$INITIAL_SVG_FOLDER" -type f -name "*.svg" | while read -r init_svg; do
    rel_path="${init_svg#$INITIAL_SVG_FOLDER/}"
    final_path="$FINAL_OPTIMIZED_SVGS_FOLDER/$rel_path"
    workspace_path="$WORKSPACE_DIR/$rel_path"

    if [[ -f "$final_path" ]]; then
        continue # Already finished
    fi

    if [[ -f "$workspace_path" ]]; then
        continue # Already queued
    fi

    mkdir -p "$(dirname "$workspace_path")"
    cp "$init_svg" "$workspace_path"
done

echo "Pruning workspace: Removing files that are already present in $FINAL_OPTIMIZED_SVGS_FOLDER ..."
if [[ -d "$FINAL_OPTIMIZED_SVGS_FOLDER" ]]; then
    find "$FINAL_OPTIMIZED_SVGS_FOLDER" -type f -name "*.svg" | while read -r final_svg; do
        rel_path="${final_svg#$FINAL_OPTIMIZED_SVGS_FOLDER/}"
        workspace_path="$WORKSPACE_DIR/$rel_path"
        if [[ -f "$workspace_path" ]]; then
            rm -f "$workspace_path"
        fi
    done
fi

mkdir -p "$FINAL_OPTIMIZED_SVGS_FOLDER" "$FINAL_ANALYSIS_FOLDER" "$FINAL_RENDERED_IMGS_FOLDER"

echo "🚀 Starting Blackboard SVG Optimization Pipeline (MM-Vet)"
echo "  - Max loops: $MAX_LOOPS"
echo "  - Rounds per file: $ROUNDS_PER_LOOP"
echo "  - Model: $MODEL"
echo "  - Base URL: $BASE_URL"
echo "======================================================================"

# --- Resume Logic ---
REVISED_DIR_LOOP="$WORKSPACE_DIR/revised_loop_outputs"
ANALYSIS_DIR_LOOP="$WORKSPACE_DIR/analysis_loop_outputs"

if [[ -d "$REVISED_DIR_LOOP" ]]; then
    echo "🔄 Resume: Found existing revised_loop_outputs. Harvesting successful optimizations..."
    find "$REVISED_DIR_LOOP" -type f -name "*_optimized.svg" | while read -r optimized_svg_path; do
        relative_optimized_path="${optimized_svg_path#$REVISED_DIR_LOOP/}"
        original_relative_path=$(echo "$relative_optimized_path" | sed 's/_optimized\.svg/\.svg/')
        original_relative_stem="${original_relative_path%.*}"

        final_svg_path="$FINAL_OPTIMIZED_SVGS_FOLDER/$original_relative_path"
        mkdir -p "$(dirname "$final_svg_path")"

        mv "$optimized_svg_path" "$final_svg_path"

        rm -f "$WORKSPACE_DIR/$original_relative_path" 2>/dev/null || true
        find "$WORKSPACE_DIR/${original_relative_stem%/*}" -maxdepth 1 \
          -name "${original_relative_stem##*/}*" -not -name "*.svg" -exec rm {} \; 2>/dev/null || true
    done
fi

if [[ -d "$ANALYSIS_DIR_LOOP" ]]; then
    echo "🔄 Resume: Found existing analysis_loop_outputs. Skipping already processed files..."
    # Move logs to final analysis folder
    cp -a "$ANALYSIS_DIR_LOOP/." "$FINAL_ANALYSIS_FOLDER/" 2>/dev/null || true
    
    # Remove corresponding SVGs from workspace to avoid re-processing
    for dir in "$ANALYSIS_DIR_LOOP"/*; do
        if [[ -d "$dir" ]]; then
            base=$(basename "$dir")
            # Assuming folder name matches SVG stem
            if [[ -f "$WORKSPACE_DIR/$base.svg" ]]; then
                echo "   -> Skipping $base.svg (already in analysis outputs)"
                rm -f "$WORKSPACE_DIR/$base.svg"
            fi
        fi
    done
fi
# --------------------

for (( i=1; i<=MAX_LOOPS; i++ )); do
  if ! find "$WORKSPACE_DIR" -type f -name "*.svg" -print -quit | grep -q .; then
    echo -e "\n🎉 Success! The workspace is empty. All files processed."
    break
  fi

  NUM_FILES=$(find "$WORKSPACE_DIR" -type f -name "*.svg" | wc -l)
  echo -e "\n🌀 Starting Loop $i/$MAX_LOOPS: Processing $NUM_FILES remaining files..."
  echo "----------------------------------------------------------------"

  REVISED_DIR_LOOP="$WORKSPACE_DIR/revised_loop_outputs"
  ANALYSIS_DIR_LOOP="$WORKSPACE_DIR/analysis_loop_outputs"
  mkdir -p "$REVISED_DIR_LOOP" "$ANALYSIS_DIR_LOOP"

  echo "   [1/3] Running Blackboard Revision recursively on the entire workspace..."
  "$PYTHON_EXECUTABLE" "$REVISION_SCRIPT" \
    --api-key "$API_KEY_INFERENCE" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --svg-folder "$WORKSPACE_DIR" \
    --original-folder "$ORIGINAL_IMAGES_FOLDER" \
    --output-folder "$REVISED_DIR_LOOP" \
    --analysis-folder "$ANALYSIS_DIR_LOOP" \
    --rounds "$ROUNDS_PER_LOOP" 2>&1 | tee -a "$ANALYSIS_DIR_LOOP/pipeline_run.log"

  echo -e "\n   [2/3] Cleaning all successfully revised SVGs (recursively)..."
  "$PYTHON_EXECUTABLE" "$FILTER_SCRIPT" --svg-folder "$REVISED_DIR_LOOP"

  echo -e "\n   [3/3] Sorting results and updating workspace..."
  SUCCESS_COUNT=0

  if [[ -d "$ANALYSIS_DIR_LOOP" ]]; then
    echo "   -> Moving analysis logs to $FINAL_ANALYSIS_FOLDER"
    cp -a "$ANALYSIS_DIR_LOOP/." "$FINAL_ANALYSIS_FOLDER/" 2>/dev/null || true
  fi

  find "$REVISED_DIR_LOOP" -type f -name "*_optimized.svg" | while read -r optimized_svg_path; do
    ((SUCCESS_COUNT++))

    relative_optimized_path="${optimized_svg_path#$REVISED_DIR_LOOP/}"
    original_relative_path=$(echo "$relative_optimized_path" | sed 's/_optimized\.svg/\.svg/')
    original_relative_stem="${original_relative_path%.*}"

    final_svg_path="$FINAL_OPTIMIZED_SVGS_FOLDER/$original_relative_path"
    mkdir -p "$(dirname "$final_svg_path")"

    mv "$optimized_svg_path" "$final_svg_path"

    rm -f "$WORKSPACE_DIR/$original_relative_path" 2>/dev/null || true
    find "$WORKSPACE_DIR/${original_relative_stem%/*}" -maxdepth 1 \
      -name "${original_relative_stem##*/}*" -not -name "*.svg" -exec rm {} \; 2>/dev/null || true
  done

  FAILURE_COUNT=$((NUM_FILES - SUCCESS_COUNT))
  echo "   -> Loop $i Summary: $SUCCESS_COUNT succeeded, $FAILURE_COUNT failed and will be retried."

  rm -rf "$REVISED_DIR_LOOP" "$ANALYSIS_DIR_LOOP"

  if [[ $i -eq $MAX_LOOPS ]]; then
    echo -e "\n🏁 Pipeline finished after reaching the maximum of $MAX_LOOPS loops."
  fi
done

echo -e "\n🎨 Rendering final optimized SVGs to images..."
if [[ -n "$(ls -A "$FINAL_OPTIMIZED_SVGS_FOLDER" 2>/dev/null || true)" ]]; then
  "$PYTHON_EXECUTABLE" "$RENDER_SCRIPT" \
    --svg-input-dir "$FINAL_OPTIMIZED_SVGS_FOLDER" \
    --image-output-dir "$FINAL_RENDERED_IMGS_FOLDER" \
    --reference-dir "$ORIGINAL_IMAGES_FOLDER"
else
  echo "   -> No SVGs to render, skipping."
fi

echo -e "\n🩹 Fallback render: using per-round revised SVGs for any missing images..."
if [[ -n "$(ls -A "$FINAL_OPTIMIZED_SVGS_FOLDER" 2>/dev/null || true)" ]]; then
  FALLBACK_SVG_DIR="$(mktemp -d)"
  missing_count=0

  while IFS= read -r svg_path; do
    rel_path="${svg_path#$FINAL_OPTIMIZED_SVGS_FOLDER/}"
    stem="$(basename "$rel_path" .svg)"
    rel_dir="$(dirname "$rel_path")"

    out_ext="png"
    if [[ -f "$ORIGINAL_IMAGES_FOLDER/$rel_dir/$stem.jpg" || -f "$ORIGINAL_IMAGES_FOLDER/$rel_dir/$stem.jpeg" ]]; then
      out_ext="jpg"
    fi

    expected_img="$FINAL_RENDERED_IMGS_FOLDER/$rel_dir/$stem.$out_ext"
    if [[ -f "$expected_img" ]]; then
      continue
    fi

    analysis_dir="$FINAL_ANALYSIS_FOLDER/$rel_dir/$stem"
    if [[ ! -d "$analysis_dir" ]]; then
      continue
    fi

    # Prefer the latest round_*_revised.svg; fall back to round_1_revised.svg.
    revised_svg=""
    revised_svg="$(ls -1 "$analysis_dir"/round_*_revised.svg 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -z "$revised_svg" || ! -f "$revised_svg" ]]; then
      continue
    fi

    mkdir -p "$FALLBACK_SVG_DIR/$rel_dir"
    cp -f "$revised_svg" "$FALLBACK_SVG_DIR/$rel_dir/$stem.svg"

    # Also replace the final optimized SVG with the renderable revised SVG to stabilize future runs.
    cp -f "$revised_svg" "$FINAL_OPTIMIZED_SVGS_FOLDER/$rel_dir/$stem.svg"

    missing_count=$((missing_count + 1))
  done < <(find "$FINAL_OPTIMIZED_SVGS_FOLDER" -type f -name "*.svg" | sort)

  if (( missing_count > 0 )); then
    echo "   -> Found $missing_count missing images; rendering fallbacks..."
    "$PYTHON_EXECUTABLE" "$RENDER_SCRIPT" \
      --svg-input-dir "$FALLBACK_SVG_DIR" \
      --image-output-dir "$FINAL_RENDERED_IMGS_FOLDER" \
      --reference-dir "$ORIGINAL_IMAGES_FOLDER" || true
  else
    echo "   -> No missing images detected."
  fi

  rm -rf "$FALLBACK_SVG_DIR"
fi

echo -e "\n✅ Pipeline complete."
echo "   Optimized SVGs:     $FINAL_OPTIMIZED_SVGS_FOLDER"
echo "   Rendered images:    $FINAL_RENDERED_IMGS_FOLDER"
echo "   Workspace leftover: $WORKSPACE_DIR"
