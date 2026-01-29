#!/bin/bash

# ==============================================================================
#      🚀 AUTOMATED SVG OPTIMIZATION PIPELINE (BLACKBOARD MULTI-ROUND) - MMMU
# ==============================================================================
#
# This script orchestrates the multi-round blackboard optimization process for MMMU.
#
# ========================== CONFIGURATION =====================================

# --- Python Script Paths ---
REVISION_SCRIPT="vcode-suite/revision_blackboard_multiround.py"
FILTER_SCRIPT="vcode-suite/filter.py"
RENDER_SCRIPT="vcode-suite/svg_render_img.py"

# --- Pipeline Control ---
MAX_LOOPS=5
ROUNDS_PER_LOOP=1  # How many blackboard rounds per file per loop

# --- Workspace Configuration ---
WORKSPACE_DIR="results-mmmu-gpt5/mmmu/revision_workspace"

# --- Input Folders (Initial State) ---
# Using the example results as the starting point for optimization
INITIAL_SVG_FOLDER="..."
ORIGINAL_IMAGES_FOLDER="data/mmmu/mmmu_dev_processed_single_img_subset"

# --- Final Output Folders ---
FINAL_OPTIMIZED_SVGS_FOLDER="results-mmmu-gpt5/mmmu/optimized_svgs_blackboard"
FINAL_ANALYSIS_FOLDER="results-mmmu-gpt5/mmmu/visual_analysis_blackboard"
FINAL_RENDERED_IMGS_FOLDER="results-mmmu-gpt5/mmmu/optimized_imgs_blackboard"

# --- API Configuration ---
API_KEY=""
BASE_URL=""
MODEL="gpt-5"

# ======================= SCRIPT EXECUTION (DO NOT EDIT) =======================

# --- 0. Initialization: Replicate Input Structure in Workspace ---
echo "Preparing a clean workspace at '$WORKSPACE_DIR' with original directory structure..."
rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

# Copy contents while preserving structure
if [ -d "$INITIAL_SVG_FOLDER" ]; then
    echo "Copying initial SVGs from $INITIAL_SVG_FOLDER..."
    cp -a "$INITIAL_SVG_FOLDER/." "$WORKSPACE_DIR/" 2>/dev/null || true
else
    echo "Error: Initial SVG folder '$INITIAL_SVG_FOLDER' does not exist!"
    exit 1
fi

# Prepare final output directories
mkdir -p "$FINAL_OPTIMIZED_SVGS_FOLDER" "$FINAL_ANALYSIS_FOLDER"

echo "🚀 Starting Blackboard SVG Optimization Pipeline. Max loops: $MAX_LOOPS, Rounds per file: $ROUNDS_PER_LOOP"
echo "======================================================================"

# --- Main Loop ---
for (( i=1; i<=$MAX_LOOPS; i++ )); do
    # Check if there are any svg files left in the entire workspace tree
    if ! find "$WORKSPACE_DIR" -type f -name "*.svg" -print -quit | grep -q .; then
        echo -e "\n🎉 Success! The workspace is empty. All files processed."
        break
    fi
    
    NUM_FILES=$(find "$WORKSPACE_DIR" -type f -name "*.svg" | wc -l)

    echo -e "\n🌀 Starting Loop $i/$MAX_LOOPS: Processing $NUM_FILES remaining files..."
    echo "----------------------------------------------------------------"

    # Temporary directories for this loop's aggregated results
    REVISED_DIR_LOOP="$WORKSPACE_DIR/revised_loop_outputs"
    ANALYSIS_DIR_LOOP="$WORKSPACE_DIR/analysis_loop_outputs"
    mkdir -p "$REVISED_DIR_LOOP" "$ANALYSIS_DIR_LOOP"

    # --- Step 1: Revision (Blackboard Multi-round) ---
    echo "   [1/3] Running Blackboard Revision recursively on the entire workspace..."
    python "$REVISION_SCRIPT" \
        --api-key "$API_KEY" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --svg-folder "$WORKSPACE_DIR" \
        --original-folder "$ORIGINAL_IMAGES_FOLDER" \
        --output-folder "$REVISED_DIR_LOOP" \
        --analysis-folder "$ANALYSIS_DIR_LOOP" \
        --rounds "$ROUNDS_PER_LOOP" 2>&1 | tee -a "$ANALYSIS_DIR_LOOP/pipeline_run.log"

    # --- Step 2: Filter ---
    echo -e "\n   [2/3] Cleaning all successfully revised SVGs (recursively)..."
    python "$FILTER_SCRIPT" --svg-folder "$REVISED_DIR_LOOP"

    # --- Step 3: Sorting and Moving ---
    echo -e "\n   [3/3] Sorting results and updating workspace..."
    SUCCESS_COUNT=0
    
    # Move ALL analysis results (both success and failure) to final analysis folder
    # This ensures we keep the intermediate blackboard rounds like run_full_benchmark.sh
    if [ -d "$ANALYSIS_DIR_LOOP" ]; then
        echo "   -> Moving analysis logs to $FINAL_ANALYSIS_FOLDER"
        cp -a "$ANALYSIS_DIR_LOOP/." "$FINAL_ANALYSIS_FOLDER/" 2>/dev/null || true
    fi

    # Note: revision_blackboard_multiround.py outputs files with _optimized.svg suffix
    find "$REVISED_DIR_LOOP" -type f -name "*_optimized.svg" | while read -r optimized_svg_path; do
        ((SUCCESS_COUNT++))
        relative_optimized_path="${optimized_svg_path#$REVISED_DIR_LOOP/}"
        # Remove _optimized suffix to get original relative path structure
        original_relative_path=$(echo "$relative_optimized_path" | sed 's/_optimized\.svg/\.svg/')
        original_relative_stem="${original_relative_path%.*}"
        
        final_svg_path="$FINAL_OPTIMIZED_SVGS_FOLDER/$original_relative_path"
        
        mkdir -p "$(dirname "$final_svg_path")"
        
        # Move the optimized SVG
        mv "$optimized_svg_path" "$final_svg_path"
        
        # Remove the original file from workspace so it's not processed in the next loop
        rm "$WORKSPACE_DIR/$original_relative_path"
        # Clean up related files in workspace if any
        find "$WORKSPACE_DIR/${original_relative_stem%/*}" -maxdepth 1 -name "${original_relative_stem##*/}*" -not -name "*.svg" -exec rm {} \; 2>/dev/null || true
    done

    FAILURE_COUNT=$((NUM_FILES - SUCCESS_COUNT))
    echo "   -> Loop $i Summary: $SUCCESS_COUNT succeeded, $FAILURE_COUNT failed and will be retried."

    rm -rf "$REVISED_DIR_LOOP" "$ANALYSIS_DIR_LOOP"

    if [[ $i -eq $MAX_LOOPS ]]; then
        echo -e "\n🏁 Pipeline finished after reaching the maximum of $MAX_LOOPS loops."
    fi
done

# --- Final Rendering Step ---
echo -e "\n🎨 Rendering final optimized SVGs to images..."
if [ -n "$(ls -A $FINAL_OPTIMIZED_SVGS_FOLDER 2>/dev/null)" ]; then
    python "$RENDER_SCRIPT" \
        --svg-input-dir "$FINAL_OPTIMIZED_SVGS_FOLDER" \
        --image-output-dir "$FINAL_RENDERED_IMGS_FOLDER" \
        --reference-dir "$ORIGINAL_IMAGES_FOLDER"
else
    echo "   -> No SVGs to render, skipping."
fi


# --- Final Report ---
echo -e "\n✅ Pipeline complete."
echo "   Final successful results are in:"
echo "   - Optimized SVGs: $FINAL_OPTIMIZED_SVGS_FOLDER"
echo "   - Analysis Reports: $FINAL_ANALYSIS_FOLDER"
echo "   - Rendered Images: $FINAL_RENDERED_IMGS_FOLDER"
echo ""
echo "   Please check the '$WORKSPACE_DIR' directory for any files that failed to process."
