#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
# ==============================================================================
#      📊 COMBINED 2D EVALUATION PIPELINE (COUNT + SPATIAL)
# ==============================================================================

# --- Configuration ---
PROJECT_ROOT=""
RESULTS_ROOT="$PROJECT_ROOT/results"
EVAL_ROOT="$RESULTS_ROOT/eval_2d_combined_blackboard"
VQA_EVAL_ROOT="$RESULTS_ROOT/eval_vqa_2d_combined"

# Source 1: Count Task
SRC_COUNT_IMGS="$PROJECT_ROOT/results/optimized_imgs_blackboard"
SRC_COUNT_SVGS="$PROJECT_ROOT/results/optimized_svgs_blackboard"
REF_COUNT_DIR="$PROJECT_ROOT/data/cv-bench/2D/count"

# Source 2: Spatial Task
SRC_SPATIAL_IMGS="$PROJECT_ROOT/results-cvbench2D-spatial/optimized_imgs_blackboard"
SRC_SPATIAL_SVGS="$PROJECT_ROOT/results-cvbench2D-spatial/optimized_svgs_blackboard"
REF_SPATIAL_DIR="$PROJECT_ROOT/data/cv-bench/2D/spatial"

# Reference Root (for Metrics matching)
REF_ROOT_2D="$PROJECT_ROOT/data/cv-bench/2D"

# API Config
API_KEY=""
BASE_URL=""
MODEL_NAME="${MODEL:-gpt-4o-mini}"

# ==============================================================================

echo "🚀 Starting Combined 2D Evaluation Pipeline..."

# --- 1. Prepare Directory Structure ---
echo "📂 Setting up evaluation directories..."
rm -rf "$EVAL_ROOT" "$VQA_EVAL_ROOT"
mkdir -p "$EVAL_ROOT/generated_imgs/count/images"
mkdir -p "$EVAL_ROOT/generated_imgs/spatial/images"
mkdir -p "$EVAL_ROOT/generated_svgs/count/images"
mkdir -p "$EVAL_ROOT/generated_svgs/spatial/images"

mkdir -p "$VQA_EVAL_ROOT/generated_imgs/count/images"
mkdir -p "$VQA_EVAL_ROOT/generated_imgs/spatial/images"

# --- 2. Link Files (Images & SVGs) ---
echo "🔗 Linking Count files..."
# Use find to link only files, avoiding errors if directories are empty or contain subdirs
find "$SRC_COUNT_IMGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) -exec ln -sf {} "$EVAL_ROOT/generated_imgs/count/images/" \;
find "$SRC_COUNT_SVGS" -maxdepth 1 -type f -name "*.svg" -exec ln -sf {} "$EVAL_ROOT/generated_svgs/count/images/" \;
# Link for VQA
find "$SRC_COUNT_IMGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) -exec ln -sf {} "$VQA_EVAL_ROOT/generated_imgs/count/images/" \;

echo "🔗 Linking Spatial files..."
find "$SRC_SPATIAL_IMGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) -exec ln -sf {} "$EVAL_ROOT/generated_imgs/spatial/images/" \;
find "$SRC_SPATIAL_SVGS" -maxdepth 1 -type f -name "*.svg" -exec ln -sf {} "$EVAL_ROOT/generated_svgs/spatial/images/" \;
# Link for VQA
find "$SRC_SPATIAL_IMGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) -exec ln -sf {} "$VQA_EVAL_ROOT/generated_imgs/spatial/images/" \;

# --- 3. Copy Annotations ---
echo "📋 Copying annotations..."
cp "$REF_COUNT_DIR/annotations.jsonl" "$VQA_EVAL_ROOT/generated_imgs/count/"
cp "$REF_SPATIAL_DIR/annotations.jsonl" "$VQA_EVAL_ROOT/generated_imgs/spatial/"

# --- 4. Run Visual Similarity & Token Metrics ---
echo "📉 Running SigLIP Similarity & Token Metrics (Combined)..."
# Note: metrics.py expects folder2 to contain 'generated_imgs' and 'generated_svgs' subfolders
# So we pass EVAL_ROOT as folder2
python "$PROJECT_ROOT/evaluation/metrics.py" \
  --folder1 "$REF_ROOT_2D" \
  --folder2 "$EVAL_ROOT"

# --- 5. Run VQA Evaluation (ADE20K) ---
echo "🤖 Running VQA Evaluation (ADE20K - Combined)..."
# This will recursively find annotations.jsonl in both count and spatial folders
python "$PROJECT_ROOT/evaluation/cv-bench/eval/ade/api_eval_new.py" \
    --data_root "$VQA_EVAL_ROOT/generated_imgs" \
    --answers_file "$VQA_EVAL_ROOT/answers_ade.jsonl" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME"

# --- 6. Run VQA Evaluation (COCO) ---
echo "🤖 Running VQA Evaluation (COCO - Combined)..."
python "$PROJECT_ROOT/evaluation/cv-bench/eval/coco/api_eval_new.py" \
    --data_root "$VQA_EVAL_ROOT/generated_imgs" \
    --answers_file "$VQA_EVAL_ROOT/answers_coco.jsonl" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME"

# --- 7. Merge Results & Calculate Scores ---
echo "📊 Merging Results & Calculating Scores..."
python "$PROJECT_ROOT/evaluation/cv-bench/merge_results_final.py" \
    --base_dir "$VQA_EVAL_ROOT" \
    --input_files "answers_ade.jsonl" "answers_coco.jsonl" \
    --output_csv_file "$VQA_EVAL_ROOT/merged_results.csv"

python "$PROJECT_ROOT/evaluation/cv-bench/calculate_score.py" \
    --input_csv "$VQA_EVAL_ROOT/merged_results.csv" \
    --output_csv "$VQA_EVAL_ROOT/final_scores.csv"

echo "✅ Combined Evaluation Completed!"
echo "   - Final Scores: $VQA_EVAL_ROOT/final_scores.csv"
