#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=""
# ==============================================================================
#      📊 AUTOMATED EVALUATION PIPELINE (BLACKBOARD)
# ==============================================================================

# --- Configuration ---
# Paths
PROJECT_ROOT=""
RESULTS_ROOT="$PROJECT_ROOT/results"
EVAL_ROOT="$RESULTS_ROOT/eval_final_blackboard"
VQA_EVAL_ROOT="$RESULTS_ROOT/eval_vqa_blackboard"

# Source Data (Blackboard Outputs)
SRC_SVGS="$RESULTS_ROOT/optimized_svgs_blackboard"
SRC_IMGS="$RESULTS_ROOT/optimized_imgs_blackboard"

# Reference Data (Ground Truth)
# Adjust this if you are evaluating a different subset (e.g., count, relation)
REF_IMGS_DIR="$PROJECT_ROOT/data/cv-bench/2D/spatial/images"

# API Config (for VQA Evaluation)
API_KEY=""
BASE_URL=""
MODEL_NAME="${MODEL:-gpt-4o-mini}"

# ==============================================================================

echo "🚀 Starting Evaluation Pipeline..."

# --- 1. Prepare Directory Structure for Metrics.py ---
echo "📂 Setting up evaluation directories at $EVAL_ROOT..."
rm -rf "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT"

# Link generated files to the structure expected by metrics.py
# metrics.py expects: folder2/generated_svgs and folder2/generated_imgs
ln -sf "$SRC_SVGS" "$EVAL_ROOT/generated_svgs"
ln -sf "$SRC_IMGS" "$EVAL_ROOT/generated_imgs"

# --- 2. Run Visual Similarity & Token Metrics ---
echo "📉 Running SigLIP Similarity & Token Metrics..."
python "$PROJECT_ROOT/evaluation/metrics.py" \
  --folder1 "$REF_IMGS_DIR" \
  --folder2 "$EVAL_ROOT"

# --- 3. Prepare Directory Structure for VQA Evaluation ---
echo "📂 Setting up VQA evaluation directories at $VQA_EVAL_ROOT..."
rm -rf "$VQA_EVAL_ROOT"
mkdir -p "$VQA_EVAL_ROOT/generated_imgs/2D/spatial/images"

# Flatten/Link images for VQA script
# The VQA script often expects images in a specific structure matching the dataset
# Here we link our optimized images to where the VQA script looks for them
echo "🔗 Linking images for VQA..."
# We assume the optimized images have the same filenames as the originals
find "$SRC_IMGS" -name "*.png" -o -name "*.jpg" | while read img; do
    ln -sf "$img" "$VQA_EVAL_ROOT/generated_imgs/2D/spatial/images/$(basename "$img")"
done

# Copy annotations.jsonl so the eval script can find the questions
echo "📋 Copying annotations..."
cp "$PROJECT_ROOT/data/cv-bench/2D/spatial/annotations.jsonl" "$VQA_EVAL_ROOT/generated_imgs/2D/spatial/"

# --- 4. Run VQA Evaluation (ADE20K / Spatial) ---
echo "🤖 Running VQA Evaluation (ADE/Spatial)..."
python "$PROJECT_ROOT/evaluation/cv-bench/eval/ade/api_eval_new.py" \
    --data_root "$VQA_EVAL_ROOT/generated_imgs" \
    --answers_file "$VQA_EVAL_ROOT/answers_ade.jsonl" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME"

# --- 5. Run VQA Evaluation (COCO / Spatial) ---
echo "🤖 Running VQA Evaluation (COCO/Spatial)..."
python "$PROJECT_ROOT/evaluation/cv-bench/eval/coco/api_eval_new.py" \
    --data_root "$VQA_EVAL_ROOT/generated_imgs" \
    --answers_file "$VQA_EVAL_ROOT/answers_coco.jsonl" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME"

# --- 6. Merge Results & Calculate Scores ---
echo "📊 Merging Results & Calculating Scores..."
python "$PROJECT_ROOT/evaluation/cv-bench/merge_results_final.py" \
    --base_dir "$VQA_EVAL_ROOT" \
    --input_files "answers_ade.jsonl" "answers_coco.jsonl" \
    --output_csv_file "$VQA_EVAL_ROOT/merged_results.csv"

python "$PROJECT_ROOT/evaluation/cv-bench/calculate_score.py" \
    --input_csv "$VQA_EVAL_ROOT/merged_results.csv" \
    --output_csv "$VQA_EVAL_ROOT/final_scores.csv"

echo "✅ Evaluation Pipeline Completed!"
echo "   - Metrics Report: See console output above"
echo "   - VQA Answers (ADE): $VQA_EVAL_ROOT/answers_ade.jsonl"
echo "   - VQA Answers (COCO): $VQA_EVAL_ROOT/answers_coco.jsonl"
echo "   - Final Scores: $VQA_EVAL_ROOT/final_scores.csv"
