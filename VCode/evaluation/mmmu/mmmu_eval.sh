#!/bin/bash
set -euo pipefail

# ================= Config =================
src_json_root="/m2/slz/zyx/MM/VCode/data/mmmu/mmmu_dev_processed_single_img_subset"
cache_dir="/m2/slz/zyx/MM/VCode/data/mmmu"

data_dir="/m2/slz/zyx/MM/VCode/results/mmmu/optimized_imgs_blackboard"
output_path="/m2/slz/zyx/MM/VCode/results/mmmu/eval_results_blackboard"

base_url="http://123.129.219.111:3000/v1"
api_key="sk-7yk0ue1nwUfAnrrgCYMI7XUdu19It891XzhMDhEzOYBhav9d"
model="gpt-4o-mini-2024-07-18"

# Script Directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "================= Step 1 ================="
python "$SCRIPT_DIR/prepare_output_jsons.py" \
  --src_json_root "$src_json_root" \
  --dst_root "$data_dir" \
  --overwrite

echo "================= Step 2 ================="
python "$SCRIPT_DIR/run_api_local_images.py" \
  --data_path "$data_dir" \
  --output_path "$output_path" \
  --base_url "$base_url" \
  --api_key "$api_key" \
  --model "$model" \
  --cache_dir "$cache_dir"

echo "================= Step 3 ================="
python "$SCRIPT_DIR/split_outputs_by_subject.py" \
  --input_json "$output_path/Unknown/output.json" \
  --output_root "$output_path"

echo "================= Step 4 ================="
python "$SCRIPT_DIR/main_parse_and_eval.py" \
  --path "$output_path" \
  --subject ALL

echo "================= Step 5 ================="
python "$SCRIPT_DIR/save_scores.py" \
  --input_path "$output_path" \
  --output_path "$output_path/scores.csv"

echo "✅ Entire process is complete! The final score has been saved to: $output_path/scores.csv"
