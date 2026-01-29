#!/usr/bin/env bash
set -euo pipefail

########################################
#           CONFIG                     #
########################################

# --- API and Model Config ---
MODEL="gpt-5"
MODEL_TAG="GPT5-native"  # Used for output folder naming
BASE_URL=
API_KEY=

# --- Scripts and Tools ---
PYTHON="python3"
SVG_SCRIPT="vcode-suite/img2svg.py"
FILTER_SCRIPT="vcode-suite/filter.py"
RENDER_SCRIPT="vcode-suite/svg_render_img.py"

# --- Automation Config ---
MAX_ROUNDS=5

# --- Benchmark Definitions ---
# Order: BenchName InputPath
BENCHMARKS=(
    "cv-bench" "data/cv-bench"
    "mm-vet"   "data/mm-vet/images"
    "mmmu"     "data/mmmu/mmmu_dev_processed_single_img_subset"
)

########################################
#          HELPER FUNCTIONS            #
########################################

if [[ -z "$API_KEY" || "$API_KEY" == "YOUR_API_KEY_HERE" ]]; then
  echo "Error! API_KEY is not set. Please edit the script to provide a valid API key."
  exit 1
fi

collect_relative_paths () {
  local dir="$1"
  local file_type="$2" 
  if [[ ! -d "$dir" ]]; then echo ""; return; fi
  
  if [[ "$file_type" == "images" ]]; then
    find "$dir" -type f \
      | awk 'tolower($0) ~ /\.(png|jpg|jpeg|bmp|gif|webp)$/ {print}' \
      | sed "s|^$dir/||" | sed -E 's/\.[^.]+$//' | sort -u
  else
    find "$dir" -type f -name "*.svg" \
      | sed "s|^$dir/||" | sed -E 's/\.[^.]+$//' | sort -u
  fi
}

prepare_work_input () {
  local src_dir="$1"; local out_dir="$2"; shift 2; local missing_list=("$@")
  mkdir -p "$out_dir"; find "$out_dir" -mindepth 1 -exec rm -rf {} +
  for rel_path in "${missing_list[@]}"; do
    found_any=0; target_dir="$out_dir/$(dirname "$rel_path")"; mkdir -p "$target_dir"
    for ext in png jpg jpeg bmp gif webp; do
      src_file="$src_dir/$rel_path.$ext"
      if [[ -f "$src_file" ]]; then
        cp -f "$src_file" "$out_dir/$rel_path.$ext"; found_any=1
      fi
    done
  done
}

########################################
#           PIPELINE STEPS             #
########################################

run_svg_generation_loop() {
    echo "================================================="
    echo "  STEP 1: GENERATING SVGS FROM IMAGES"
    echo "================================================="
    
    local round=0
    local prev_missing_count=999999
    
    mkdir -p "$SVG_OUTPUT_DIR"
    mkdir -p "$WORK_INPUT_DIR"

    while (( round < MAX_ROUNDS )); do
        round=$((round+1))
        echo ""
        echo "--- SVG Generation: Round $round ---"

        mapfile -t all_image_paths < <(collect_relative_paths "$IMAGES_DIR" "images")
        mapfile -t svg_paths < <(collect_relative_paths "$SVG_OUTPUT_DIR" "svgs")

        echo "Total source images found: ${#all_image_paths[@]}"
        echo "Total SVGs generated so far: ${#svg_paths[@]}"

        local tmp_all; tmp_all=$(mktemp)
        local tmp_svg; tmp_svg=$(mktemp)
        printf "%s\n" "${all_image_paths[@]}" > "$tmp_all"
        printf "%s\n" "${svg_paths[@]}" > "$tmp_svg"
        mapfile -t missing_paths < <(comm -23 "$tmp_all" "$tmp_svg")
        rm -f "$tmp_all" "$tmp_svg"

        local missing_count=${#missing_paths[@]}
        if (( missing_count == 0 )); then
            echo "All SVGs have been generated. Proceeding to the next step."
            return 0
        fi

        echo "Found $missing_count missing SVGs to generate."

        if (( missing_count >= prev_missing_count )); then
            echo "Warning! No progress in SVG generation. Stopping this step."
            break
        fi
        prev_missing_count=$missing_count

        echo "Preparing missing images in '$WORK_INPUT_DIR'..."
        prepare_work_input "$IMAGES_DIR" "$WORK_INPUT_DIR" "${missing_paths[@]}"
        
        echo "Calling $(basename ${SVG_SCRIPT}) script..."
        "$PYTHON" "$SVG_SCRIPT" \
            "$WORK_INPUT_DIR" \
            "$SVG_OUTPUT_DIR" \
            --model "$MODEL" \
            --base-url "$BASE_URL" \
            --api-key "$API_KEY" \
            --max-workers 32

        sleep 3
    done

    if (( missing_count > 0 )); then
        echo "Reached max rounds or generation stalled. Some SVGs may still be missing."
    fi
}

run_svg_filter() {
    echo ""
    echo "================================================="
    echo "  STEP 2: CLEANING AND FILTERING SVGS"
    echo "================================================="
    
    if [ ! -d "$SVG_OUTPUT_DIR" ] || [ -z "$(ls -A "$SVG_OUTPUT_DIR")" ]; then
        echo "Warning! SVG output directory '$SVG_OUTPUT_DIR' is empty or does not exist. Skipping filtering."
        return 1
    fi
    
    echo "Calling filter.py to process all files in '$SVG_OUTPUT_DIR'..."
    "$PYTHON" "$FILTER_SCRIPT" --svg-folder "$SVG_OUTPUT_DIR"
    
    echo "SVG filtering complete."
}

run_image_rendering() {
    echo ""
    echo "================================================="
    echo "  STEP 3: RENDERING SVGS TO IMAGES"
    echo "================================================="

    if [ ! -d "$SVG_OUTPUT_DIR" ] || [ -z "$(ls -A "$SVG_OUTPUT_DIR")" ]; then
        echo "Warning! SVG output directory '$SVG_OUTPUT_DIR' is empty. Skipping rendering."
        return 1
    fi

    echo "Calling render_img.py..."
    echo "  - SVG Source: $SVG_OUTPUT_DIR"
    echo "  - Image Output: $RENDERED_IMG_OUT_DIR"
    echo "  - Reference: $IMAGES_DIR"

    mkdir -p "$RENDERED_IMG_OUT_DIR"
    
    "$PYTHON" "$RENDER_SCRIPT" \
        --svg-input-dir "$SVG_OUTPUT_DIR" \
        --image-output-dir "$RENDERED_IMG_OUT_DIR" \
        --reference-dir "$IMAGES_DIR" \
        --max-workers 32

    echo "Image rendering complete."
}


########################################
#              MAIN EXECUTION          #
########################################

main() {
    echo "STARTING FULL AUTOMATION PIPELINE (MULTI-BENCHMARK PARALLEL)"
    
    local num_benchs=${#BENCHMARKS[@]}
    
    # Iterate in pairs (Name, Path)
    for (( i=0; i<num_benchs; i+=2 )); do
        local bench_name="${BENCHMARKS[i]}"
        local input_path="${BENCHMARKS[i+1]}"
        
        (
            echo ""
            echo "#################################################"
            echo "  STARTING BENCHMARK: $bench_name"
            echo "  INPUT: $input_path"
            echo "#################################################"
            
            # Set Global Variables for the steps to use
            export IMAGES_DIR="$input_path"
            export SVG_OUTPUT_DIR="blackboard_results/${bench_name}/SVG/${MODEL_TAG}/generated_svgs"
            export RENDERED_IMG_OUT_DIR="blackboard_results/${bench_name}/SVG/${MODEL_TAG}/generated_imgs"
            export WORK_INPUT_DIR="blackboard_results/${bench_name}/SVG/${MODEL_TAG}/missing_images_temp"
            
            echo "  [$bench_name] Output SVG: $SVG_OUTPUT_DIR"
            echo "  [$bench_name] Output IMG: $RENDERED_IMG_OUT_DIR"
            
            # Execute steps for this benchmark
            run_svg_generation_loop 
            run_svg_filter
            run_image_rendering
            
            echo ">> Completed $bench_name"
        ) &
    done
    
    echo "All benchmarks started in background. Waiting for completion..."
    wait
    
    echo ""
    echo "ALL PIPELINES COMPLETED SUCCESSFULLY!"
}

main