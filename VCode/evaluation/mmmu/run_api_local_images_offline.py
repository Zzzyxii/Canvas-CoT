import os, random, base64, json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

# from datasets import load_dataset # Removed HF dependency
from utils.data_utils import load_yaml, construct_prompt, process_single_sample
from utils.eval_utils import parse_multi_choice_response
from openai import OpenAI
from zai import ZhipuAiClient

def encode_image(pil_image):
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_api_engine(args, sample):
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )

    content = [{"type": "text", "text": sample["prompt"]}]
    if "image" in sample and sample["image"] is not None:
        base64_img = encode_image(sample["image"])
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
        })

    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=args.config.get("temperature", 0),
        max_tokens=4096,
        stream=False
    )
    return resp.choices[0].message.content.strip()

# ===== main =====
def main():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True,
                        help="Root dir to save outputs (will save in Unknown/output.json)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root dir with subject subfolders that contain images and output.json for id list")
    parser.add_argument("--api_key", type=str, required=True, help="API key string")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. claude-4-opus")
    parser.add_argument('--config_path', type=str, default="configs/api.yaml")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to HuggingFace cache dir (for text metadata only)")
    parser.add_argument("--base_url", type=str, default="https://tao.plus7.plus/v1",
                        help="Base URL for API endpoint")
    args = parser.parse_args()

    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != "eval_params" and isinstance(value, list):
            assert len(value) == 1
            args.config[key] = value[0]

    id_set = set()
    root = Path(args.data_path)
    for subj_dir in root.iterdir():
        if subj_dir.is_dir():
            out_path = subj_dir / "output.json"
            if not out_path.exists():
                continue
            with out_path.open("r", encoding="utf-8") as f:
                local_records = json.load(f)
                ids = [rec["id"] for rec in local_records if "id" in rec]
                id_set.update(ids)

    # Load local prompts.json instead of HF dataset
    prompts_path = Path("/m2/slz/zyx/MM/VCode/data/mmmu/prompts.json")
    if not prompts_path.exists():
        print(f"❌ Error: Local prompts file not found at {prompts_path}")
        return

    print(f"📥 Loading local prompts from {prompts_path}")
    with open(prompts_path, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    
    # Load answer_dict_val.json to get question_type and ground_truth
    answer_dict_path = Path("/m2/slz/zyx/MM/VCode/evaluation/mmmu/answer_dict_val.json")
    if not answer_dict_path.exists():
        print(f"❌ Error: Answer dict file not found at {answer_dict_path}")
        return
    
    print(f"📥 Loading answer dict from {answer_dict_path}")
    with open(answer_dict_path, "r", encoding="utf-8") as f:
        answer_dict = json.load(f)

    # Filter prompts based on id_set
    dataset = []
    for rec in all_prompts:
        if rec["id"] in id_set:
            # Merge with answer_dict info
            # The key in answer_dict is like "validation_Accounting_1" but id is "dev_Accounting_1"
            # Wait, the answer_dict keys are "validation_..." but prompts are "dev_..."
            # Let's check if we can map them.
            # Usually dev set is validation set in MMMU context?
            # Let's try to replace "dev_" with "validation_" to find key
            
            key = rec["id"].replace("dev_", "validation_")
            if key in answer_dict:
                rec.update(answer_dict[key])
            else:
                print(f"⚠️ Warning: No answer dict entry for {rec['id']} (tried key {key})")
                # Try direct key just in case
                if rec["id"] in answer_dict:
                    rec.update(answer_dict[rec["id"]])

            # Update missing fields
            if "all_choices" not in rec:
                # If all_choices is missing in answer_dict, default to ['A', 'B', 'C', 'D', 'E']
                # Most MMMU questions are multiple choice with these options.
                choices = answer_dict.get(key, {}).get("all_choices")
                if not choices:
                    choices = ["A", "B", "C", "D", "E", "F", "G"] # Safe default
                rec["all_choices"] = choices
                
            if "index2ans" not in rec:
                rec["index2ans"] = answer_dict.get(key, {}).get("index2ans", {})

            dataset.append(rec)

    print(f"✅ Loaded {len(dataset)} samples from local prompts.json")

    samples = []
    for sample in dataset:
        if isinstance(sample.get("options"), list):
            sample["options"] = str(sample["options"])

        # sample = process_single_sample(sample) # prompts.json already has 'prompt' field constructed usually?
        # Let's check if we need construct_prompt. 
        # The prompts.json seems to have "prompt" field directly.
        # But construct_prompt might add options formatting if not present.
        # In the prompts.json snippet, "prompt" already includes options (A)... (B)...
        # So we might skip construct_prompt or use it carefully.
        
        # The original code did:
        # sample = process_single_sample(sample)
        # sample = construct_prompt(sample, args.config)
        
        # If prompts.json is already formatted, we just use it.
        # Let's assume prompts.json has the final prompt.
        
        if "prompt" not in sample:
             print(f"⚠️ Sample {sample.get('id')} missing prompt")
             sample["prompt"] = ""

        _id = sample["id"]
        img_path = None
        for subj_dir in root.iterdir():
            candidate = subj_dir / f"{_id}_img1.png"
            if candidate.exists():
                img_path = candidate
                break
            # Also check jpg
            candidate_jpg = subj_dir / f"{_id}_img1.jpg"
            if candidate_jpg.exists():
                img_path = candidate_jpg
                break
            candidate_jpeg = subj_dir / f"{_id}_img1.jpeg"
            if candidate_jpeg.exists():
                img_path = candidate_jpeg
                break


        if img_path and img_path.exists():
            try:
                sample["image"] = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to open image {img_path}: {e}")
                sample["image"] = None
        else:
            # print(f"⚠️ Image not found for {_id}")
            sample["image"] = None
        
        samples.append(sample)

    # Run inference
    records = []
    for sample in tqdm(samples):
        response = call_api_engine(args, sample)
        
        rec = {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "response": response,
            "question_type": sample.get("question_type", ""),
            "answer": sample.get("ground_truth", ""), # main_parse_and_eval expects 'answer'
            "all_choices": sample.get("all_choices", None), # If available
            "index2ans": sample.get("index2ans", None) # If available
        }
        records.append(rec)

    out_root = Path(args.output_path) / "Unknown"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "output.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(records)} samples -> {out_path}")

if __name__ == "__main__":
    main()
