<h2 align="center" style="font-size: 2.5em; font-weight: bold; color: #2c3e50;">
  RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs
</h2>

<p align="center">
  <a href="https://evalmodels.github.io/rbenchv/" style="margin: 0 10px;">🌐 Homepage</a> |
  <a href="https://huggingface.co/datasets/R-Bench/R-Bench-V" style="margin: 0 10px;">🤗 Dataset</a> |
  <a href="https://arxiv.org/pdf/2505.16770" style="margin: 0 10px;">📖 ArXiv</a> |
  <a href="https://evalmodels.github.io/rbenchv/#leaderboard" style="margin: 0 10px;">🏆 Leaderboard</a> |
  <a href="https://github.com/CHEN-Xinsheng/VLMEvalKit_RBench-V" style="margin: 0 10px;">🐙 GitHub</a>
</p>


This repository contains the evaluation code for the paper "[RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs](https://evalmodels.github.io/rbenchv)". The code is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

---


## 🔔 Introduction

<p align="center">
  <img src="assets/teaser.png" alt="RBench-V Overview" style="width: 800px;"> 
</p>

The rapid advancement of native multi-modal models and omni-models, exemplified by GPT-4o, Gemini and o3 with their capability to process and generate content across modalities such as text and images, marks a significant milestone in the evolution of intelligence. Systematic evaluation of their multi-modal output capabilities in visual thinking process (a.k.a., multi-modal chain of thought, M-CoT) becomes critically important. However, existing benchmarks for evaluating multi-modal models primarily focus on assessing multi-modal inputs and text-only reasoning process while neglecting the importance of reasoning through multi-modal outputs.

In this paper, we present a benchmark, dubbed as **RBench-V**, designed to assess models’ multi-modal reasoning. To conduct RBench-V, we carefully hand-pick 803 questions covering math, physics, counting and games. Unlike problems in previous benchmarks, which typically specify certain input modalities, RBench-V presents problems centered on **multi-modal outputs**, which requires **image manipulation**, such as generating novel images and constructing auxiliary lines to support reasoning process.

We evaluate numerous open- and closed-source models on RBench-V, including o3, Gemini 2.5 pro, Qwen2.5-VL, etc. Even the best-performing model, o3, achieves only **25.8%** accuracy on RBench-V, far below the human score of **82.3%**, which shows current models struggle to leverage multi-modal reasoning.



## 🏆 Main Results

The table below presents the performance of various open- and closed-source models. All results can be viewed on the [leaderboard](https://evalmodels.github.io/rbenchv/#leaderboard).

<p align="center">
  <img src="assets/compare_models.png" alt="RBench-V Results Overview" style="width: 600px;"> 
</p>


| Model | Overall | w/o Math | Math | Physics | Counting | Game
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Human Expert 👑 | 82.3 | 81.7 | 84.7 | 69.4 | 81.0 | 89.1
| OpenAI o3 🥇 |  25.8 | 19.5 | 48.3 | 20.4 | 22.1 | 17.1
| OpenAI o4-mini 🥈 |  20.9 | 14.6 | 43.2 | 12.7 | 17.4 | 13.8
| Gemini 2.5 pro-preview-0506 🥉 |  20.2 | 13.9 | 42.6 | 9.6 | 19.0 | 12.7
| Doubao-1.5-thinking-pro-m |  17.1 | 11.0 | 38.6 | 13.4 | 9.7 | 10.5
| OpenAI o1 |  16.2 | 11.0 | 34.7 | 5.7 | 12.3 | 13.1
| Doubao-1.5-vision-pro |  15.6 | 11.5 | 30.1 | 8.9 | 12.8 | 12.0
| OpenAI GPT-4o-20250327 |  14.1 | 11.2 | 24.4 | 3.2 | 13.3 | 14.2
| OpenAI GPT-4.1 |  13.6 | 11.7 | 20.5 | 5.7 | 11.3 | 15.3
| Step-R1-V-Mini |  13.2 | 8.8 | 29.0 | 6.4 | 10.3 | 9.1
| OpenAI GPT-4.5 |  12.6 | 11.0 | 18.2 | 2.5 | 11.8 | 15.3
| Claude-3.7-sonnet |  11.5 | 9.1 | 19.9 | 3.8 | 8.7 | 12.4
| QVQ-Max |  11.0 | 8.1 | 21.0 | 5.7 | 6.2 | 10.9
| Qwen2.5VL-72B |  10.6 | 9.2 | 15.3 | 3.8 | 6.2 | 14.5
| InternVL-3-38B |  10.0 | 7.2 | 20.5 | 0.6 | 5.1 | 12.4
| Qwen2.5VL-32B |  10.0 | 6.4 | 22.7 | 2.5 | 4.1 | 10.2
| MiniCPM-2.6-o |  9.7 | 7.5 | 17.6 | 1.3 | 3.6 | 13.8
| Llama4-Scout (109B MoE) |  9.5 | 6.9 | 18.8 | 3.2 | 4.1 | 10.9
| MiniCPM-2.6-V |  9.1 | 7.2 | 15.9 | 1.3 | 6.2 | 11.3
| LLaVA-OneVision-72B |  9.0 | 8.9 | 9.1 | 4.5 | 4.6 | 14.5
| DeepSeek-VL2 |  9.0 | 7.0 | 15.9 | 0.6 | 5.6 | 11.6
| LLaVA-OneVision-7B |  8.5 | 6.8 | 14.2 | 2.5 | 4.6 | 10.9
| Qwen2.5VL-7B |  8.3 | 7.0 | 13.1 | 2.5 | 3.6 | 12.0
| InternVL-3-8B |  8.2 | 6.0 | 15.9 | 1.9 | 5.6 | 8.7
| InternVL-3-14B |  8.0 | 7.0 | 11.4 | 1.3 | 5.1 | 11.6
| Qwen2.5-Omni-7B |  7.7 | 4.5 | 11.4 | 1.9 | 2.1 | 7.7




---

## ⚙️ Installation 

To install the required packages, run:

```bash
# Prepare repository and environment
git clone git@github.com:CHEN-Xinsheng/VLMEvalKit_RBench-V.git
cd ./VLMEvalKit_RBench-V
pip install -e .
```

In some cases, you may need to install additional dependencies.

```bash
pip install 'accelerate>=0.26.0'
pip install flash-attn --no-build-isolation
pip install qwen-vl-utils
```

For more details on installation and setup, please refer to the [[VLMEvalKit Quickstart](docs/en/Quickstart.md) | [快速开始](docs/zh-CN/Quickstart.md)].

---

## 🧠 Inference & Evaluation

You can directly perform inference and evaluation on the selected open-source models using the following command:

```bash
python run.py --model <MODEL_NAME> --data RBench_V --api-nproc <NUMBER_OF_PROCS_TO_CALL_API_JUDGE_MODEL> --verbose
```

Examples:
```bash
python run.py --model Qwen2.5-VL-7B-Instruct --data RBench_V --api-nproc 16 --verbose
python run.py --model MiniCPM-V-2_6 --data RBench_V --api-nproc 16 --verbose
python run.py --model MiniCPM-o-2_6 --data RBench_V --api-nproc 16 --verbose
python run.py --model Llama-4-Scout-17B-16E-Instruct --data RBench_V --api-nproc 4 --verbose --use-vllm 
python run.py --model llava_onevision_qwen2_72b_ov --data RBench_V --api-nproc 4 --verbose 
python run.py --model llava_onevision_qwen2_7b_ov --data RBench_V --api-nproc 4 --verbose
```

**Some Arguments**

- `--model (list[str])`: Set the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`).
- `--mode (str, default to 'all', choices are ['all', 'infer'])`: When `mode` set to "all", will perform both inference and evaluation; when set to "infer", will only perform the inference.
- `--api-nproc (int, default to 4)`: The number of threads for OpenAI API calling.
- `--reuse`: reuse the latest inference results and temporary pickle files.


📝 Call API Judge Model

- In order to call the API judge model, you need to configure your API Key in the code, which has now been marked as placeholders with `MY_API_KEY`.

🛠️ Run Custom Model

- If you want to run a custom model, you can do so by following [VLMEvalKit development guidelines](docs/en/Development.md#implement-a-new-model).


---


## 📚 Citation

**BibTeX:**
```bibtex
@misc{guo2025rbenchvprimaryassessmentvisual,
      title={RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs}, 
      author={Meng-Hao Guo and Xuanyu Chu and Qianrui Yang and Zhe-Han Mo and Yiqing Shen and Pei-lin Li and Xinjie Lin and Jinnian Zhang and Xin-Sheng Chen and Yi Zhang and Kiyohiro Nakayama and Zhengyang Geng and Houwen Peng and Han Hu and Shi-Min Hu},
      year={2025},
      eprint={2505.16770},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.16770}, 
}
