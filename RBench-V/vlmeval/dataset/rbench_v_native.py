from typing import Dict, Any
import pandas as pd
from .image_base import ImageBaseDataset
from ..utils import track_progress_rich
from ..smp import *
from .utils import build_judge

# python run.py --model deepseek_vl2 --data RBench_V --verbose --reuse --use-vllm



class RBench_V(ImageBaseDataset):
    """Custom dataset class for handling TSV files with image QA pairs"""
    
    TYPE = 'VQA'  # 设置数据集类型为VQA
    DATASET_URL = {
        'RBench_V': 'datasets/RBench-V.tsv'  # 设置本地TSV文件路径
    }
    
    def load_data(self, dataset):
        local_path = self.DATASET_URL[dataset]
        if osp.exists(local_path):
            print(f"Loading dataset from local path: {local_path}")
            return load(local_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # 构建详细的提示词
        instruction = (
            "You are an expert assistant. "
        )
        
        # 根据是否有图片调整提示词
        if line['image'] and line['image'] != "nan":
            if self.meta_only:
                tgt_path = toliststr(line['image'])
            else:
                tgt_path = self.dump_image(line)
            instruction += "Solve the following question according to the given picture step-by-step.\n\n"
        else:
            instruction += "Solve the following question step-by-step.\n\n"
            tgt_path = None

        instruction += (
            "At the VERY END of your answer, output ONLY the FINAL ANSWER in this format:\n\n"
            "\\[\n\\boxed{your_final_answer_here}\n\\]\n\n"
            " You MUST put the final answer in the \\boxed{} environment.\n"
            " This applies even if the answer is a text explanation like \"The singlet state is lower in energy.\"\n"
            "Do NOT include multiple boxes.\n"
            "Do NOT include \\boxed anywhere else in your reasoning.\n"
            " The box must appear on the last line of the response.\n\n"
            "WARNING: DO NOT forget to include \\boxed{} with the final answer. Responses without it will be considered INVALID.\n\n"  # noqa: E501
            "Example:\n"
            "Question: What is the energy difference between n=2 and n=1 in hydrogen?\n"
            "Answer: The energy levels are E_n = -13.6 / n² (in eV).\n"
            "E_2 = -13.6 / 4 = -3.4 eV\n"
            "E_1 = -13.6 eV\n"
            "ΔE = 13.6 - 3.4 = 10.2 eV\n"
            "\\[\n\\boxed{10.2\\ \\text{eV}}\n\\]\n\n"
            f"Question: {line['question']}\nAnswer:"
        )
 
        msgs = []
        # 只有当有图片时才添加图片消息
        if tgt_path:
            if isinstance(tgt_path, list):
                msgs.extend([{"type": "image", "value": p} for p in tgt_path if p])
            elif tgt_path:
                msgs.append({"type": "image", "value": tgt_path})

        msgs.append({"type": "text", "value": instruction})
        print(msgs) # Debugging line
        return msgs
    
    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathverse import MathVerse_auxeval_extract, MathVerse_auxeval_score, MathVerse_acc

        suffix = eval_file.split('.')[-1]
        # model = "4O_MINI_API"
        # model = 'api_google_gemini-2.5-pro-preview-05-06'
        model = 'api_openai_chatgpt-4o-latest'
        storage_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.xlsx')
        tmp_file_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.pkl')
        storage_score = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file_score = eval_file.replace(f'.{suffix}', f'_{model}_score.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            # model = build_judge(max_tokens=128, **judge_kwargs)
            from vlmeval.api import GPT4V
            model = GPT4V(
                model="gpt-4o",
                key=,
                api_base=
                temperature=0,
                retry=10,
                verbose=False
            )
                            
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_extract'] == v['log_extract'] and ans[k]['extract'] == v['extract']

            data['extract'] = [ans[idx]['extract'] for idx in data['index']]
            data['log_extract'] = [ans[idx]['log_extract'] for idx in data['index']]
            dump(data, storage_extract)

        # stage2: score the answer
        if not osp.exists(storage_score):
            data = load(storage_extract)
            from vlmeval.api import GPT4V
            model = GPT4V(
                model="gpt-4o",
                key=,
                api_base=,
                temperature=0,
                retry=10,
                verbose=False
            )
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_score,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k]['score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [ans[idx]['log_score'] for idx in data['index']]
            dump(data, storage_score)

        results = pd.read_excel(storage_score)
        
        # 按学科统计正确率
        subjects = ['physics', 'math', 'game', 'counting']
        accuracies = {}
        
        # 计算每个学科的正确率
        for subject in subjects:
            subject_data = results[results['subject'] == subject]
            if len(subject_data) > 0:
                accuracy = subject_data['score'].mean() * 100
                accuracies[subject] = accuracy
            else:
                accuracies[subject] = 0.0
                
        # 计算总体正确率
        overall_accuracy = results['score'].mean() * 100
        accuracies['overall'] = overall_accuracy
        
        # 创建结果DataFrame
        score_df = pd.DataFrame({
            'Metric': subjects + ['Overall'],
            'Score': [accuracies[subject] for subject in subjects] + [overall_accuracy]
        })
        
        # 保存结果
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(score_df, score_file)
        
        return {
            'accuracies': accuracies,
            'total_samples': len(results),
            'samples_per_subject': {
                subject: len(results[results['subject'] == subject]) 
                for subject in subjects
            }
        }