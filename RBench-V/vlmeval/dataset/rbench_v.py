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
        'RBench-V-math': 'datasets/RBench-V-math.tsv',  # 设置本地TSV文件路径
        'RBench-V-count': 'datasets/RBench-V-count.tsv',
        'RBench-V-physics': 'datasets/RBench-V-physics.tsv',
        'RBench-V-game': 'datasets/RBench-V-game.tsv',
        'case': 'datasets/case.tsv',
        'RBench-V':'datasets/RBench-V.tsv'
    }

    def dump_image(self, line):
        if 'image' in line:
            # If image is already a path, return it directly
            if isinstance(line['image'], str) and os.path.exists(line['image']):
                return [line['image']]
            # If image is base64 data (not a list), decode it
            elif not isinstance(line['image'], list) and line['image'] and line['image'] != "nan":
                os.makedirs(self.img_root, exist_ok=True)
                tgt_path = osp.join(self.img_root, f"{line['index']}.png")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                return [tgt_path]
        return super().dump_image(line)
    
    def load_data(self, dataset):
        local_path = self.DATASET_URL[dataset]
        if osp.exists(local_path):
            print(f"Loading dataset from local path: {local_path}")
            return load(local_path)

    def build_prompt(self, line):
        # `line` may be either a row position (iloc) or a sample id (the `index` column).
        # Prefer the explicit `index` column to avoid off-by-one/id mismatch.
        if isinstance(line, int):
            if isinstance(self.data, pd.DataFrame) and 'index' in self.data.columns:
                matched = self.data[self.data['index'] == line]
                if len(matched) >= 1:
                    line = matched.iloc[0]
                else:
                    line = self.data.iloc[line]
            else:
                line = self.data.iloc[line]

        # 构建详细的提示词
        instruction = ""
        
        # 根据是否有图片调整提示词
        if line['image'] and str(line['image']) != "nan" and line['image'] is not None:
            if self.meta_only:
                tgt_path = toliststr(line['image'])
            else:
                tgt_path = self.dump_image(line)
            instruction += "Solve the following question according to the given picture step-by-step.\n\n"
        else:
            instruction += "Solve the following question step-by-step.\n\n"
            tgt_path = None

        instruction += f"Question: {line['question']}\nAnswer:"
 
        msgs = []
        # 只有当有图片时才添加图片消息
        if tgt_path:
            if isinstance(tgt_path, list):
                msgs.extend([{"type": "image", "value": p} for p in tgt_path if p])
            elif tgt_path:
                msgs.append({"type": "image", "value": tgt_path})

        msgs.append({"type": "text", "value": instruction})

        # Debug: print sample index and image filename(s) for quick sanity check
        try:
            sample_idx = line['index'] if isinstance(line, dict) else line.get('index', None)
        except Exception:
            sample_idx = None

        img_files = []
        if tgt_path:
            if isinstance(tgt_path, list):
                img_files = [osp.basename(p) for p in tgt_path if p]
            else:
                img_files = [osp.basename(tgt_path)]
        print(f"[RBench_V] index={sample_idx} images={img_files}")
        return msgs
    
    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathverse import MathVerse_auxeval_extract, MathVerse_auxeval_score, MathVerse_acc

        def _to_openai_chat_completions_url(base_url: str):
            if not base_url:
                return None
            u = str(base_url).strip().rstrip('/')
            if not u:
                return None
            if u.endswith('/chat/completions'):
                return u
            if u.endswith('/v1'):
                return u + '/chat/completions'
            return u

        suffix = eval_file.split('.')[-1]
        # `run.py` passes judge-args including `model`; we must not pass it twice.
        judge_model = judge_kwargs.pop('model', None) or 'gemini-3-flash-preview'
        # Build file names based on the judge model to avoid collision.
        judge_tag = str(judge_model).replace('/', '_')
        storage_extract = eval_file.replace(f'.{suffix}', f'_openai_{judge_tag}_extract.xlsx')
        tmp_file_extract = eval_file.replace(f'.{suffix}', f'_openai_{judge_tag}_extract.pkl')
        storage_score = eval_file.replace(f'.{suffix}', f'_openai_{judge_tag}_score.xlsx')
        tmp_file_score = eval_file.replace(f'.{suffix}', f'_openai_{judge_tag}_score.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        # Optional: force re-eval from scratch by clearing aux caches.
        no_cache = bool(judge_kwargs.pop('no_cache', False))
        if no_cache:
            for p in [tmp_file_extract, storage_extract, tmp_file_score, storage_score]:
                try:
                    if p and osp.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            print(f'[RBench_V.evaluate] no_cache=True: cleared aux files for judge={judge_model}')

        # If requested, evaluate from scratch (ignore & wipe any existing aux caches).
        # This affects BOTH stages:
        # - extract caches: *_extract.pkl / *_extract.xlsx
        # - score caches: *_score.pkl / *_score.xlsx
        no_cache = bool(judge_kwargs.pop('no_cache', False) or judge_kwargs.pop('force_rerun', False))
        if no_cache:
            for p in [tmp_file_extract, storage_extract, tmp_file_score, storage_score]:
                try:
                    if osp.exists(p):
                        os.remove(p)
                except Exception as e:
                    print(f'[RBench_V.evaluate] WARNING: failed to remove cache file {p}: {type(e).__name__}: {e}')
            print(f'[RBench_V.evaluate] no_cache=True: cleared aux files for judge={judge_tag}')

        # Allow passing credentials/endpoint via judge_kwargs OR env vars.
        # - api_key: from judge_kwargs['api_key'] or env OPENAI_API_KEY
        # - base_url: from judge_kwargs['base_url'] or env BASE_URL (OpenAI SDK style)
        # - api_base: from judge_kwargs['api_base'] or env OPENAI_API_BASE (OpenAIWrapper style)
        api_key = judge_kwargs.pop('api_key', None) or os.environ.get('OPENAI_API_KEY', None)
        base_url = judge_kwargs.pop('base_url', None) or os.environ.get('BASE_URL', None)
        api_base = judge_kwargs.pop('api_base', None) or os.environ.get('OPENAI_API_BASE', None)
        api_base = api_base or _to_openai_chat_completions_url(base_url)

        # Build judge model (OpenAI-compatible) used by MathVerse aux eval.
        # Note: OpenAIWrapper asserts key starts with 'sk-' when not Azure; setting env is fine.
        if api_key:
            os.environ['OPENAI_API_KEY'] = str(api_key)
        if api_base:
            os.environ['OPENAI_API_BASE'] = str(api_base)

        model = build_judge(model=judge_model, max_tokens=128, **judge_kwargs)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(ans):
                print(f'[RBench_V.evaluate] extract: total={lt}, cached={len(ans)}, remaining={len(indices)}')
            else:
                print(f'[RBench_V.evaluate] extract: total={lt}, remaining={len(indices)}')

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
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(ans):
                print(f'[RBench_V.evaluate] score: total={lt}, cached={len(ans)}, remaining={len(indices)}')
            else:
                print(f'[RBench_V.evaluate] score: total={lt}, remaining={len(indices)}')

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
        
        # Handle column name mismatch and case sensitivity
        if 'subject' not in results.columns and 'catagory' in results.columns:
            results['subject'] = results['catagory']
        
        if 'subject' in results.columns:
            results['subject'] = results['subject'].astype(str).str.lower()
        
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

        # 计算去掉 math 的总体正确率（若存在 math 子集）
        wo_math = results
        if 'subject' in results.columns:
            wo_math = results[results['subject'] != 'math']
        if len(wo_math) > 0:
            accuracies['overall_wo_math'] = wo_math['score'].mean() * 100
        else:
            accuracies['overall_wo_math'] = 0.0
        
        # 创建结果DataFrame
        score_df = pd.DataFrame({
            'Metric': subjects + ['Overall', 'Overall w/o Math'],
            'Score': [accuracies[subject] for subject in subjects] + [overall_accuracy, accuracies['overall_wo_math']]
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
