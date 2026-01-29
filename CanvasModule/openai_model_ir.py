import copy
import re, os
from .model_utils import LLM, call_api, truncate_images
from .model_utils import encode_image_base64
from functools import partial
import openai
from PIL import Image
import time
import requests
import io
import uuid
import base64
import json
from typing import List, Dict, Any, Optional, Union

# from .blackboard_tools import blackboard_tools
# from .blackboard import Blackboard


SYSTEM_PROMPT = """You are a helpful assistant for solving complex problems step-by-step.

Instructions:
1. Think carefully about the problem
2. Put your reasoning in <think> tags
3. Provide intermediate reasoning steps
4. When you have the final answer, put it in \\boxed{}
5. Each iteration should build on previous feedback

Example format:
<think>I need to analyze this problem step by step...</think>
After calculation, the answer is \\boxed{42}.
"""



def is_url(path):
    return path.startswith('http://') or path.startswith('https://')

# huggingface datasets cannot support a folder with more than 10K iamge
# we split the images in mm-niah/obelics and vh/train2017
def string_to_int_mod(string_value, mod_num=20):
    char_sum = sum(ord(char) for char in string_value)
    result = char_sum % mod_num
    return result


def image_input(img_url):
    try:
        # 直接读取本地文件的二进制内容
        with open(img_url, "rb") as f:
            img_file = f.read()
        # 用 PIL 打开本地图片
        with Image.open(io.BytesIO(img_file)) as img:
            img_format = img.format
            img_base64 = base64.b64encode(img_file).decode("utf-8")
            image_url = f"data:image/{img_format.lower()};base64,{img_base64}"
            return image_url
        # img_file = requests.get(img_url).content
        # with Image.open(io.BytesIO(img_file)) as img:
        #     img_format = img.format
        #     img_base64 = base64.b64encode(img_file).decode("utf-8")
        #     image_url = f"data:image/{img_format.lower()};base64,{img_base64}"
        #     return image_url
        
    except:
        print('url 图片解析错误，请更换 url')
        return ''
    
    

def add_url_bucket(url):
    new_url = url
    check_dir_list = ["vh/train2017/", "mm-niah/obelics/"]
    for check_dir in check_dir_list:
        if check_dir in url:
            prefix, file_name = url.split(check_dir)
            bucket_id = string_to_int_mod(file_name)
            bucket_str = f"bucket_{bucket_id}"
            new_url = f"{prefix}{check_dir}{bucket_str}/{file_name}"

    return new_url

def parse_response(text):
    # 只需要提取思考内容和清理文本
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    reasoning_content = think_match.group(1).strip() if think_match else None
    
    # 清理标签
    clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    
    return {
        'raw_response': text,
        'reasoning_content': reasoning_content,
        'content': clean_text
    }


class OpenAIModel(LLM):
    def __init__(
            self,
            model_name,
            temperature=0.9,
            top_p=0.9,
            max_length=32768,
            generation_max_length=32768,
            generation_min_length=0,
            do_sample=True,
            stop_newline=False,
            use_chat_template=True,
            **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        self.api_model = True
        self.max_image_num = kwargs.get("max_image_num", None)
        self.api_sleep = kwargs.get("api_sleep", 10)
        self.image_detail = kwargs.get("image_detail", "auto")
        self.image_resize = kwargs.get("image_resize", None)
        self.your_logid = str(uuid.uuid4())

        # self.blackboard = None
        # self.provided_tools = blackboard_tools
        # self.system_prompt = SYSTEM_PROMPT.format(provided_tools=self.provided_tools)
        self.system_prompt = """You are a helpful assistant for solving complex problems step-by-step.
        
        Instructions:
        1. Think carefully about the problem
        2. Put your reasoning in <think> tags
        3. Provide intermediate reasoning steps
        4. When you have the final answer, put it in \\boxed{}
        5. Each iteration should build on previous feedback

        Example format:
        <think>I need to analyze this problem step by step...</think>
        After calculation, the answer is \\boxed{42}.
        """
        


        if "claude" in self.model_name:
            self.model = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],  # Your Anthropic API key
                base_url="https://api.anthropic.com/v1/"  # Anthropic's API endpoint
            )
        else:
            self.model = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["BASE_URL"]
            )

        print(f"----------currently, using image_detail: {self.image_detail}----------")

        # make sure to set the OPENAI_API_KEY environment variable

        self.model_name = model_name
        self.processor = None

    def format_chat(self, text, image_list, system_prompt, is_url_image=False):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        # print('** content: ', content)
        for c in content:
            if c == "<image>":
                if is_url_image:
                    curr_image_url = image_list[image_idx]
                else:
                    curr_image_url = f"data:image/png;base64,{image_list[image_idx]}"
                new_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": curr_image_url,
                        "detail": self.image_detail
                    },
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content}]
        return messages

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        if(len(image_list) == 0):
            messages = self.format_chat(text, [], data["system_template"])
            return {"messages": messages}
        
        if is_url(image_list[0]):
            # image_flag = [check_image_url_mime(url) for url in image_list]
            image_list = [add_url_bucket(url) for url in image_list]
            messages = self.format_chat(text, image_list, data["system_template"], is_url_image=True)
        else:
            image_list = [Image.open(image).convert('RGB') for image in image_list]

            if self.image_resize is not None:
                from .model_utils import resize_image
                image_list = resize_image(image_list, self.image_resize)

            # convert all format images to png
            image_list = [encode_image_base64(image) for image in image_list]

            messages = self.format_chat(text, image_list, data["system_template"])

        return {"messages": messages}

    def generate_roll(self, inputs=None, prompt=None, data_ori=None, **kwargs):
        data_idx = data_ori['pid']
        print('='*10, f'** data_idx: {data_idx}')
        print('** question: ', data_ori['question'])
        
        # 添加系统提示
        if inputs["messages"][0]["role"] == "user":
            inputs["messages"].insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        
        max_iterations = 5  # 最大迭代次数
        iteration = 0
        response = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f'--- Iteration {iteration}/{max_iterations} ---')
            
            # 1. 生成推理
            response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
            parsed_response = parse_response(response['output'])
            print('** Model Response: ', parsed_response['content'])
            
            # 2. 检查是否有最终答案
            if "\\boxed{" in response['output']:
                print('** Found final answer in boxed format')
                break
            
            # 3. 生成批判（使用纯文本批判）
            critique = self.generate_critique(
                question=data_ori['question'],
                current_reasoning=parsed_response['content'],
                iteration=iteration
            )
            
            # 4. 将批判作为用户输入添加
            critique_message = {
                "role": "user",
                "content": [{"type": "text", "text": critique}]
            }
            inputs["messages"].append(critique_message)
            
            # 5. 将模型回复添加到历史
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response['output']}]
            }
            inputs["messages"].append(assistant_message)
        
        # 如果达到最大迭代次数仍无答案，要求最终答案
        if iteration >= max_iterations and "\\boxed{" not in response['output']:
            print('** Max iterations reached, asking for final answer')
            inputs["messages"].append({
                "role": "user",
                "content": [{"type": "text", "text": "Please provide the final answer directly in \\boxed{} format."}]
            })
            response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
        
        if response is not None:
            response['history'] = inputs["messages"]
            
        return response

    def generate_critique(self, question, current_reasoning, iteration):
        """生成对当前推理的文本批判"""
        critique_prompt = f"""
    Analyze the following reasoning step for the question: "{question}"

    Current reasoning (Iteration {iteration}):
    {current_reasoning}

    Please provide constructive critique:
    1. Identify any logical errors or inconsistencies
    2. Point out missing steps or assumptions
    3. Suggest improvements or corrections
    4. Ask clarifying questions if needed

    Keep your critique focused and helpful for the next iteration.
    """
        
        critique_inputs = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful critique assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": critique_prompt}]}
            ]
        }
        
        critique_response = self.generate(inputs=critique_inputs)
        return critique_response['output']

    def generate(self, inputs=None, prompt=None, **kwargs):
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        save_prompt = [c["text"] if c["type"] == "text" else "<image>" for c in inputs["messages"][0]["content"]]
        save_prompt = "".join(save_prompt)
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]

        func = partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
            # max_tokens=1024*10,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            # tools=self.provided_tools,
            # tool_choice="auto",
            extra_headers={"X-TT-LOGID": self.your_logid},
            **kwargs,
        )

        start_time = time.time()
        try:
            output = call_api(func, limit=5, pause=5)
        except Exception as e:
            print("current call_api failed, assign None to output.")
            output = None

        end_time = time.time()

        print(f"example finished, used {end_time - start_time} secs, sleep {max(self.api_sleep - (end_time - start_time), 0.1)} secs")
        time.sleep(max(self.api_sleep - (end_time - start_time), 0.1))

        print('-'*10)
        # print('** inputs["messages"]: ', inputs["messages"])
        # print('** inputs["messages"]: ', inputs["messages"][0]['content'][-1]['text'].split("\n\nQuestion:")[-1])
        
        if output is None:
            print('** output: <None>')
            return {"output": "", "input_len": -1, "output_len": -1, "input_text": save_prompt}

        content = None
        try:
            content = output.choices[0].message.content
        except Exception as e:
            print(f'** output parse failed: {type(e).__name__}: {e}')
            content = None

        print('** output: ', content)

        if content is not None:
            usage = getattr(output, 'usage', None)
            input_len = getattr(usage, 'prompt_tokens', -1) if usage is not None else -1
            output_len = getattr(usage, 'completion_tokens', -1) if usage is not None else -1
            return {
                "output": content,
                "input_len": input_len,
                "output_len": output_len,
                "input_text": save_prompt,
            }
        return {"output": "", "input_len": -1, "output_len": -1, "input_text": save_prompt}
