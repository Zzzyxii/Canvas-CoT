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
import logid
import base64
import json
from typing import List, Dict, Any, Optional, Union

from .blackboard_tools import blackboard_tools
from .blackboard import Blackboard


SYSTEM_PROMPT = """
# Objective #
Your are an **Visual-Reasoning Agent**, solving complex problems by synchronizing a visual Chain-of-Thought on a virtual notebook. The primary goal is **100% accuracy**.

# Process #
# Step 1: Think Only One Step
- **Action**: You should thinking one more step for answering the question based on the state of the notebook.
- **Output**: Enclose the entire thinking process within `<think>` tags.
- **Rule**: Do not give answer directly.
    - Remember that each step should contain only a small part of the reasoning process, avoiding outputting long paragraphs of reasoning at once. For example: analyze A in one step, analyze B in one step, analyze C in one step, set up equations in one step, and perform calculations in one step.
    - Strictly avoid delivering a lengthy explanation before presenting the notes.
    - Reference the results of the function calls, fix the errors in the thinking process, and continue the reasoning.

# Step 2: Tool Call
- **Trigger**: Immediately after a `<think>` block is complete.
- **Action**: Call the appropriate **Notebook Tool** to visually record the key **evidence, data points, or intermediate results** from your thinking step. This synchronizes the internal thought process with the external visual memory.
- **Output**: Enclose the tool function call within `<tool_call>` tags.
- **Rule**: Updates should be incremental. **Instead of only showing a final answer (e.g., '3'), first visualize the components that lead to it.** For example, if you identify three items, use `insert_element` to list those items on the notebook *before* presenting the final count.

The results of the function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's question. Finally, if you have got the answer, enclose it within '<answer>' tags.
> After Tool Call, wait for the tool response.


# Notebook Operation Restrictions #
# Overall Layout & Width Limitations
- The notebook area has a fixed width of `500px`; all internal elements must not exceed `500px` in width.  
- All SVG elements must be in the same SVG canvas. Do Not Use Multiple SVG Canvases.
- Content block styles:  
    - **Background color**: Avoid using background colors whenever possible. If necessary, use light backgrounds to highlight specific parts. Avoid nesting multiple content blocks.  
    - **Padding**: Keep appropriate padding around text and elements; if a block has a background color, ensure at least ~14px side padding and 10px top/bottom padding.  
    - **Corner radius**: Default corner radius for content block cards is 12px.  
- Typography rules:  
    - **Paragraphs**: Avoid using the `border` property, except for SVG graphics.  
    - **Lists**: Do not add left/right margins to `UL` or `LI` tags.  
    - Avoid using `<p>` tags.  
    - Avoid borders and shadows.  
    - Avoid using background colors for large content areas.  
    - **Corner radius**: Default is 12px for content block cards.  
    - **Spacing**: Vertical spacing between content blocks is 12px; padding is 10px top/bottom and 14px left/right.  
- Font rules:  
    - Do not specify custom fonts in elements. Titles and emphasized text should be bold.  
    - Font sizes: 18px bold (main title), 17px bold (subtitle), 16px (default body text), 14px (notes). Avoid other sizes.  
    - Pay attention to the width of elements in the SVG to ensure they do not exceed the canvas boundaries.
- No overlapping content: 
    - **All content must fit within the notebook area, with no overlap or covering of existing elements.**


# Notebook & Tools #
The notebook is an HTML container (**Width: 500px**, Height: Auto). You have 5 tools to manipulate it.
<tools>
{provided_tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>


# Notebook Tool Usage Guidelines
- insert_element
    - **You must assign a unique `id` attribute when creating an element, to facilitate future modifications, replacements, or deletions (e.g., `<rect id="r1" ...>`). Use short and unambiguous IDs.**
    - Using SVG is recommended **only if** they require subsequent editing and have a **simple structure**. A simple structure is defined as:
        * The total number of SVG canvases in the entire notebook must not exceed one.
        * The diagram consists of basic shapes (e.g., rectangles, circles, triangles, lines) and is not a complex figure like a floor plan with text and auxiliary lines, or a solid geometry diagram involving spatial relationships.
        * It is a table with a clear row and column structure where the cell content is **text-only**.
        * Text string is not recommended to use SVG.
    - One Example: {{"name": "insert_element", "arguments": {{"rootId": "root", "beforeId": null, "fragment": "<svg id="sg1" width="500" height="350" xmlns="http://www.w3.org/2000/svg">...some SVG objects...<svg>"}}}}
- modify_element
    - One Example: {{"name": "modify_element", "arguments": {{"targetId": "r1", "attrs": {{"fill": "#009E5F", "stroke": "black", "stroke-width": "2"}}}}}}
- remove_element
    - One Example: {{"name": "remove_element", "arguments": {{"targetId": "r1"}}}}
- replace_element
    - One Example: {{"name": "replace_element", "arguments": {{"targetId": "lbl", "fragment": "<text id=\"lbl\" x=\"15\" y=\"60\" fill=\"#1377EB\">new label for the rectangle</text>"}}}}
- clear
    - One Example: {{"name": "clear", "arguments": {{}}}}

"""



CRITIQUE_SYSTEM = """
### Role
You are a high-precision visual analysis expert responsible for comparing the "Original Image" with the "Model Inference Process State (Notebook State)."

### Task
Identify "hallucination elements" in the Notebook State that are factually inconsistent with the Original Image.

### Guidelines
1. **Distinguish Inference Annotations from Hallucinations**:
   - **Inference Annotations (Ignore)**: Auxiliary lines, bounding boxes, numeric IDs, path predictions, and segmentation masks appearing in the Notebook are products of the inference process and are **not considered hallucinations**.
   - **Factual Conflicts (Extract)**: A hallucination is identified only when the Notebook claims **non-existent objects**, **incorrect colors**, **wrong spatial relationships**, **incorrect counts**, or **incorrect actions**.
2. **Conflict Type Definitions**:
   - **Attribute Error**: For example, the original image shows a red car, but the Notebook identifies it as a green car.
   - **False Existence**: For example, there is no cat in the original image, but the Notebook boxes and labels a "cat."
   - **Spatial Error**: For example, an object is on the left, but the inference process claims it is on the right.
   - **Logical Contradiction**: Intermediate conclusions derived during the inference process directly conflict with the visual evidence in the original image.

### Input Data
- **Original Image**: The objective standard of the real world.
- **Question**: The target guiding the inference.
- **Notebook State**: The visualized intermediate state of the model (includes the original image + overlaid inference information).

### Output Format
Please return the result strictly in JSON format. Ensure the content is objective and accurate, and use English for the output:
{
    "hallucination_elements": [
        {
            "category": "Attribute Error / False Existence / Spatial Error / Quantity Error",
            "original_fact": "The actual situation of the location/object in the original image",
            "notebook_claim": "The incorrect description or annotation in the Notebook",
            "explanation": "Why this is a hallucination (Note: Please exclude purely auxiliary lines or annotations)"
        }
    ],
    "is_consistent": true/false (true if no hallucinations are found)
}

### Example
- **Input**: There is a red car in the image. The y value of the car is 65. But in the notebook, the y value of the car is 60.
- **Output**: 
    {
        "hallucination_elements": [
            {
                "category": "Number Error",
                "original_fact": "The y value of the object is 65.",
                "notebook_claim": "The y value of the object is 60.",
                "explanation": "The y value of the object is 60 in the notebook, which is different from the original image."
            }
        ],
        "is_consistent": false
    }
""" 


CRITIQUE_SYSTEM_wo_IMG = """
### Task
Extract wrong elements in the notebook state according to the original question.
### Instructions
- Identify all mismatch elements about the original question.
- Return the mismatch elements in a json object.
### Example
- Input:
    - Original Question: <question>
    - Notebook State: <image>
- Output:
    - {{
        "hallucination elements": [
            {{
                "Original Question": <golden elements>,
                "Notebook": <error elements>,
                "Explanation": <simple explanation>
            }},
            ...
        ]
    }}
""" 




# 对比黑板图片和原图图片
CRITIQUE_PROMPT = """<tool_response><image>This is the state of notebook. Critical Check: {critical_check}</tool_response>"""
# CRITIQUE_PROMPT = """<tool_response>
# [Action Result] The input image is the result of tool_call.

# [Critique Task] At the beginning of this new turn, your first step is to perform a critical review.
# 1. Bring your intermediate analysis back to the original question for a reverse check.
# 2. Specifically verify that the bar chart’s y-axis scale and each bar’s y-value match the provided image.
# 3. Confirm that your results align with the problem image; if anything doesn’t match, correct it.
# 4. Thinking one more step for answering the question.
# </tool_response>"""



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
    # check <think> and <tool_call> tags是否都是成对的
    if text.count('<think>') == 1 and text.count('</think>') == 0:
        # 找到第一个<tool_call>，并在前面添加</think>
        tool_call_index = text.find('<tool_call>')
        if tool_call_index != -1:
            text = text[:tool_call_index] + '</think>' + text[tool_call_index:]
    # 提取thinking_content (<think>标签内容)
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    reasoning_content = think_match.group(1).strip() if think_match else None
    # 提取tool_calls (<tool_call>标签内容)
    tool_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_matches = re.finditer(tool_pattern, text, re.DOTALL)
    tool_calls = []
    for match in tool_matches:
        try:
            # 尝试解析JSON内容
            tool_content = match.group(1).strip()
            tool_json = json.loads(tool_content)
            test_name = tool_json['name']  
            tool_calls.append(tool_json)
        except json.JSONDecodeError:
            # tool_calls.append(tool_content)
            pass
        
    # 移除所有标签内容，获取剩余文本
    clean_text = text
    clean_text = re.sub(think_pattern, '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(tool_pattern, '', clean_text, flags=re.DOTALL)
    content = clean_text.strip()

    # if(len(tool_calls) != 0):
    #     content = ''

    return {
        'raw_response': text,
        'reasoning_content': reasoning_content,
        'tool_calls': tool_calls,
        'content': content,
        "seed_content": text.split("</think>")[-1].strip()
    }

def process_tool_call(tool_call) -> Dict[str, Any]:
    print('** tool_call: ', tool_call)
    action = tool_call['name']
    attrs = tool_call['arguments']
    
    return {
        "action": action,
        "attrs": attrs
    }


class OpenAIModel(LLM):
    def __init__(
            self,
            model_name,
            temperature=0.9,
            top_p=0.9,
            max_length=32768,
            generation_max_length=2048,
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
        self.your_logid = logid.generate()

        self.blackboard = None
        self.provided_tools = blackboard_tools
        self.system_prompt = SYSTEM_PROMPT.format(provided_tools=self.provided_tools)

        


        if "claude" in self.model_name:
            self.model = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],  # Your Anthropic API key
                base_url="https://api.anthropic.com/v1/"  # Anthropic's API endpoint
            )
        else:
            # self.model = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = openai.AzureOpenAI(
                azure_endpoint=os.environ["BASE_URL"],
                api_version=os.environ["API_VERSION"],
                api_key=os.environ["OPENAI_API_KEY"]
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
        
        self.blackboard = Blackboard()
        if inputs["messages"][0]["role"] == "user":
            inputs["messages"].insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        
        # if inputs["messages"][0]["role"] == "user":
        #     ori_content = inputs["messages"][0]["content"][0]['text']
        #     inputs["messages"][0]["content"][0]['text'] = self.system_prompt + '\n' + ori_content
        
        data_idx = data_ori['pid']
        print('='*10, f'** data_idx: {data_idx}')
        print('** question: ', data_ori['question'])
        
        tool_rounds = 0
        finish_reason = None
        response = None
        while tool_rounds <= 5 and finish_reason != "stop":
            response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
            parsed_response = parse_response(response['output'])
            # parsed_response.pop("reasoning_content", None)
            print('** parsed_response: ', parsed_response)

            if len(parsed_response["seed_content"]) == 0:
                parsed_response["seed_content"] = parsed_response['reasoning_content']
            tool_calls = parsed_response["tool_calls"]
            if len(tool_calls) > 0:
                parsed_response["seed_content"] = parsed_response["seed_content"].split("<answer>")[0]
            # print('** parsed_response["seed_content"]: ', parsed_response["seed_content"])
            inputs["messages"].append({"role": "assistant", "content": [{"type": "text", "text": parsed_response["seed_content"]}]})
            if len(tool_calls) > 0:
                tool_tasks = [process_tool_call(tool_call) for tool_call in tool_calls]
                for tool_id, tool_task in enumerate(tool_tasks):
                    try:
                        img_tmp_save_path = f"blackboard_output/data_{data_idx}/tool_rounds_{tool_rounds}-tool_id{tool_id}-action_{tool_task['action']}.png"
                        
                        self.blackboard.update_state(**tool_task)
                        render_result = self.blackboard.render_state(img_tmp_save_path)
                        new_content = []
                        print('** render_result: ', render_result)
                        if render_result != "tool execute success":
                            new_content.append({
                                "type": "text",
                                "text": f"<tool_response>{render_result}</tool_response>"
                            })
                            inputs["messages"].append({"role": "user", "content": new_content})
                        else:
                            image_url = image_input(img_tmp_save_path)
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            })
                            critical_inputs = copy.deepcopy(inputs)
                            critical_inputs["messages"] = []
                            critical_content = []
                            if(len(data_ori['image_list']) > 0):
                                critical_inputs["messages"].append({"role": "system", "content": [{"type": "text", "text": CRITIQUE_SYSTEM}]})
                                
                                image_url_ori = image_input(data_ori['image_list'][0])
                                critical_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url_ori},
                                })
                                critical_content.append({
                                    "type": "text",
                                    "text": data_ori['question'],
                                })
                            else:
                                critical_inputs["messages"].append({"role": "system", "content": [{"type": "text", "text": CRITIQUE_SYSTEM_wo_IMG}]})
                                critical_content.append({
                                    "type": "text",
                                    "text": data_ori['question'],
                                })

                            critical_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            })
                            critical_inputs["messages"].append({"role": "user", "content": critical_content})
                            # print('** critical_inputs: ', critical_inputs)
                            critical_output = self.generate(inputs=critical_inputs, **kwargs)
                            # print('** critical_output: ', critical_output)
                            critical_response = critical_output['output']

                            new_content.append({
                                "type": "text",
                                "text": CRITIQUE_PROMPT.format(critical_check=critical_response)
                            })
                            # inputs["messages"].append({"role": "tool", **parsed_response})
                            print('** render success.')
                            inputs["messages"].append({"role": "user", "content": new_content})
                    except:
                        # inputs["messages"].append({"role": "tool", **parsed_response})
                        inputs["messages"].append({"role": "user", "content": f"<tool_response>Failed to execute the tool call. {tool_task['action']}</tool_response>"})
            
                tool_rounds += 1
            else:
                # inputs["messages"].append({"role": "assistant", **parsed_response})
                break
            if len(tool_calls) > 4:
                break
        # exit(0)
        if response is not None and response['output'].find("</answer>") == -1:
            print('** give final answer')
            if(inputs["messages"][-1]["role"] == "user"):
                inputs["messages"].pop()
            inputs["messages"].append({"role": "user", "content": "Directly give the final answer of the user question with <answer>...</answer>."})
            retry_number = 0
            while response['output'].find("</answer>") == -1:
                response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
                response['output'] = parse_response(response['output'])['content']
                retry_number += 1 
                if(retry_number > 3):
                    break
        return response




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
        
        print('** output: ', output.choices[0].message.content)
        # print('** output: ', output.choices[0].message)
        
        if output is not None and output.choices[0].message.content is not None:
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": save_prompt,
            }
        return {"output": "", "input_len": -1, "output_len": -1, "input_text": save_prompt}
