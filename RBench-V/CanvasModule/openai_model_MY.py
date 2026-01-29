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

from .blackboard_tools import blackboard_tools
from .blackboard import Blackboard


SYSTEM_PROMPT = """
# Objective #
Your are an **Visual-Reasoning Agent**, solving complex problems by synchronizing a visual Chain-of-Thought on a virtual notebook. The primary goal is **100% accuracy**.

# Process #
# Step 1: Think One Step
- **Action**: Thinking one more step for answering the question.
- **Output**: Enclose the entire thinking process within `<think>` tags.
- **Rule**: Do not give answer directly. (First of all, describe the image precisely.)
    - Remember that each step should contain only a small part of the reasoning process, avoiding outputting long paragraphs of reasoning at once. For example: analyze A in one step, analyze B in one step, analyze C in one step, set up equations in one step, and perform calculations in one step.
    * Strictly avoid delivering a lengthy explanation before presenting the notes.

# Step 2: Tool Call
- **Trigger**: Immediately after a `<think>` block is complete.
- **Action**: Call the appropriate **Notebook Tool** to visually record the key **evidence, data points, or intermediate results** from your thinking step. This synchronizes the internal thought process with the external visual memory.
- **Output**: Enclose the tool function call within `<tool_call>` tags.
- **Rule**: Updates should be incremental. **Instead of only showing a final answer (e.g., '3'), first visualize the components that lead to it.** For example, if you identify three items, use `insert_element` to list those items on the notebook *before* presenting the final count.

# Step 3: Iterate
- **Trigger**: Current evidence is insufficient for a **100% accuracy** answer.
- **Action**: Identify missing information, reformulate queries, and Repeat **Step 1 (Think)** and **Step 2 (Tool Call)** .
- **Process**: Each loop should build upon the previous state on the notebook, progressively moving towards the solution.

# Step 4: Final Answer
- **Trigger**: Current evidence is sufficient for a **100% accuracy** answer.
- **Action**: State the final, correct answer clearly and concisely.
- **Output**: Enclose the final answer within `<answer>` tags.


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


# Core Principles #
1.  **Visualize to Solve**: Do not rely solely on textual reasoning. Your thought process is incomplete until its key components are externalized on the notebook. Geometric, spatial, or logical relationships **must** be drawn. **Key findings, identified objects, and intermediate calculations must be recorded.**
2.  **Visual Validation**: Continuously use the visual state on the notebook to check for contradictions and verify intermediate steps.
3.  **Precision**: All coordinates and values used in tool calls must precisely match the calculations in your reasoning steps.
4.  **No content overlap**: When adding any content to the notebook, ensure careful layout to avoid overlapping or covering existing content. If needed, remove or adjust old content before adding new ones.  
5.  **Traceable Visual Reasoning**: The notebook is not just a canvas for the final answer; it is a **step-by-step log of your reasoning path**. Each significant finding in your `<think>` block must have a corresponding visual representation on the board. This creates a traceable audit trail that makes your work verifiable and easy to debug.


# Notebook Operation Restrictions #
# Overall Layout & Width Limitations
- The notebook area has a fixed width of `500px`; all internal elements must not exceed `500px` in width.  
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

# Visual Semantics
Use colors to represent logical states consistently.
- **Black/Grey (#000, #666)**: Static constraints, given information, or background elements.
- **Blue (#1377EB)**: The current element of focus, a hypothesis being tested, or an active calculation.
- **Green (#009E5F)**: A verified truth, a correct intermediate step, or the final result.
- **Red (#ED2633)**: A detected contradiction, an error in reasoning, or an important warning.
- **Dashed Lines**: Auxiliary lines, imaginary paths, or construction guides.
"""



CRITIQUE_PROMPT = """<tool_response>
[Action Result] The first image is the result of tool_call. The second image is the original question.

[Critique Task] At the beginning of this new turn, your first step is to perform a critical review.
1. Bring your intermediate analysis back to the original question for a reverse check.
2. Specifically verify that the bar chart’s y-axis scale and each bar’s y-value match the provided image.
3. Confirm that your results align with the problem image; if anything doesn’t match, correct it.
</tool_response>"""


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
        'content': content
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
        self.your_logid = str(uuid.uuid4())

        self.blackboard = None
        self.provided_tools = blackboard_tools
        self.system_prompt = SYSTEM_PROMPT.format(provided_tools=self.provided_tools)

        


        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        if "claude" in self.model_name:
            self.model = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.anthropic.com/v1/",
            )
        else:
            self.model = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
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
        # if inputs["messages"][0]["role"] == "user":
        #     inputs["messages"].insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        self.blackboard = Blackboard()
        if inputs["messages"][0]["role"] == "user":
            ori_content = inputs["messages"][0]["content"][0]['text']
            # print('** ori_content: ', inputs["messages"][0]["content"])
            inputs["messages"][0]["content"][0]['text'] = self.system_prompt + '\n' + ori_content
        
        data_idx = data_ori['pid']
        print('='*10, f'** data_idx: {data_idx}')
        print('** question: ', data_ori['question'])
        
        tool_rounds = 0
        finish_reason = None
        response = None
        while tool_rounds <= 5 and finish_reason != "stop":
            response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
            parsed_response = parse_response(response['output'])
            print('** parsed_response: ', parsed_response)
            inputs["messages"].append({"role": "assistant", "content": [{"type": "text", "text": response['output']}]})
            tool_calls = parsed_response["tool_calls"]
            if len(tool_calls) > 0:
                tool_tasks = [process_tool_call(tool_call) for tool_call in tool_calls]
                for tool_task in tool_tasks:
                    try:
                        img_tmp_save_path = f"blackboard_output/data_{data_idx}/tool_rounds_{tool_rounds}-action_{tool_task['action']}.png"
                        
                        self.blackboard.update_state(**tool_task)
                        render_result = self.blackboard.render_state(img_tmp_save_path)
                        new_content = []
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

                            if(len(data_ori['image_list']) > 0):
                                image_url_ori = image_input(data_ori['image_list'][0])
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url_ori},
                                })

                            new_content.append({
                                "type": "text",
                                "text": CRITIQUE_PROMPT.format(tool_task['action'])
                                # "text": "The golden answer is 0. EXPLAN."
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
        # exit(0)
        if response is not None and response['output'].find("</answer>") == -1:
            inputs["messages"].append({"role": "user", "content": "Please output the final answer with <answer>...</answer>."})
            response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
            response['output'] = parse_response(response['output'])['content']
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
            # max_tokens=self.generation_max_length,
            max_tokens=1024*10,
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
