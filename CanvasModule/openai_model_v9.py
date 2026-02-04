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

from .blackboard_tools import blackboard_tools
from .blackboard import Blackboard


SYSTEM_PROMPT = """
# Objective #
Your are an **Visual-Reasoning Agent**, solving complex problems by synchronizing a visual Chain-of-Thought on a virtual notebook. The primary goal is **100% accuracy**.

# Special Handling for Physics Problems #
If the question involves physics (Mechanics, Kinematics, Dynamics, etc.):
1. **Identify Constraints Explicitly**: Before calculating, explicitly state the motion constraints for every moving part (e.g., "Point A slides on surface -> v_A tangent to surface", "Rod slides against corner B -> v_B along the rod").
2. **Verify Assumptions**: Do not assume standard positions (like "Instantaneous Center is at the origin") unless derived from velocity vectors.
3. **Cross-Check**: If your result depends entirely on a visual feature (like "it looks like a circle center"), pause and verify if the text supports it.

# Critical Instruction: Text over Vision #
**WARNING**: The provided image may be schematic or illustrative. **Do not rely solely on visual intuition.**
- If the text describes a physical constraint (e.g., "rod slides on rim"), you must model it physically (velocity along the rod), even if the image looks like a simple geometric shape.
- **Physics First**: Apply rigorous physical laws (Instantaneous Center, Newton's Laws) based on the *text description* of constraints, rather than guessing from the image appearance.

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

The results of the function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's question. Finally, if you have got the answer, enclose it within '\\boxed{{}}' tags. The answer should be in standard LaTeX formula format or numbers. Be careful not to mistake multiplication signs (dot product) as commas.
> After Tool Call, wait for the tool response.


# Notebook Operation Restrictions #
# Overall Layout & Width Limitations
- The notebook area has a fixed width of `800px`; all internal elements must not exceed `800px` in width.  
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
The notebook is an HTML container (**Width: 800px**, Height: Auto). You have 5 tools to manipulate it.
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
You are a high-precision visual analysis expert and a strict reasoning auditor. Your role is to compare the "Original Image" and "Question" with the "Model Inference Process State (Notebook State)."

### Task
1. **Visual Verification**: Identify "hallucination elements" in the Notebook State that are factually inconsistent with the Original Image.
2. **Reasoning Verification**: Scrutinize the text and formulas written on the Notebook. Check for **physical principle errors**, **mathematical derivation errors**, or **logical inconsistencies** with the problem context.

### Guidelines
1. **Visual Consistency**:
   - A hallucination is identified when the Notebook claims **non-existent objects**, **incorrect colors**, **wrong spatial relationships**, **incorrect counts**, or **incorrect actions**.
   - Ignore standard inference annotations (bounding boxes, auxiliary lines) unless they are placed incorrectly.

2. **Reasoning & Logic Check (Crucial)**:
   - **Physical Laws**: Are the applied physical principles (e.g., Newton's laws, conservation of energy, kinematic equations) correct for this specific scenario?
   - **Mathematical Derivation**: Are the formulas derived correctly? Are the calculations accurate?
   - **Contextual Logic**: Does the reasoning contradict the visual setup? (e.g., using a formula for a fixed pulley when the image shows a movable pulley).
   - **Plausibility Check**: Are the results physically reasonable? (e.g., Efficiency $\eta$ must be $< 100\%$; Friction coefficient $\mu$ usually $< 1$).
   - **Avoid Pedantry**: Do not flag missing intermediate steps as errors if the conclusion is physically sound. Focus on identifying *wrong* steps, not *skipped* steps. If an assumption (like symmetry) leads to a correct physical outcome, accept it.

3. **Text-First Verification (Anti-Hallucination)**:
   - If the problem text defines a constraint (e.g., "smooth surface", "light rod"), the Notebook MUST follow it, even if the image suggests otherwise (e.g., drawing texture).
   - **Physics Problem Special**: For physics problems, prioritize text descriptions of constraints and connections over ambiguous visual details. If the text implies a specific setup (e.g., "single movable pulley"), trust the text over a potentially schematic diagram.

4. **Conflict Type Definitions**:
   - **Visual Error**: Attribute, Existence, Spatial, or Quantity errors regarding the image content.
   - **Physical Error**: Misapplication of physical laws, incorrect force analysis, or wrong assumptions (e.g., assuming friction is zero when $f=0.5$).
   - **Math Error**: Calculation mistakes or incorrect algebraic manipulation.
   - **Logical Error**: Contradictions between steps or conclusions that don't follow from premises.

### Input Data
- **Original Image**: The objective standard of the real world.
- **Question**: The problem statement.
- **Notebook State**: The visualized intermediate state (contains reasoning steps, formulas, and diagrams).

### Output Format
Return the result strictly in JSON format. Use English.
{
    "hallucination_elements": [
        {
            "category": "Visual Error / Physical Error / Math Error / Logical Error",
            "original_fact": "The correct physical law / visual fact / calculation result",
            "notebook_claim": "The incorrect statement or formula in the Notebook",
            "explanation": "Detailed explanation of why this is incorrect."
        }
    ],
    "is_consistent": true/false (true ONLY if NO errors are found)
}

### Example
- **Input**: Notebook shows "Kinetic Energy = mv".
- **Output**: 
    {
        "hallucination_elements": [
            {
                "category": "Physical Error",
                "original_fact": "Kinetic Energy formula is 1/2 * mv^2",
                "notebook_claim": "Kinetic Energy = mv",
                "explanation": "The formula used for kinetic energy is dimensionally incorrect and violates standard physics definitions."
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
        "hallucination_elements": [
            {{
                "Original Question": <golden elements>,
                "Notebook": <error elements>,
                "Explanation": <simple explanation>
            }},
            ...
        ]
    }}
""" 




# compare
CRITIQUE_PROMPT = """<tool_response><image>This is the state of notebook. Critical Check: {critical_check}</tool_response>"""




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
        with open(img_url, "rb") as f:
            img_file = f.read()
        #  PIL 
        with Image.open(io.BytesIO(img_file)) as img:
            img_format = img.format
            img_base64 = base64.b64encode(img_file).decode("utf-8")
            image_url = f"data:image/{img_format.lower()};base64,{img_base64}"
            return image_url
        
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
            # Fix: Handle potential JSON escaping issues in tool arguments
            tool_content = tool_content.replace('\\n', '\\\\n') 
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

        self.blackboard = None
        self.provided_tools = blackboard_tools
        self.system_prompt = SYSTEM_PROMPT.format(provided_tools=self.provided_tools)

        


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
            if len(parsed_response["seed_content"]) == 0:
                parsed_response["seed_content"] = parsed_response['reasoning_content']
                
            tool_calls = parsed_response["tool_calls"]
            if len(tool_calls) > 0:
                # 保留原截断逻辑，但使用 \\boxed 标记 replace <answer>
                parsed_response["seed_content"] = parsed_response["seed_content"].split("\\boxed")[0]
            inputs["messages"].append({"role": "assistant", "content": [{"type": "text", "text": parsed_response["seed_content"]}]})   
            
            if len(tool_calls) > 0:
                tool_tasks = [process_tool_call(tool_call) for tool_call in tool_calls]
                last_image_url = None
                for tool_id, tool_task in enumerate(tool_tasks):
                    try:
                        if 'save_dir' in data_ori and data_ori['save_dir']:
                            save_base = data_ori['save_dir']
                        else:
                            save_base = f"blackboard_output/data_{data_idx}"
                        
                        if not os.path.exists(save_base):
                            os.makedirs(save_base, exist_ok=True)

                        img_tmp_save_path = os.path.join(save_base, f"tool_rounds_{tool_rounds}-tool_id{tool_id}-action_{tool_task['action']}.png")
                        
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
                            last_image_url = image_url
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            })
                            
                            print('** render success.')
                            inputs["messages"].append({"role": "user", "content": new_content})
                    except:
                        inputs["messages"].append({"role": "user", "content": f"<tool_response>Failed to execute the tool call. {tool_task['action']}</tool_response>"})
                
                # Perform critique after all tools in the batch are executed
                if last_image_url is not None:
                    try:
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
                            "image_url": {"url": last_image_url},
                        })
                        critical_inputs["messages"].append({"role": "user", "content": critical_content})
                        # print('** critical_inputs: ', critical_inputs)
                        critical_output = self.generate(inputs=critical_inputs, **kwargs)
                        # print('** critical_output: ', critical_output)
                        critical_response = critical_output['output']

                        critique_content = []
                        critique_content.append({
                            "type": "text",
                            "text": CRITIQUE_PROMPT.format(critical_check=critical_response)
                        })
                        inputs["messages"].append({"role": "user", "content": critique_content})
                    except Exception as e:
                        print(f"Critique generation failed: {e}")
            
                tool_rounds += 1
            else:
                # inputs["messages"].append({"role": "assistant", **parsed_response})
                break
            if len(tool_calls) > 8:
                break
        # exit(0)
        # Check for boxed answer in the CLEANED content first, then raw content
        # This allows answers inside tool arguments to be accepted if explicit text is missing
        parsed_final = parse_response(response['output'])
        has_boxed_content = parsed_final['content'].find("\\boxed{") != -1
        has_boxed_raw = response['output'].find("\\boxed{") != -1
        
        if response is not None and not has_boxed_content and not has_boxed_raw:
            print('** give final answer')
            if(inputs["messages"][-1]["role"] == "user"):
                inputs["messages"].pop()
            inputs["messages"].append({"role": "user", "content": "Please provide the final answer directly. The answer must be enclosed in \\boxed{}. Do not output any reasoning or SVG code."})
            retry_number = 0
            while True:
                response = self.generate(inputs=inputs, prompt=prompt, **kwargs)
                # Check the new response content
                new_parsed = parse_response(response['output'])
                if new_parsed['content'].find("\\boxed{") != -1:
                    break
                
                retry_number += 1 
                if(retry_number > 3):
                    break
        
        if response is not None:
            response['history'] = inputs["messages"]
            
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
