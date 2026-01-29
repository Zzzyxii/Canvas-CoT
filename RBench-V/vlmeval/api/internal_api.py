import os
import base64
import requests
import uuid
from .base import BaseAPI


class InternalAPI(BaseAPI):
    is_api: bool = True
    
    def __init__(self,
                 model_name="api_openai_chatgpt-4o-latest",
                 model_marker="api_openai_chatgpt-4o-latest",
                 api_key="MY_API_KEY",
                 **kwargs):
        super().__init__(**kwargs)
        self.api_base = "http://trpc-utools-prod.turbotke.production.polaris:8009/"
        self.api_key = api_key or os.getenv('INTERNAL_API_KEY', 'MY_API_KEY')
        self.bid = "open_api_test"
        self.server = "open_api"
        self.bid_2 = "B端"
        self.bid_3 = "产品A"
        self.model_marker = model_marker
        self.model_name = model_name

    def encode_image(self, image_path):
        """将图像文件编码为Base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_messages(self, inputs):
        """准备API所需的消息格式"""
        messages = []
        for msg in inputs:
            if msg['type'] == 'text':
                messages.append({"type": "text", "value": msg['value']})
            elif msg['type'] == 'image':
                base64_image = self.encode_image(msg['value'])
                img_txt_prompt = f"data:image/jpeg;base64,{base64_image}"
                messages.append({"type": "image_url", "value": img_txt_prompt})
        return messages

    def generate_inner(self, inputs, **kwargs):
        """调用内部API生成响应"""
        content_list = self.prepare_messages(inputs)
        json_data = {
            "bid": self.bid,
            "server": self.server,
            "services": [],
            "bid_2": self.bid_2,
            "bid_3": self.bid_3,
            "request_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "api_key": self.api_key,
            "model_marker": self.model_marker,
            "system": kwargs.get("system_prompt", ""),
            "messages": [{"role": "user", "content": content_list}],
            "params": {},
            "general_params": {},
            "timeout": 300,
            "extension": {},
            "model_name": self.model_name,
        }

        try:
            response = requests.post(url=self.api_base, json=json_data, proxies={"http": None, "https": None})
            if response.status_code == 200:
                result = response.json()
                if "answer" in result and len(result["answer"]) > 0:
                    return 0, result["answer"][0]["value"], None
            return -1, self.fail_msg, response.text
        except Exception as e:
            return -1, self.fail_msg, str(e)