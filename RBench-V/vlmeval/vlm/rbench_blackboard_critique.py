import os
import uuid
import logid
import json
import logging
from vlmeval.smp import *
try:
    from CanvasModule.openai_model_v9 import OpenAIModel as MYOpenAIModel
except ImportError:
    # Fallback if CanvasModule is not in path, though it should be if running from root
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from CanvasModule.openai_model_v9 import OpenAIModel as MYOpenAIModel

class RBenchBlackboardCritique:
    is_api: bool = True

    def __init__(self, model_name, api_key=None, base_url=None, **kwargs):
        self.model_name = model_name
        # Set env vars for the inner model as it reads from env.
        # If not explicitly provided, fall back to existing env vars.
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("BASE_URL")
        
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["BASE_URL"] = base_url
        os.environ["API_VERSION"] = kwargs.get("api_version", "2024-03-01-preview")  
            
        self.your_logid = logid.generate()  
        os.environ["LOGID"] = self.your_logid  
        
        self.inner_model = MYOpenAIModel(
            model_name=model_name,
            **kwargs
        )
        self.kwargs = kwargs

    def set_dump_image(self, dump_image_func):
        # This method is required by VLMEvalKit for some models, 
        # but RBenchBlackboard handles images internally or via message paths.
        # We can just store it or ignore it if not needed.
        self.dump_image_func = dump_image_func

    def generate(self, message, dataset=None):
        # message is a list of dicts: [{'type': 'image', 'value': path}, {'type': 'text', 'value': text}]
        
        # Extract images and text
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        question = "\n".join(texts)
        
        # Construct test_item and data
        image_placeholders = "<image>" * len(images)
        if image_placeholders:
            question = image_placeholders + "\n" + question

        test_item = {
            "image_list": images,
            "question": question
        }
        
        # Dummy templates required by prepare_inputs
        data_template = {
            "user_template": "{question}",
            "system_template": "You are a helpful assistant." 
        }
        
        inputs = self.inner_model.prepare_inputs(test_item, data_template)
        
        # Generate a unique PID for this inference to avoid collision in blackboard outputs
        pid = str(uuid.uuid4())[:8]
        
        # Determine save directory
        dataset_name = dataset if dataset else "unknown_dataset"
        save_dir = os.path.join("blackboard_output_vlmeval", f"{dataset_name}_v9_gemini", pid)
        os.makedirs(save_dir, exist_ok=True)
        
        data_ori = {
            "pid": pid,
            "question": question,
            "save_dir": save_dir,
            "image_list": images,
            "logid": self.your_logid
        }
        
        try:
            # Call generate_roll
            response = self.inner_model.generate_roll(inputs=inputs, data_ori=data_ori)
            
            # Save full history log
            log_path = os.path.join(save_dir, "inference_log.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pid": pid,
                    "dataset": dataset_name,
                    "question": question,
                    "history": response.get('history', []),
                    "final_output": response['output'],
                    "logid": self.your_logid,
                }, f, indent=2, ensure_ascii=False)
                
            return response['output']
        except Exception as e:
            logging.error(f"Error in RBenchBlackboardCritique generate: {e}")
            return f"Error: {str(e)}"
