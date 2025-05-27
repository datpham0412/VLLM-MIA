import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer

from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Iterator
import requests
import time
import subprocess
from threading import Thread
import os

# Local weights location
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

class Predictor:
    def setup(self) -> None:
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            "/fred/oz402/tiend/VLLM-MIA/target_models/llava-v1.5-7b",
            model_name="llava-v1.5-7b",
            model_base=None,
            load_8bit=False,
            load_4bit=False
        )

    def predict(
        self,
        image: Path,
        prompt: str,
        top_p: float = 1.0,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()

        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=dict(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                streamer=streamer,
                use_cache=True
            ))
            thread.start()

            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()

def load_image(image_file):
    if image_file.startswith('http'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
