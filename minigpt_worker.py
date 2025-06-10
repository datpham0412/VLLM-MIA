import argparse
import base64
import io
import logging
import threading
import time
import uuid
import socket

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import StoppingCriteriaList
import requests

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (
    Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
)

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--cfg-path", required=True)
parser.add_argument("--controller-address", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--options", nargs="+", help="Overrides for config", default=[])
args = parser.parse_args()

# -------------------------
# Setup
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

worker_id = str(uuid.uuid4())[:6]
ip = socket.gethostbyname(socket.gethostname())
worker_name = f"http://{ip}:{args.port}"

logging.info(f"MiniGPT-4 worker {worker_id} using port {args.port}")

cfg = Config(args)
model_config = cfg.model_cfg
model_config.device_8bit = 0
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda")

conv_dict = {
    "pretrain_vicuna0": CONV_VISION_Vicuna0,
    "pretrain_llama2": CONV_VISION_LLama2,
}
CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to("cuda") for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device="cuda", stopping_criteria=stopping_criteria)

# -------------------------
# Controller Registration (moved into FastAPI startup event)
# -------------------------
@app.on_event("startup")
async def on_startup():
    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": worker_name,
        "check_heart_beat": True,
        "worker_status": {
            "model_names": [args.model_name],
            "speed": 1,
            "queue_length": 0
        }
    }
    while True:
        try:
            requests.post(url, json=data, timeout=5)
            logging.info(f"Registered MiniGPT-4 ({args.model_name}) to controller at {args.controller_address}")
            break
        except Exception as e:
            logging.warning(f"Failed to register. Retrying in 5s. Error: {e}")
            time.sleep(5)

    def heartbeat_loop():
        hb_url = args.controller_address + "/receive_heart_beat"
        while True:
            try:
                requests.post(hb_url, json={"worker_name": worker_name, "queue_length": 0})
                logging.info("Heartbeat sent to controller")
            except Exception as e:
                logging.warning(f"Heartbeat error: {e}")
            time.sleep(15)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

# -------------------------
# FastAPI Endpoints
# -------------------------
def decode_image(base64_str):
    image_data = base64.b64decode(base64_str.split(",")[-1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.post("/worker_generate_stream")
async def generate(request: Request):
    req = await request.json()
    prompt = req["prompt"]
    image = decode_image(req["images"][0])

    conv = CONV_VISION.copy()
    img_list = []

    chat.upload_img(image, conv, img_list)
    chat.encode_img(img_list)
    chat.ask(prompt, conv)

    output = chat.answer(
        conv=conv,
        img_list=img_list,
        num_beams=1,
        temperature=1.0,
        max_new_tokens=300,
        max_length=2000
    )[0]

    logging.info(f"Processed prompt: {prompt[:50]}...")

    return JSONResponse({
        "text": prompt + "\n" + output,
        "error_code": 0
    })

@app.get("/health")
def health():
    return {"status": f"MiniGPT-4 worker '{args.model_name}' running"}

@app.post("/worker_get_status")
async def worker_get_status():
    return JSONResponse({
        "model_names": [args.model_name],
        "speed": 1,
        "queue_length": 0
    })

# -------------------------
# Launch FastAPI Server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
