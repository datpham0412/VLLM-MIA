#!/usr/bin/env python3
# gradio_web_server_llama.py
"""
Minimal Gradio front-end for Llama-2-chat models only.
– Uses the “llama_2” prompt template
– Stops generation when the model emits either '</s>'  or 'USER:'
"""

import argparse, json, time, datetime, os, hashlib, requests
import gradio as gr
from llava.conversation import conv_templates, default_conversation
from llava.utils import build_logger, server_error_msg
from llava.constants   import LOGDIR
from llava.conversation import SeparatorStyle

logger = build_logger("llama_web", "llama_web.log")
headers = {"User-Agent": "Llama-2 Client"}

# --------------------------------------------------------------------------- helpers
def get_conv_log_filename():
    t = datetime.datetime.now()
    return os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")

def get_model_list(ctrl_url):
    requests.post(ctrl_url + "/refresh_all_workers")
    r = requests.post(ctrl_url + "/list_models")
    return r.json().get("models", [])

# --------------------------------------------------------------------------- chat loop
def chat(state, model_name, temperature, top_p, max_new_tokens, ctrl_url):
    # pick template once (always llama_2 here)
    if len(state.messages) == state.offset + 2:
        tmpl = conv_templates["llama_2"].copy()
        tmpl.append_message(tmpl.roles[0], state.messages[-2][1])
        tmpl.append_message(tmpl.roles[1], None)
        state = tmpl

    # find worker
    worker = requests.post(ctrl_url + "/get_worker_address",
                           json={"model": model_name}).json()["address"]
    if not worker:
        state.messages[-1][-1] = server_error_msg
        yield state, state.to_gradio_chatbot(); return

    prompt = state.get_prompt()

    # -------- stop-word logic: '</s>'  OR  'USER:'
    stop_word = "USER:"
    base_stop = state.sep if state.sep_style in [SeparatorStyle.SINGLE,
                                                 SeparatorStyle.MPT] else state.sep2
    pload = dict(model=model_name, prompt=prompt,
                 temperature=float(temperature), top_p=float(top_p),
                 max_new_tokens=min(int(max_new_tokens), 1536),
                 stop=[base_stop, stop_word])          # list → JSON array
    # ---------------------------------------------------------------------

    state.messages[-1][-1] = "▌"
    yield state, state.to_gradio_chatbot()

    resp = requests.post(worker + "/worker_generate_stream",
                         headers=headers, json=pload, stream=True, timeout=10)
    for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if not chunk:
            continue
        data = json.loads(chunk.decode())
        if data["error_code"] != 0:
            state.messages[-1][-1] = server_error_msg
            yield state, state.to_gradio_chatbot()
            return

        new_text = data["text"][len(prompt):]
        # ---------------- trim everything after the first "USER:" -----
        cut = new_text.find("USER:")
        if cut != -1:
            new_text = new_text[:cut]
            state.messages[-1][-1] = new_text.strip()
            yield state, state.to_gradio_chatbot()
            break                    # stop reading further chunks
        # ----------------------------------------------------------------
        state.messages[-1][-1] = new_text + "▌"
        yield state, state.to_gradio_chatbot()

# --------------------------------------------------------------------------- UI
def build_demo(ctrl_url):
    init_state = default_conversation.copy()
    with gr.Blocks(title="Llama-2 chat") as demo:
        state = gr.State(value=init_state)
        models = get_model_list(ctrl_url)
        model_sel = gr.Dropdown(models, label="Model")

        temp = gr.Slider(0,1,0.2,label="Temperature")
        topp = gr.Slider(0,1,0.7,label="Top-p")
        max_tok = gr.Slider(0,1024,512,step=64,label="Max new tokens")

        chatbox = gr.Chatbot(height=550)
        txt = gr.Textbox(show_label=False, placeholder="Type and press Enter")

        def add_user(s, t):
            if not t: s.skip_next=True; return s, s.to_gradio_chatbot(), ""
            s.append_message(s.roles[0], t[:1536]); s.append_message(s.roles[1], None)
            return s, s.to_gradio_chatbot(), ""

        txt.submit(add_user, [state, txt], [state, chatbox, txt]) \
           .then(chat, [state, model_sel, temp, topp, max_tok, gr.State(ctrl_url)],
                 [state, chatbox])

    return demo

# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--controller-url", default="http://localhost:21101")
    ap.add_argument("--model-list-mode", choices=["once", "reload"],
                default="reload")  
    args = ap.parse_args()

    build_demo(args.controller_url).launch(server_name=args.host, server_port=args.port)
