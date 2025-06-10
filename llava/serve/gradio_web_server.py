import argparse
import datetime
import json
import os
import time
 
import gradio as gr
import requests
 
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib

models = []


logger = build_logger("gradio_web_server", "gradio_web_server.log")
 
headers = {"User-Agent": "LLaVA Client"}
 
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
 
priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}
 
 
def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name
 
 
def get_model_list():

    try:
        logger.info(f"Sending refresh request to {args.controller_url}/refresh_all_workers")
        ret = requests.post(args.controller_url + "/refresh_all_workers")
        logger.info(f"Refresh status: {ret.status_code}, content: {ret.text}")
        ret.raise_for_status()

        logger.info(f"Sending list_models request to {args.controller_url}/list_models")
        ret = requests.post(args.controller_url + "/list_models")
        logger.info(f"List models response: {ret.status_code}")
        logger.info(f"List models raw content: {ret.text}")
        ret.raise_for_status()

        data = ret.json()
        logger.info(f"Decoded JSON: {data}")

        models = data.get("models", [])
        if not models:
            logger.warning("⚠️ get_model_list returned empty model list.")
        else:
            logger.info(f"✅ Models received: {models}")

        models.sort(key=lambda x: priority.get(x, x))
        return models
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ HTTP error in get_model_list: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error in get_model_list: {e}, raw response: {ret.text}")
        return []

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""
 
 
def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
 
    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)
 
    state = default_conversation.copy()
    return state, dropdown_update
 
 
def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"🔄 load_demo_refresh_model_list triggered by IP: {request.client.host}")
    model_list = get_model_list()
    logger.info(f"Model list retrieved: {model_list}")
    if not model_list:
        model_list = ["No models available"]
        logger.warning("Using fallback model list due to empty response.")

    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=model_list,
        value=model_list[0],
        visible=True,
        allow_custom_value=True  # prevent crash
    )
    return state, dropdown_update

def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")
 
 
def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3
 
 
def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3
 
 
def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3
 
 
def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5
 
 
def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5
 
 
def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5
 
    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5
 
 
def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
 
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return
 
    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if ("llava" in model_name.lower()) or (model_name.lower() == "bias-7b"):
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower() or model_name.lower() == "bias-7b":
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state
 
    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")
 
    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    prompt = state.get_prompt()
 
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    
    if "llama-2" in model_name.lower():
        # rebuild just for Llama-2 so we inherit the system instruction…
        tmpl = conv_templates["llama_2"].copy()
        tmpl.append_message(tmpl.roles[0], state.messages[-2][1])
        tmpl.append_message(tmpl.roles[1], None)
        full_prompt = tmpl.get_prompt()
        # strip only the literal labels, keep the rest of the template
        prompt = full_prompt.replace("USER:", "").replace("ASSISTANT:", "").strip()


    pload = {
       "model": model_name,
       "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
                  else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")
 
    pload['images'] = state.get_images()
 
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
 
    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return
 
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
 
    finish_tstamp = time.time()
    logger.info(f"{output}")
 
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

block_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Modern styling */
:root[data-theme="light"] {
    --primary-color: #2D2D2D;
    --secondary-color: #0EA5E9;
    --background-color: #FFFFFF;
    --text-color: #1F2937;
    --border-color: #E5E7EB;
    --chat-user-bg: #F3F4F6;
    --chat-bot-bg: #F8FAFC;
    --hover-color: #F3F4F6;
}

:root[data-theme="dark"] {
    --primary-color: #E5E7EB;
    --secondary-color: #60A5FA;
    --background-color: #1F2937;
    --text-color: #F9FAFB;
    --border-color: #374151;
    --chat-user-bg: #374151;
    --chat-bot-bg: #2D3748;
    --hover-color: #374151;
}

/* Global font settings */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
}

/* Theme switcher */
#theme-switcher {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 1000;
    display: flex;
    gap: 0.5rem;
    background: var(--background-color);
    padding: 0.5rem;
    border-radius: 2rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

#theme-switcher button {
    border: none;
    background: none;
    padding: 0.5rem;
    border-radius: 1.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

#theme-switcher button:hover {
    background: var(--hover-color);
}

#theme-switcher button.active {
    background: var(--secondary-color);
    color: white;
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
}

#component-0 {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

#chatbot {
    height: 600px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
    background: var(--background-color);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Custom chat styling */
#chatbot .message {
    padding: 1rem;
    margin: 0.5rem 1rem;
    border-radius: 0.5rem;
    position: relative;
    line-height: 1.5;
    max-width: 85%;
    clear: both;
    color: var(--text-color);
}

#chatbot .user {
    background: var(--chat-user-bg);
    margin-left: auto;
    border-bottom-right-radius: 0.2rem;
}

#chatbot .bot {
    background: var(--chat-bot-bg);
    margin-right: auto;
    border-bottom-left-radius: 0.2rem;
}

#chatbot .user:before {
    content: "👤";
    position: absolute;
    right: -25px;
    top: 0;
}

#chatbot .bot:before {
    content: "🤖";
    position: absolute;
    left: -25px;
    top: 0;
}

#component-0 h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-color);
    text-align: center;
    margin-bottom: 1rem;
}


#buttons button {
    min-width: min(120px,100%);
    border-radius: 8px;
    transition: all 0.2s ease;
    font-weight: 500;
    background-color: var(--background-color);
    color: var(--text-color);
    border: 1px solid var(--border-color);
}

#buttons button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    background-color: var(--hover-color);
}

#model_selector_row .gr-dropdown {
    border-radius: 8px;
    border: 1px solid var(--border-color);
    background-color: var(--background-color);
    color: var(--text-color);
}

.gr-button-primary {
    background: var(--secondary-color) !important;
    color: white !important;
}

.gr-button-secondary {
    border: 1px solid var(--border-color) !important;
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

/* Parameters styling */
#parameter_row .gr-form {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    background-color: var(--background-color);
}

.gr-form > div {
    margin: 0.5rem 0;
    color: var(--text-color);
}

/* Input box styling */
.gr-text-input {
    border-radius: 8px !important;
    border: 1px solid var(--border-color) !important;
    padding: 0.75rem !important;
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

.gr-text-input:focus {
    border-color: var(--secondary-color) !important;
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
}

/* Image upload area */
.gr-image-input {
    border-radius: 8px;
    border: 2px dashed var(--border-color);
    padding: 1rem;
    background-color: var(--background-color);
}

.gr-image-input:hover {
    border-color: var(--secondary-color);
}

/* Examples section */
.gr-examples {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    background-color: var(--background-color);
}

.gr-examples-title {
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--text-color);
}

/* Loading animation */
@keyframes typing {
    0% { content: "▌"; }
    25% { content: "▌▌"; }
    50% { content: "▌▌▌"; }
    75% { content: "▌▌"; }
    100% { content: "▌"; }
}

.typing-indicator::after {
    content: "▌";
    animation: typing 1s infinite;
}
"""

theme_switch_js = """
function toggleTheme() {
    const root = document.documentElement;
    const currentTheme = root.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', newTheme);
    
    // Update button states
    document.querySelectorAll('#theme-switcher button').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-theme') === newTheme) {
            btn.classList.add('active');
        }
    });
}

// Initialize theme
document.addEventListener('DOMContentLoaded', function() {
    const root = document.documentElement;
    root.setAttribute('data-theme', 'light');
    
    const switcher = document.createElement('div');
    switcher.id = 'theme-switcher';
    
    const lightBtn = document.createElement('button');
    lightBtn.innerHTML = '☀️';
    lightBtn.setAttribute('data-theme', 'light');
    lightBtn.classList.add('active');
    lightBtn.onclick = toggleTheme;
    
    const darkBtn = document.createElement('button');
    darkBtn.innerHTML = '🌙';
    darkBtn.setAttribute('data-theme', 'dark');
    darkBtn.onclick = toggleTheme;
    
    switcher.appendChild(lightBtn);
    switcher.appendChild(darkBtn);
    document.body.appendChild(switcher);
});
"""

title_markdown = ("""
# 🤖 AI Vision Assistant
A powerful multimodal AI that can see, understand, and communicate.
""")

tos_markdown = ("""
### Terms of Use
This is a research preview intended for non-commercial use only. By using this service, you agree to:
- Not use it for any illegal, harmful, violent, racist, or sexual purposes
- Be aware that it may generate offensive content despite safety measures
- Allow collection of dialogue data for research purposes
- Report inappropriate responses using the Flag button

For optimal experience, please use on desktop devices.
""")

learn_more_markdown = ("""
### About
This service is powered by state-of-the-art vision-language models. It is provided for research purposes only, subject to the [LLaMA License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md), [OpenAI Terms](https://openai.com/policies/terms-of-use), and [ShareGPT Privacy Policy](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb).
""")


def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(
        show_label=False, 
        placeholder="Ask me anything about the image...", 
        container=False,
        scale=8
    )
    
    with gr.Blocks(title="AI Vision Assistant", theme=gr.themes.Soft(), css=block_css) as demo:
        gr.HTML(f"<script>{theme_switch_js}</script>")
        state = gr.State()
 
        if not embed_mode:
            gr.Markdown(title_markdown)
 
        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="AI Vision Assistant",
                    height=700,
                    layout="panel"
                )
                
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary", size="lg")
                
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍 Helpful", variant="secondary", interactive=False)
                    downvote_btn = gr.Button(value="👎 Not Helpful", variant="secondary", interactive=False)
                    flag_btn = gr.Button(value="⚠️ Report", variant="secondary", interactive=False)
                    regenerate_btn = gr.Button(value="🔄 Regenerate", variant="secondary", interactive=False)
                    clear_btn = gr.Button(value="🗑️ Clear Chat", variant="secondary", interactive=False)

            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if models else None,
                        label="Select Model",
                        allow_custom_value=True,
                        interactive=True,

                        container=True
                    )
                
                imagebox = gr.Image(
                    type="pil",
                    label="Upload Image",
                    elem_id="image_upload"
                )
                
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Image Processing",
                    visible=False
                )

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                    examples=[
                        [f"{cur_dir}/examples/coffee-shop.jpg", "What's the atmosphere of this coffee shop and what kind of experience might customers expect?"],
                        [f"{cur_dir}/examples/tech-workspace.jpg", "Analyze this workspace setup and suggest any ergonomic improvements."],
                    ],
                    inputs=[imagebox, textbox],
                    label="Try these examples"
                )

                with gr.Accordion("Advanced Settings", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P"
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max Output Length"
                    )


        if not embed_mode:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(tos_markdown)
                with gr.Column():
                    gr.Markdown(learn_more_markdown)
                    
        url_params = gr.JSON(visible=False)
 
        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
 
        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )
 
        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )
 
        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )
 
        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )
 
        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
 
    return demo
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    logger.info("📦 Fetching models before starting demo...")
    models = get_model_list()
    logger.info(f"🧠 Final models used for UI: {models}")


    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
 