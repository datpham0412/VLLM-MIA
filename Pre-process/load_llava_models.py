import os
import sys
sys.path.append("/fred/oz402/aho/VLLM-MIA/")
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_llama import LlavaConfig

from typing import Type, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

def load_or_download_custom_model(
    folder: str,
    remote: str,
    model_class: Type[PreTrainedModel],
    tokenizer_class: Type[PreTrainedTokenizer],
    model_use_device_map: bool = False,
    tokennizer_use_device_map: bool = False,
    **model_kwargs
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads a model/tokenizer from a local folder or downloads from Hugging Face.

    Args:
        folder (str): Local folder path to save/load the model.
        remote (str): Remote model name on Hugging Face (e.g., "meta-llama/Llama-2-7b-chat-hf").
        model_class (Type[PreTrainedModel]): Model class to load (e.g., AutoModelForCausalLM).
        tokenizer_class (Type[PreTrainedTokenizer]): Tokenizer class to load (e.g., LlamaTokenizer).
        use_device_map (bool): Whether to automatically place model on available device(s).
        model_kwargs: Additional kwargs passed to `from_pretrained()`.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: The tokenizer and model objects.
    """
    model_exists = os.path.exists(folder) and os.path.isfile(os.path.join(folder, "config.json"))

    if model_exists:
        print(f"🔍 Loading model from local folder: {folder}")
        tokenizer = tokenizer_class.from_pretrained(folder)
        model = model_class.from_pretrained(folder, device_map="auto" if model_use_device_map else None, **model_kwargs)
    else:
        print(f"🌐 Downloading model from remote: {remote}")
        tokenizer = tokenizer_class.from_pretrained(remote)
        model = model_class.from_pretrained(remote, device_map="auto" if tokennizer_use_device_map else None, **model_kwargs)

        print(f"💾 Saving model and tokenizer to: {folder}")
        os.makedirs(folder, exist_ok=True)
        tokenizer.save_pretrained(folder)
        model.save_pretrained(folder)

    return tokenizer, model


def download_clip_processor(local_path: str, remote_model_name: str = "openai/clip-vit-large-patch14-336"):
    if os.path.exists(os.path.join(local_path, "preprocessor_config.json")):
        print(f"Image processor already exists at: {local_path}")
        return
    print(f"Downloading image processor: {remote_model_name}")
    processor = CLIPImageProcessor.from_pretrained(remote_model_name)
    processor.save_pretrained(local_path)
    print(f"Saved image processor to: {local_path}")


def download_clip_model(local_path: str, remote_model_name: str = "openai/clip-vit-large-patch14-336"):
    if os.path.exists(os.path.join(local_path, "config.json")):
        print(f"Vision model already exists at: {local_path}")
        return
    print(f"Downloading vision model: {remote_model_name}")
    model = CLIPVisionModel.from_pretrained(remote_model_name)
    os.makedirs(local_path, exist_ok=True)
    model.save_pretrained(local_path)
    print(f"Saved vision model to: {local_path}")


if __name__ == "__main__":
    print("🚀 Starting model and processor download...")

    # === LLaVA model ===
    llava_local = "/fred/oz402/aho/VLLM-MIA/target_models/llava-v1.5-7b"
    llava_remote = "liuhaotian/llava-v1.5-7b"

    tokenizer, model = load_or_download_custom_model(
        folder=llava_local,
        remote=llava_remote,
        model_class=LlavaLlamaForCausalLM,
        tokenizer_class=AutoTokenizer,
        model_use_device_map=True,
    )

    # === CLIP Image Processor ===
    clip_path = "/fred/oz402/aho/VLLM-MIA/target_models/pretrained/clip-vit-large-patch14-336"
    # download_clip_processor(local_path=clip_path)

    # === CLIP Vision Model ===
    # download_clip_model(local_path=clip_path)

    print("🏁 All components downloaded and saved locally.")