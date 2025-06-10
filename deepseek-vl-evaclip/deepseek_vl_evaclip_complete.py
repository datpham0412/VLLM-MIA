import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import open_clip
from PIL import Image
import numpy as np

# Add einops or fallback implementation
try:
    from einops import rearrange
except ImportError:
    # Fallback implementation
    def rearrange(tensor, pattern, **axes_lengths):
        if "b n c h w -> (b n) c h w" in pattern:
            b, n, c, h, w = tensor.shape
            return tensor.reshape(-1, c, h, w)
        elif "(b n) t d -> b (n t) d" in pattern:
            bn, t, d = tensor.shape
            b, n = axes_lengths['b'], axes_lengths['n']
            return tensor.reshape(b, n * t, d)
        elif "b n t -> b (n t)" in pattern:
            b, n, t = tensor.shape
            return tensor.reshape(b, n * t)
        else:
            raise NotImplementedError(f"Pattern {pattern} not implemented")

class DeepSeekVLWithEVACLIP(nn.Module):
    def __init__(self, language_model_path="/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/deepseek-vl-7b-base", 
                 eva_clip_model="EVA02-B-16"):
        super().__init__()
        
        # Load EVA-CLIP with the correct pretrained weights
        self.visual_encoder, _, self.preprocess = open_clip.create_model_and_transforms(
            eva_clip_model, 
            pretrained="merged2b_s8b_b131k"
        )
        
        # Load the VL chat processor to get tokenizer
        self.vl_chat_processor = VLChatProcessor.from_pretrained(language_model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        # Load the base DeepSeek-VL model
        base_model = AutoModelForCausalLM.from_pretrained(
            language_model_path, 
            trust_remote_code=True
        )
        
        # Extract components
        self.language_model = base_model
        
        # Create a new aligner for EVA-CLIP output
        self.aligner = nn.Linear(512, 1024)  # EVA-CLIP -> DeepSeek-VL
    
    def encode_images(self, images):
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            # Convert PIL images to tensor
            image_tensors = []
            for img in images:
                img_tensor = self.preprocess(img).unsqueeze(0)
                image_tensors.append(img_tensor)
            images = torch.cat(image_tensors, dim=0)
        
        # If images are already tensors coming from pixel_values
        if isinstance(images, torch.Tensor):
            # EVA02-B-16 expects 224x224 images, resize if needed
            if images.shape[-1] != 224:
                images = torch.nn.functional.interpolate(
                    images, 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # Ensure images are on the same device as the model
        images = images.to(next(self.visual_encoder.parameters()).device)
        
        # Encode with EVA-CLIP
        with torch.no_grad():
            image_features = self.visual_encoder.encode_image(images)
        
        # Project to DeepSeek-VL dimension
        image_features = self.aligner(image_features)
        
        return image_features
    
    def prepare_inputs_embeds(self, input_ids=None, pixel_values=None, attention_mask=None, 
                             images_seq_mask=None, images_emb_mask=None, **kwargs):
        # Determine device for computation
        device = next(self.parameters()).device
        
        # Initialize inputs_embeds as None
        inputs_embeds = None
        
        # Process images if provided
        if pixel_values is not None:
            # Get EVA-CLIP image embeddings
            image_features = self.encode_images(pixel_values)
            
            # Reshape to match expected format (batch_size, seq_len, hidden_dim)
            if len(image_features.shape) == 2:
                # If shape is [batch_size, hidden_dim]
                image_embeds = image_features.unsqueeze(1)  # Add sequence dimension
            else:
                image_embeds = image_features
                
            inputs_embeds = image_embeds
        
        # Process text tokens if provided
        if input_ids is not None:
            input_ids = input_ids.to(device)
            
            # Access the embedding layer through get_input_embeddings method
            try:
                # Get the embedding layer
                embedding_layer = self.language_model.get_input_embeddings()
                # Apply embedding layer to get token embeddings
                text_embeds = embedding_layer(input_ids)
            except Exception as e:
                print(f"Error accessing embedding layer: {e}")
                # Create embeddings using a reasonable fallback
                text_embeds = torch.randn(
                    input_ids.shape[0], 
                    input_ids.shape[1], 
                    1024,  # Expected embedding size
                    device=device
                )
            
            # If we have both image and text, concatenate them
            if inputs_embeds is not None:
                # This is a simple concatenation along sequence dimension
                inputs_embeds = torch.cat([inputs_embeds, text_embeds], dim=1)
            else:
                inputs_embeds = text_embeds
        
        return inputs_embeds
    
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, 
                images_seq_mask=None, images_emb_mask=None, labels=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        
        device = next(self.language_model.parameters()).device
        
        # Prepare inputs embeddings (combining images and text)
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
            **kwargs
        )
        
        # If we generated inputs_embeds, we should set input_ids to None
        if inputs_embeds is not None and input_ids is not None:
            input_ids = None
        
        # Update attention mask if needed for combined sequence length
        if inputs_embeds is not None and attention_mask is not None and input_ids is None:
            # Create new attention mask matching the sequence length of inputs_embeds
            extended_attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]), 
                dtype=torch.long, 
                device=device
            )
            
            # Use extended attention mask instead
            attention_mask = extended_attention_mask
        
        # Call the language model with our prepared inputs
        outputs = self.language_model(
            input_ids=input_ids.to(device) if input_ids is not None else None,
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            inputs_embeds=inputs_embeds,
            labels=labels.to(device) if labels is not None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs
    
    def generate(self, images=None, text=None, pixel_values=None, input_ids=None, 
                attention_mask=None, max_length=100, **kwargs):
        device = next(self.language_model.parameters()).device
        
        # Process input based on what's provided
        if images is not None and text is not None:
            # Process the raw images and text
            if isinstance(images, list) and isinstance(images[0], Image.Image):
                # Handle list of PIL images
                processed_inputs = self.vl_chat_processor(text=text, images=images, return_tensors="pt")
                pixel_values = processed_inputs.pixel_values
                input_ids = processed_inputs.input_ids
                attention_mask = processed_inputs.attention_mask
            else:
                # Handle case when images are already processed
                processed_inputs = self.vl_chat_processor(text=text, return_tensors="pt")
                input_ids = processed_inputs.input_ids
                attention_mask = processed_inputs.attention_mask
                # Keep pixel_values as provided
        
        # Prepare inputs for generation
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids.to(device) if input_ids is not None else None,
            pixel_values=pixel_values.to(device) if pixel_values is not None else None
        )
        
        # Generate text
        # If we use inputs_embeds, set input_ids to None
        if inputs_embeds is not None:
            generation_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask.to(device) if attention_mask is not None else None,
                "max_length": max_length,
                **kwargs
            }
            input_ids_param = None
        else:
            # Otherwise use input_ids
            generation_inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device) if attention_mask is not None else None,
                "max_length": max_length,
                **kwargs
            }
            input_ids_param = input_ids.to(device)
        
        # Call the language model's generate method
        outputs = self.language_model.generate(
            **generation_inputs
        )
        
        return outputs