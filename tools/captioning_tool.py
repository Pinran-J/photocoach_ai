from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import tool

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True)

@tool
def caption_image(image_path: str) -> str:
    """Generate a caption for an image using BLIP image captioning model"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
