import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from langchain_core.tools import tool
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.aesthetic_resnet import AestheticResNet50
from PIL import Image, ImageOps

MODEL_URL = "https://huggingface.co/icecram/aesthetic_ranker/resolve/main/best_aesthetic_model_gpt_torch.pth"
MODEL_PATH = "./models/best_aesthetic_model_gpt_torch.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model weights from {MODEL_URL} ...")
    torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH)
    print(f"Model downloaded and saved to {MODEL_PATH}")

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# No normalisation — used for the Grad-CAM RGB overlay
gradcam_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

def load_model():
    model = AestheticResNet50(pretrained=False)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()

def generate_gradcam(image_path: str) -> str:
    """Generate a Grad-CAM heatmap overlay on the image using the aesthetic model.

    Returns the file path of a JPEG with the heatmap blended onto the original image.
    Uses pytorch-grad-cam library with layer4[-1] as the target layer.
    """
    pil_img = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    input_tensor = gradcam_transform(pil_img).unsqueeze(0).to(DEVICE)
    rgb_img = np.float32(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

    target_layers = [MODEL.backbone.layer4[-1]]
    cam = GradCAM(model=MODEL, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Resize overlay back to the original image dimensions so aspect ratio matches
    overlay = Image.fromarray(visualization).resize(pil_img.size, Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    overlay.save(tmp.name)
    return tmp.name


@tool
def score_aesthetic(image_path: str) -> tuple[list, float]:
    """Score the aesthetic quality of an image with a distribution over scores 1-10 and a mean score."""
    
    pil_img = Image.open(image_path).convert("RGB")
    x = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    scores = np.arange(1, 11)
    mean_score = float((probs * scores).sum())
    return probs.tolist(), mean_score