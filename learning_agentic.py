from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from models_utils import score_aesthetic_tool

pil_img = Image.open("DSCF0677.JPG").convert("RGB")
probs, mean_score = score_aesthetic_tool(pil_img)
print(probs)
print(mean_score)

