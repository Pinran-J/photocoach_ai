# ── PhotoCoach AI — Gradio App ────────────────────────────────────────────────
#
# Two models run in this container:
#   ResNet50 (~100MB)  — downloaded at BUILD time via wget (fast startup)
#   BLIP (~1GB)        — downloaded at RUNTIME via HuggingFace transformers
#                        (too large to bake in; cached to /app/models volume)
#
# API keys (OPENAI_API_KEY, PINECONE_API_KEY) are injected at runtime via
# docker-compose env_file or Kubernetes secrets — never baked into the image.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# System libraries:
#   libmagic1      → python-magic
#   libgl1         → OpenCV
#   libglib2.0-0   → OpenCV
#   wget           → model weight download during build
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only reruns if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (.dockerignore excludes .env, data/, lambda_package/, .pth)
COPY . .

# Download ResNet50 aesthetic scorer weights from HuggingFace during build.
# Baking them in means the container starts immediately without a network download.
RUN mkdir -p /app/models && \
    wget -q --show-progress \
        -O /app/models/best_aesthetic_model_gpt_torch.pth \
        https://huggingface.co/icecram/aesthetic_ranker/resolve/main/best_aesthetic_model_gpt_torch.pth

# BLIP and other HuggingFace models download here at runtime (cached via volume mount)
ENV TRANSFORMERS_CACHE=/app/models/hf_cache
ENV HF_HOME=/app/models/hf_cache

# Gradio must bind to 0.0.0.0 to be reachable outside the container
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
