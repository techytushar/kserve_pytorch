FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt requirements.txt

RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY mnist mnist
COPY data/models models

CMD ["python", "mnist/serve.py"]
