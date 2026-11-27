FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY maintidoc maintidoc
COPY data data
COPY .streamlit/config.toml .streamlit/config.toml
CMD ["python","-m","streamlit","run","app.py","--server.address=0.0.0.0","--server.headless=true"]
