# SmartFlow — Streamlit dashboard container.
# Build:  docker build -t smartflow .
# Run:    docker run -p 8501:8501 smartflow   → http://localhost:8501
#
# The processed data (data/processed/*.csv) and trained models (models/*.pkl)
# are committed, so the dashboard runs out of the box without the raw CSV or a
# training step. To retrain inside the container, mount the raw CSV and run
# `python src/model_training.py`.

FROM python:3.13-slim

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit's own health endpoint — no curl needed in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
