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

# Local default; cloud hosts (Render, etc.) inject their own $PORT, honoured below.
ENV PORT=8501
EXPOSE 8501

# Streamlit's own health endpoint — no curl needed in the slim image. Reads $PORT
# so the check matches whatever port the platform assigned.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8501'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/_stcore/health').status==200 else 1)"

# Shell form so ${PORT} expands. Bind 0.0.0.0:$PORT; CORS/XSRF off so Streamlit's
# websocket connects through a cloud reverse proxy.
CMD streamlit run dashboard/app.py \
    --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true \
    --browser.gatherUsageStats=false --server.enableCORS=false --server.enableXsrfProtection=false
