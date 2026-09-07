FROM python:3.11-slim

# libgomp1 is LightGBM's OpenMP runtime; build-essential is needed only while
# pip compiles wheels, so it is removed again to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# Copy project source. data/ and models/ arrive as docker-compose volumes, and
# .env is deliberately NOT copied — secrets are injected at run time via
# env_file, so they never become a layer in a shareable image.
COPY src/ ./src/
COPY notebooks/ ./notebooks/

EXPOSE 8000

# python:slim ships no curl, so the healthcheck uses the interpreter that is
# guaranteed to be present. A curl-based check here silently fails forever and
# leaves any service that waits on `service_healthy` stuck.
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
