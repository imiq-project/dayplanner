FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOTCO_TORCH_NUM_THREADS=1 \
    DYCONET_DEVICE=cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY DynamicTopology_Symmetric_HOTCO_Grossberg_baseline.py .
COPY config.yaml .
COPY parsers ./parsers

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "--threads", "1", "--timeout", "120", "main:app"]
