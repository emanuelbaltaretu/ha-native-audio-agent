FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Intel GPU compute stack for OpenVINO GPU inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "supertonic" "soundfile>=0.12" "psutil" "numpy" "openvino"

COPY pyproject.toml README.md ./
COPY src ./src
COPY docker/vm101/tts_server.py ./tts_server.py

RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8020

CMD ["python", "tts_server.py"]
