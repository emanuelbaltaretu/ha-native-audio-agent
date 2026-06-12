FROM --platform=linux/arm64 python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Audio deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      alsa-utils \
      libasound2 \
      libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install supertonic
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "supertonic" "soundfile>=0.12"

# Copy optimized model (FINAL: vector_estimator INT8, rest FP32)
COPY models/supertonic3_final/ /root/.cache/supertonic3/

# Copy TTS server
COPY tts_server.py .

EXPOSE 8020

CMD ["python", "tts_server.py"]
