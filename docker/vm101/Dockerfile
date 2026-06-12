FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "supertonic" "soundfile>=0.12" "psutil" "numpy"

COPY models/supertonic3_final/ /root/.cache/supertonic3/
COPY tts_server.py .

EXPOSE 8020

CMD ["python", "tts_server.py"]
