FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libgtk2.0-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/recognize_faces.py"] 