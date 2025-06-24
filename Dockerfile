# Usa uma imagem base com Python 3
FROM python:3.10-slim

# Instala dependências do sistema para OpenCV e acesso à webcam
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libgtk2.0-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão (pode ser alterado ao rodar o container)
CMD ["python", "src/recognize_faces.py"] 