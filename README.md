# Reconhecimento Facial com OpenCV

Este projeto simples utiliza OpenCV para capturar imagens de rostos, treinar um modelo de reconhecimento facial e identificar pessoas em tempo real usando sua webcam local.

---

## Funcionalidades

- Captura de imagens faciais a partir da webcam local
- Treinamento de modelo LBPH para reconhecimento facial
- Reconhecimento em tempo real com indicação do nome e confiança
- Armazenamento organizado das imagens capturadas por pessoa

---

## Requisitos

- Python 3.6+
- OpenCV (`opencv-python` e `opencv-contrib-python`)
- NumPy
- pickle (biblioteca padrão do Python)

Ou utilize o Docker (veja abaixo).

## Uso

### 1. Capturar imagens do rosto

Execute o script `src/capture_faces.py` para capturar imagens da pessoa que deseja reconhecer:

```bash
python src/capture_faces.py
```

Digite o nome da pessoa quando solicitado e mantenha o rosto visível para a câmera. O script salvará até 50 imagens automaticamente.

---

### 2. Treinar o modelo

Após capturar imagens de todas as pessoas desejadas, execute o script de treinamento para criar o modelo:

```bash
python src/train_faces.py
```

Esse script lê as imagens na pasta `data/`, treina o modelo LBPH e salva o arquivo `face_model.yml` e o arquivo `labels.pickle`.

---

### 3. Reconhecer rostos em tempo real

Para rodar o reconhecimento facial ao vivo usando a webcam local:

```bash
python src/recognize_faces.py
```

O vídeo exibirá um retângulo verde em torno dos rostos reconhecidos com o nome e a confiança, ou "Desconhecido" para faces não reconhecidas.

---

## Utilizando com Docker

Você pode rodar todo o projeto facilmente usando Docker, sem precisar instalar dependências no seu sistema:

### 1. Build da imagem

```bash
docker build -t face-recognitor .
```

### 2. Executando os scripts

#### Capturar imagens:
```bash
docker run --rm -it --device=/dev/video0:/dev/video0 face-recognitor python src/capture_faces.py
```

#### Treinar modelo:
```bash
docker run --rm -it face-recognitor python src/train_faces.py
```

#### Reconhecer rostos:
```bash
docker run --rm -it --device=/dev/video0:/dev/video0 face-recognitor
```

> Obs: O parâmetro `--device=/dev/video0:/dev/video0` é necessário para dar acesso à webcam ao container.

---

## Configuração da webcam

Por padrão, o código utiliza a webcam local com índice `0`. Caso queira usar outra câmera (exemplo: webcam do celular via DroidCam), ajuste o índice no código.

---

## Licença

Este projeto é open-source e livre para uso e modificação.

---

## Autor

Matheus Silvano Pereira
