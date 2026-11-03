# Face Recognition with OpenCV

A simple project that uses OpenCV to capture face images, train a face recognition model and identify people in real time through your local webcam.

<img width="640" height="480" alt="Recognition" src="https://github.com/user-attachments/assets/08ac6962-23d4-4ed3-8097-b5a192404e78" />

---

## Features

- Capture face images from the local webcam
- LBPH model training for face recognition
- Real-time recognition showing the name and confidence
- Captured images stored per person

---

## Requirements

- Python 3.6+
- OpenCV (`opencv-python` and `opencv-contrib-python`)
- NumPy
- pickle (Python standard library)

Or use Docker (see below).

## Usage

### 1. Capture face images

Run `src/capture_faces.py` to capture images of the person you want to recognize:

```bash
python src/capture_faces.py
```

Enter the person's name when prompted and keep your face visible to the camera. The script saves up to 50 images automatically.

---

### 2. Train the model

After capturing images of everyone you want to recognize, run the training script to build the model:

```bash
python src/train_faces.py
```

It reads the images in the `data/` folder, trains the LBPH model and saves `face_model.yml` and `labels.pickle`.

---

### 3. Recognize faces in real time

To run live face recognition on the local webcam:

```bash
python src/recognize_faces.py
```

The video shows a green rectangle around recognized faces with the name and confidence, or "Unknown" for faces it does not recognize.

---

## Running with Docker

You can run the whole project with Docker, without installing dependencies on your system:

### 1. Build the image

```bash
docker build -t face-recognitor .
```

### 2. Run the scripts

#### Capture images:
```bash
docker run --rm -it --device=/dev/video0:/dev/video0 face-recognitor python src/capture_faces.py
```

#### Train the model:
```bash
docker run --rm -it face-recognitor python src/train_faces.py
```

#### Recognize faces:
```bash
docker run --rm -it --device=/dev/video0:/dev/video0 face-recognitor
```

> Note: `--device=/dev/video0:/dev/video0` gives the container access to the webcam.

---

## Webcam configuration

By default the code uses the local webcam at index `0`. To use another camera (for example a phone webcam through DroidCam), change the index in the code.

---

## License

This project is open source and free to use and modify.

---

## Author

Matheus Silvano Pereira
