import cv2
import os
import numpy as np
import pickle

DATA_DIR = "data"
MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.pickle"

images, labels, label_names = [], [], {}
current_id = 0

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    label_names[current_id] = person
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(current_id)
    current_id += 1

print(f"Treinando com {len(images)} imagens…")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save(MODEL_FILE)

with open(LABELS_FILE, "wb") as f:
    pickle.dump(label_names, f)

print("Modelo salvo em", MODEL_FILE)
