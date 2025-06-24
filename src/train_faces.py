import cv2
import os
import numpy as np
import pickle

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"  # opcional, para avaliação
MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.pickle"

images, labels, label_names = [], [], {}
current_id = 0

# Carregar dados de treino
for person in os.listdir(TRAIN_DIR):
    person_dir = os.path.join(TRAIN_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    label_names[current_id] = person

    for pose_folder in os.listdir(person_dir):
        pose_dir = os.path.join(person_dir, pose_folder)
        if not os.path.isdir(pose_dir):
            continue

        for img_name in os.listdir(pose_dir):
            img_path = os.path.join(pose_dir, img_name)
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

# --- Opcional: função simples para avaliar no conjunto de teste ---
def avaliar_modelo(recognizer, label_names, test_dir=TEST_DIR):
    import numpy as np

    total = 0
    corretos = 0
    for label_id, person in label_names.items():
        person_dir = os.path.join(test_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for pose_folder in os.listdir(person_dir):
            pose_dir = os.path.join(person_dir, pose_folder)
            if not os.path.isdir(pose_dir):
                continue

            for img_name in os.listdir(pose_dir):
                img_path = os.path.join(pose_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                label_predito, conf = recognizer.predict(img)
                total += 1
                if label_predito == label_id:
                    corretos += 1

    acc = (corretos / total) * 100 if total > 0 else 0
    print(f"Acurácia no conjunto de teste: {acc:.2f}% ({corretos}/{total})")

# Exemplo de uso da avaliação:
avaliar_modelo(recognizer, label_names)
