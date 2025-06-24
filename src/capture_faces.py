import cv2
import os
import shutil
import random

name = input("Nome da pessoa: ").strip()
if not name:
    print("Nome vazio. Saindo.")
    exit(1)

base_train_dir = os.path.join("data", "train", name)
base_test_dir = os.path.join("data", "test", name)
os.makedirs(base_train_dir, exist_ok=True)
os.makedirs(base_test_dir, exist_ok=True)

cam_index = 0
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("Não foi possível abrir a webcam.")
    exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

poses = [
    "Olhe para frente",
    "Olhe para a esquerda",
    "Olhe para a direita",
    "Olhe para cima",
    "Olhe para baixo"
]
images_per_pose = 20
image_size = (200, 200)
test_split_ratio = 0.2

for pose_index, pose_instruction in enumerate(poses, start=1):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Pose {pose_index}/5:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, pose_instruction, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "Pressione ENTER para começar", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Captura de Faces", frame)

        key = cv2.waitKey(1)
        if key == 13:
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Captura interrompida.")
            exit(0)

    temp_pose_dir = os.path.join("data", "temp", name, f"pose_{pose_index}")
    os.makedirs(temp_pose_dir, exist_ok=True)

    count = 0
    while count < images_per_pose:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, image_size)

            filepath = os.path.join(temp_pose_dir, f"{count:03}.png")
            cv2.imwrite(filepath, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count += 1

            if count >= images_per_pose:
                break

        cv2.putText(frame, f"Capturando: {pose_instruction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        cv2.putText(frame, f"Imagem {count}/{images_per_pose}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Captura de Faces", frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Captura interrompida pelo usuário.")
            exit(0)

    all_images = os.listdir(temp_pose_dir)
    random.shuffle(all_images)

    n_test = int(len(all_images) * test_split_ratio)
    test_images = all_images[:n_test]
    train_images = all_images[n_test:]

    train_pose_dir = os.path.join(base_train_dir, f"pose_{pose_index}")
    test_pose_dir = os.path.join(base_test_dir, f"pose_{pose_index}")
    os.makedirs(train_pose_dir, exist_ok=True)
    os.makedirs(test_pose_dir, exist_ok=True)

    for img_name in train_images:
        src = os.path.join(temp_pose_dir, img_name)
        dst = os.path.join(train_pose_dir, img_name)
        shutil.move(src, dst)

    for img_name in test_images:
        src = os.path.join(temp_pose_dir, img_name)
        dst = os.path.join(test_pose_dir, img_name)
        shutil.move(src, dst)

    os.rmdir(temp_pose_dir)

cap.release()
cv2.destroyAllWindows()
print(f"Captura finalizada para {name}. Imagens salvas em 'data/train' e 'data/test'.")
