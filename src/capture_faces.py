import cv2
import os

name = input("Nome da pessoa: ").strip()
if not name:
    print("Nome vazio. Saindo.")
    exit(1)

save_dir = os.path.join("data", name)
os.makedirs(save_dir, exist_ok=True)

cam_index = 0  

cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("Não foi possível abrir a webcam.")
    exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
MAX_IMAGES = 50

print(f"Iniciando captura de faces para '{name}'. Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame vazio.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        filepath = os.path.join(save_dir, f"{count:03}.png")
        cv2.imwrite(filepath, face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        count += 1
        if count >= MAX_IMAGES:
            break

    cv2.imshow(f"Capturando faces - {name} ({count}/{MAX_IMAGES})", frame)

    if cv2.waitKey(1) == 27 or count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captura finalizada. {count} imagens salvas em '{save_dir}'.")
