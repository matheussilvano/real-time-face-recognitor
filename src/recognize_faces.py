import cv2
import pickle
import time

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.pickle"
THRESH = 60
CAM_INDEX = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

with open(LABELS_FILE, "rb") as f:
    label_names = pickle.load(f)

cam = cv2.VideoCapture(CAM_INDEX)
if not cam.isOpened():
    print(f"Não foi possível abrir a webcam local no índice {CAM_INDEX}.")
    exit(1)

fps_tic = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Frame vazio.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

        label_id, conf = recognizer.predict(roi)

        if conf < THRESH:
            name = label_names.get(label_id, "Desconhecido")
            text = f"{name} ({conf:.0f})"
            color = (0, 255, 0)
        else:
            text = "Desconhecido"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    fps = 1 / (time.time() - fps_tic)
    fps_tic = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Reconhecimento - Q ou ESC para sair", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cam.release()
cv2.destroyAllWindows()
