import cv2
from deepface import DeepFace
faceCascade = cv2
cv2.CascadeClassifier(cv2.data.haarcascades +
                      'haarcascades_frontal_face_default.xml')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions=['emotion'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    font = cv2.FONT_HERSHY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'],
                (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('Demo video', frame)
    if cv2.watKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
