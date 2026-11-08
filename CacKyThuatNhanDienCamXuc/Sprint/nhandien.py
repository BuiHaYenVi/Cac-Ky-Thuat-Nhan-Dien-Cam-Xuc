import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('emotion_cnn_model.h5')

# Danh sách cảm xúc
emotion_labels = ['Gian du', 'ghe tom', 'So hai', 'Vui ve', 'Binh thuong', 'Buon', 'Ngac nhien']

# Khởi tạo camera và bộ phát hiện khuôn mặt
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Emotion Detection (CNN)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
