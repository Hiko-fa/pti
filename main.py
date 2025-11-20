import cv2
import time
from datetime import datetime
import winsound
import numpy as np
from collections import deque

AGE_BUFFER = deque(maxlen=15)

def smooth_age(age_raw):
    age_ranges = {
        '(0-2)': (0, 2),
        '(4-6)': (4, 6),
        '(8-12)': (8, 12),
        '(15-20)': (15, 20),
        '(25-32)': (25, 32),
        '(38-43)': (38, 43),
        '(48-53)': (48, 53),
        '(60-100)': (60, 100)
    }

    low, high = age_ranges[age_raw]
    mid = (low + high) / 2
    AGE_BUFFER.append(mid)

    avg = np.mean(AGE_BUFFER)

    if avg <= 18:
        avg *= 0.90
    elif avg >= 25:
        avg *= 1.08

    for label, (a, b) in age_ranges.items():
        if a <= avg <= b:
            return label

    return age_raw

AGE_LOCKED = None
AGE_STABLE_COUNT = 0
AGE_STABLE_THRESHOLD = 10
LAST_AGE_PRED = None

MOOD_LOCKED = None

def drawText(img, text, pos, font_scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bbox = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])

    return frame, bbox

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Laki-laki', 'Perempuan']
moodList = ['Bahagia', 'Santai', 'Serius', 'Lelah']

video = cv2.VideoCapture(0)
padding = 20

prev_time = time.time()
total_faces_detected = 0
last_face_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    frame, bboxs = faceBox(faceNet, frame)

    if len(bboxs) == 0:
        AGE_LOCKED = None
        AGE_STABLE_COUNT = 0
        LAST_AGE_PRED = None
        MOOD_LOCKED = None

    if len(bboxs) != last_face_count:
        winsound.Beep(1000, 200)
        last_face_count = len(bboxs)

    total_faces_detected += len(bboxs)

    for bbox in bboxs:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age_raw = ageList[ageNet.forward()[0].argmax()]

        if AGE_LOCKED is not None:
            age = AGE_LOCKED
        else:
            age_smoothed = smooth_age(age_raw)

            if age_smoothed == LAST_AGE_PRED:
                AGE_STABLE_COUNT += 1
            else:
                AGE_STABLE_COUNT = 1
                LAST_AGE_PRED = age_smoothed

            if AGE_STABLE_COUNT >= AGE_STABLE_THRESHOLD:
                AGE_LOCKED = age_smoothed

            age = age_smoothed

        if MOOD_LOCKED is None:
            MOOD_LOCKED = moodList[int(time.time()) % len(moodList)]
        mood = MOOD_LOCKED

        drawText(frame, f"{gender}, {age}", (bbox[0], bbox[1]-10), 0.8)

        drawText(frame, f"{mood}", (bbox[0], bbox[1]+20), 0.8, (0, 255, 255))

        color = (255, 100, 0) if gender == 'Laki-laki' else (255, 0, 200)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        if (bbox[2] - bbox[0]) > 250:
            drawText(frame, "Terlalu Dekat!", (bbox[0], bbox[3]+30), 0.9, (0, 0, 255))

    drawText(frame, f"Faces: {len(bboxs)} | Total: {total_faces_detected}", (10, 30), 0.9, (0,255,255))
    drawText(frame, f"FPS: {int(fps)}", (10, 60), 0.9, (0,255,0))
    drawText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 90), 0.8)

    cv2.imshow("Real-Time Age, Gender, Mood Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Gambar disimpan sebagai {filename}")

video.release()
cv2.destroyAllWindows()
