import cv2
import time
from datetime import datetime
import winsound
import numpy as np
from collections import deque, defaultdict
import math

AGE_BUFFERS = defaultdict(lambda: deque(maxlen=40))
EMOTION_BUFFERS = defaultdict(lambda: deque(maxlen=20))
GENDER_BUFFERS = defaultdict(lambda: deque(maxlen=40))

PREV_DETECTIONS = {} 
NEXT_FACE_ID = 0
BBOX_MEMORY = deque(maxlen=6) 
IOU_MATCH_THRESH = 0.4

FACE_COUNT_HISTORY = deque(maxlen=6)
SOUND_STABLE_FRAMES = 4

MIN_FACE_AREA = 1600
ASPECT_RATIO_MIN = 0.55
ASPECT_RATIO_MAX = 1.6
CLARITY_MIN = 20.0 

def drawText(img, text, pos, font_scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0]+1)) * max(0, (boxA[3]-boxA[1]+1))
    boxBArea = max(0, (boxB[2]-boxB[0]+1)) * max(0, (boxB[3]-boxB[1]+1))
    union = boxAArea + boxBArea - inter
    return inter / union if union > 0 else 0

def match_and_assign_ids(detected_boxes):
    global NEXT_FACE_ID, PREV_DETECTIONS
    assigned = {}
    used_prev = set()

    for box in detected_boxes:
        best_id = None
        best_iou = 0.0
        for pid, pbox in PREV_DETECTIONS.items():
            if pid in used_prev:
                continue
            val = iou(box, pbox)
            if val > best_iou:
                best_iou = val
                best_id = pid
        if best_iou >= IOU_MATCH_THRESH and best_id is not None:
            assigned[best_id] = box
            used_prev.add(best_id)
        else:
            assigned[f"face_{NEXT_FACE_ID}"] = box
            NEXT_FACE_ID += 1

    PREV_DETECTIONS = assigned.copy()
    return list(assigned.items())

def bbox_exponential_smooth(new_box, history_boxes, alpha=0.6):
    if not history_boxes:
        return new_box
    avg = np.array(new_box).astype(float)
    for b in history_boxes:
        avg = alpha * np.array(b) + (1 - alpha) * avg
    return avg.astype(int).tolist()

def detect_hijab(face):
    h, w = face.shape[:2]
    top_slice = face[0:int(h*0.35), int(w*0.05):int(w*0.95)]
    if top_slice.size == 0:
        return False, 0.0

    img_ycrcb = cv2.cvtColor(top_slice, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(img_ycrcb, np.array([0,135,85]), np.array([255,180,135]))
    skin_pixels = cv2.countNonZero(skin_mask)
    total = top_slice.shape[0] * top_slice.shape[1]
    skin_ratio = skin_pixels / total if total > 0 else 0.0

    v = np.var(cv2.cvtColor(top_slice, cv2.COLOR_BGR2GRAY))

    if skin_ratio < 0.12 and v < 800.0:
        return True, skin_ratio
    return False, skin_ratio

def smooth_age(age_raw, face_id):
    age_ranges = {
        '(0-2)': (1, 2),
        '(4-6)': (4, 6),
        '(8-12)': (9, 12),
        '(15-20)': (16, 20),
        '(25-32)': (26, 32),
        '(38-43)': (38, 43),
        '(48-53)': (48, 53),
        '(60-100)': (65, 80)
    }

    low, high = age_ranges.get(age_raw, (20,30))
    mid = (low + high) / 2

    buf = AGE_BUFFERS[face_id]
    buf.append(mid)

    median = np.median(buf)
    mean = np.mean(buf)

    final_age = (median * 0.65) + (mean * 0.35)

    if final_age < 18:
        final_age *= 0.90
    elif final_age > 40:
        final_age *= 1.08

    for label, (a, b) in age_ranges.items():
        if a <= final_age <= b:
            return label

    return age_raw

def analyze_mood(face, face_id):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

    mouth = gray[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.8)]
    if mouth.size == 0:
        mouth_edges = np.array([])
    else:
        mouth_edges = cv2.Canny(mouth, 50, 150)
    smile = np.sum(mouth_edges)

    eyes = gray[int(h*0.12):int(h*0.4), int(w*0.2):int(w*0.8)]
    eye_brightness = np.mean(eyes) if eyes.size else 100

    if smile > 38000 and clarity > 90:
        emotion = "Bahagia"
    elif clarity < 55 and eye_brightness < 85:
        emotion = "Lelah"
    elif smile < 12000 and clarity > 75:
        emotion = "Serius"
    else:
        emotion = "Santai"

    buf = EMOTION_BUFFERS[face_id]
    buf.append(emotion)
    return max(set(buf), key=buf.count)

def smooth_gender(gender_preds, face_id, clarity, face_crop):
    if gender_preds is None or len(gender_preds) < 2:
        if GENDER_BUFFERS[face_id]:
            return max(set(GENDER_BUFFERS[face_id]), key=GENDER_BUFFERS[face_id].count)
        return "Tidak Diketahui"

    prob_male = float(gender_preds[0])
    prob_female = float(gender_preds[1])
    confidence = abs(prob_male - prob_female)

    predicted = "Laki-laki" if prob_male > prob_female else "Perempuan"

    if clarity < 40:
        if GENDER_BUFFERS[face_id]:
            return max(set(GENDER_BUFFERS[face_id]), key=GENDER_BUFFERS[face_id].count)
        return "Tidak Diketahui"

    is_hijab, skin_ratio = detect_hijab(face_crop)

    if is_hijab:
        hist = list(GENDER_BUFFERS[face_id])
        if hist:
            counts = {k:hist.count(k) for k in set(hist)}
            top = max(counts, key=counts.get)
            if counts[top] >= max(3, len(hist)//2):
                return top
        if prob_female >= 0.35:
            GENDER_BUFFERS[face_id].append("Perempuan")
            return "Perempuan"
        if confidence < 0.25:
            if hist:
                return max(set(hist), key=hist.count)
            return "Perempuan"

    if confidence < 0.12:
        if GENDER_BUFFERS[face_id]:
            return max(set(GENDER_BUFFERS[face_id]), key=GENDER_BUFFERS[face_id].count)
        else:
            GENDER_BUFFERS[face_id].append(predicted)
            return predicted

    GENDER_BUFFERS[face_id].append(predicted)
    return max(set(GENDER_BUFFERS[face_id]), key=GENDER_BUFFERS[face_id].count)

def faceBox(faceNet, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxs = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0,0,i,2])
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(frameWidth-1,x2), min(frameHeight-1,y2)

            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect = (w / float(h)) if h>0 else 0

            if area < MIN_FACE_AREA:
                continue
            if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX:
                continue

            bboxs.append([x1,y1,x2,y2])
    return frame, bboxs

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
ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Laki-laki','Perempuan']

video = cv2.VideoCapture(0)
padding = 20

prev_time = time.time()
last_stable_count = 0
total_faces = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    curr = time.time()
    fps = 1 / (curr - prev_time) if (curr - prev_time) > 0 else 0
    prev_time = curr

    frame, raw_boxes = faceBox(faceNet, frame, conf_threshold=0.65)

    if raw_boxes:
        BBOX_MEMORY.append(raw_boxes)
        try:
            if len(BBOX_MEMORY) >= 2 and all(len(x)==len(raw_boxes) for x in BBOX_MEMORY):
                avg_boxes = []
                for idx in range(len(raw_boxes)):
                    hist = [b[idx] for b in BBOX_MEMORY]
                    avg = np.mean(np.array(hist), axis=0).astype(int).tolist()
                    avg_boxes.append(avg)
                smoothed_boxes = avg_boxes
            else:
                smoothed_boxes = []
                for i, b in enumerate(raw_boxes):
                    hist = []
                    for h in BBOX_MEMORY:
                        if i < len(h):
                            hist.append(h[i])
                    smoothed = bbox_exponential_smooth(b, hist, alpha=0.6) if hist else b
                    smoothed_boxes.append(smoothed)
        except Exception:
            smoothed_boxes = raw_boxes
    else:
        BBOX_MEMORY.clear()
        smoothed_boxes = []

    matched = match_and_assign_ids(smoothed_boxes)

    FACE_COUNT_HISTORY.append(len(matched))
    if len(FACE_COUNT_HISTORY) >= SOUND_STABLE_FRAMES and len(set(list(FACE_COUNT_HISTORY)[-SOUND_STABLE_FRAMES:]))==1:
        stable_count = FACE_COUNT_HISTORY[-1]
        if stable_count != last_stable_count:
            try:
                if stable_count > last_stable_count:
                    winsound.Beep(1200, 110)
                else:
                    winsound.Beep(700, 110)
            except:
                pass
            last_stable_count = stable_count

    for face_id, bbox in matched:
        x1,y1,x2,y2 = map(int, bbox)
        face = frame[max(0,y1-padding):min(y2+padding,frame.shape[0]), max(0,x1-padding):min(x2+padding,frame.shape[1])]
        if face.size == 0:
            continue

        try:
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            clarity = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        except Exception:
            clarity = 0.0

        if clarity < CLARITY_MIN:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gp = genderNet.forward()
        gender_preds = gp[0] if hasattr(gp, 'shape') and gp.shape and len(gp.shape)>1 else gp
        gender = smooth_gender(gender_preds, face_id, clarity, face)

        ageNet.setInput(blob)
        age_out = ageNet.forward()
        age_raw_idx = int(np.argmax(age_out[0]))
        age_raw = ageList[age_raw_idx] if 0 <= age_raw_idx < len(ageList) else '(25-32)'
        age = smooth_age(age_raw, face_id)

        mood = analyze_mood(face, face_id)

        drawText(frame, f"{gender}, {age}", (x1, y1-10))
        drawText(frame, mood, (x1, y1+22), 0.8, (0,255,255))

        color = (255,100,0) if gender=='Laki-laki' else (255,0,200) if gender=='Perempuan' else (200,200,200)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

        if (x2-x1) > 250:
            drawText(frame, "Terlalu Dekat!", (x1, y2+30), 0.9, (0,0,255))

        total_faces += 1 

    drawText(frame, f"Faces (tracked): {len(matched)}", (10,30), 0.9,(0,255,255))
    drawText(frame, f"FPS: {int(fps)}", (10,60), 0.9,(0,255,0))
    drawText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10,90))

    cv2.imshow("Age, Gender & Emotion Detection (Hijab-aware)", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        name = f"deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(name, frame)
        print(f"Disimpan: {name}")

video.release()
cv2.destroyAllWindows()
