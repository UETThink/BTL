import cv2
import pika
import base64
import json
import time
import sys
import requests

# Cách dùng:
#   python camera_producer.py                  -> Mở Webcam
#   python camera_producer.py video.mp4        -> Đọc từ file video

source = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"[LỖI] Không thể mở nguồn video: {source}")
    sys.exit(1)

API_URL = "http://localhost:8000/api/upload-frame"
TARGET_FPS = 10
frame_delay = 1.0 / TARGET_FPS

print(f"[INFO] Nguồn video: {'Webcam' if source == 0 else source}")
print(f"[INFO] Bắt đầu gửi frame qua API...")

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        if source != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        break

    # Nén ảnh jpg và chuyển base64
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "camera_id": "BTL-THPTTTNT-36",
        "image_base64": jpg_as_text
    }

    try:
        requests.post(API_URL, json=payload, timeout=5)
    except Exception as e:
        print("Lỗi kết nối API:", e)

    cv2.imshow("Camera Producer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Giới hạn FPS khi đọc từ file video
    if source != 0:
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

cap.release()
cv2.destroyAllWindows()
