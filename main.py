import pika
import json
import cv2
import numpy as np
import base64
import re
import psycopg2
import io
import torch
import collections
import torchvision.models as models
import torchvision.transforms as transforms
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
from ultralytics import YOLO

# ---- Khởi tạo mô hình AI ----
print("[INFO] Đang khởi tạo hệ thống và tải các mô hình AI...")

vehicle_detector = YOLO('result_train(pt)/vehicle_model.pt')   # Phát hiện phương tiện
lp_detector = YOLO('result_train(pt)/lp_model.pt')             # Phát hiện biển số
ocr_detector = YOLO('result_train(pt)/ocr_model.pt')           # Nhận diện ký tự

# Model trích xuất vector đặc trưng (dùng để chống trùng lặp)
resnet = models.resnet18(pretrained=True)
resnet.eval() 
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Kết nối các dịch vụ ----
pg_conn = psycopg2.connect(dsn="dbname=lpr_db user=admin password=admin123 host=localhost")
minio_client = Minio("localhost:9000", access_key="admin", secret_key="admin123", secure=False)
qdrant_client = QdrantClient("localhost", port=6333)


def extract_vector(image_crop):
    """Trích xuất vector đặc trưng từ ảnh bằng ResNet18"""
    input_tensor = preprocess(image_crop).unsqueeze(0)
    with torch.no_grad():
        output = resnet(input_tensor)
    return output.numpy()[0] 

def normalize_plate(plate_text):
    """Chuẩn hóa biển số: bỏ dấu gạch, chấm, khoảng trắng"""
    return re.sub(r'[^A-Za-z0-9]', '', plate_text)


def sort_characters(boxes, classes, names_dict):
    """Sắp xếp ký tự OCR theo vị trí (xử lý biển 1 dòng và 2 dòng)"""
    if len(boxes) == 0:
        return ""
    chars = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        char_name = names_dict[int(cls)]
        chars.append({'char': char_name, 'cx': cx, 'cy': cy, 'h': y2 - y1})

    chars.sort(key=lambda x: x['cy'])
    avg_h = sum(c['h'] for c in chars) / len(chars)

    # Phân tách dòng 1 và dòng 2 (biển vuông 2 dòng)
    line_1, line_2 = [], []
    min_cy = chars[0]['cy']
    for c in chars:
        if c['cy'] - min_cy > avg_h * 0.5:
            line_2.append(c)
        else:
            line_1.append(c)

    line_1.sort(key=lambda x: x['cx'])
    line_2.sort(key=lambda x: x['cx'])

    result = "".join([c['char'] for c in line_1])
    if len(line_2) > 0:
        result += "-" + "".join([c['char'] for c in line_2])
    return result


def save_to_minio(img_np, bucket, filename):
    """Lưu ảnh numpy vào MinIO, trả về URL public"""
    success, encoded_img = cv2.imencode('.jpg', img_np)
    if success:
        data = io.BytesIO(encoded_img.tobytes())
        minio_client.put_object(bucket, filename, data, len(encoded_img.tobytes()), content_type="image/jpeg")
        return f"http://localhost:9000/{bucket}/{filename}"
    return None

def check_violation_and_log(plate_text, img_url, lp_url):
    """Tra cứu phạt nguội trong DB + ghi log lịch sử camera"""
    cursor = pg_conn.cursor()
    try:
        # Kiểm tra xe có trong danh sách phạt nguội không
        query_check = """
            SELECT loi_vi_pham, thoi_gian_vi_pham 
            FROM phat_nguoi 
            WHERE bien_so = %s 
            ORDER BY thoi_gian_vi_pham DESC LIMIT 1
        """
        cursor.execute(query_check, (plate_text,))
        result = cursor.fetchone()
        
        if result:
            has_violation = True
            violation_detail = result[0]
            violation_time = result[1]
        else:
            has_violation = False
            violation_detail = "Không"
            violation_time = None

        # Ghi nhật ký vào bảng lich_su_camera
        query_insert = """
            INSERT INTO lich_su_camera (bien_so, link_anh_goc, link_anh_bien_so, co_vi_pham) 
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query_insert, (plate_text, img_url, lp_url, has_violation))
        
        pg_conn.commit()
        return has_violation, violation_detail, violation_time

    except Exception as e:
        print(f"[ERROR] Lỗi CSDL: {e}")
        pg_conn.rollback()
        return False, "Lỗi", None
    finally:
        cursor.close()

# ---- Biến trạng thái theo dõi xe ----
plate_buffer = {}             # Các lần đọc biển số theo track_id
vehicle_images = {}           # Ảnh xe mới nhất
plate_images = {}             # Ảnh biển số mới nhất
active_ids_last_frame = set() # Xe xuất hiện ở frame trước
logged_ids = set()            # Xe đã lưu DB xong


def callback(ch, method, properties, body):
    """Hàm callback xử lý mỗi frame nhận từ RabbitMQ"""
    global active_ids_last_frame, logged_ids

    # Giải mã ảnh từ RabbitMQ message
    data = json.loads(body)
    camera_id = data.get('camera_id', 'Unknown')

    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    active_ids_this_frame = set()

    # Bước 1: Phát hiện phương tiện + Tracking
    track_results = vehicle_detector.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.2, verbose=False)
    
    if track_results[0].boxes.id is not None:
        boxes = track_results[0].boxes.xyxy.cpu().numpy()
        track_ids = track_results[0].boxes.id.int().cpu().tolist()
        class_ids = track_results[0].boxes.cls.int().cpu().tolist()

        active_ids_this_frame = set(track_ids)

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            vx1, vy1, vx2, vy2 = map(int, box)
            
            vehicle_img = frame[vy1:vy2, vx1:vx2]
            if vehicle_img.size == 0: continue

            # Vẽ khung phương tiện
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(frame, f"Class: {cls_id} | ID: {track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            vehicle_images[track_id] = frame.copy()
            
            # Bước 2: Phát hiện biển số trong vùng xe
            lp_results = lp_detector.predict(vehicle_img, conf=0.5, verbose=False)

            for result in lp_results:
                for lp_box in result.boxes:
                    px1, py1, px2, py2 = map(int, lp_box.xyxy[0])
                    
                    plate_img = vehicle_img[py1:py2, px1:px2]
                    if plate_img.size == 0: continue

                    # Bước 3: OCR đọc ký tự
                    ocr_results = ocr_detector.predict(plate_img, conf=0.4, verbose=False)

                    for ocr_res in ocr_results:
                        char_boxes = ocr_res.boxes.xyxy.cpu().numpy()
                        char_classes = ocr_res.boxes.cls.cpu().numpy()

                        plate_text = sort_characters(char_boxes, char_classes, ocr_detector.names)
                        plate_images[track_id] = plate_img

                        # Chuyển tọa độ biển số về frame gốc để vẽ
                        abs_px1, abs_py1 = vx1 + px1, vy1 + py1
                        abs_px2, abs_py2 = vx1 + px2, vy1 + py2

                        cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{plate_text}", (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Gom kết quả OCR vào buffer (voting sau)
                        if track_id not in logged_ids:
                            if track_id not in plate_buffer:
                                plate_buffer[track_id] = []
                            if plate_text.strip() != "":
                                plate_buffer[track_id].append(normalize_plate(plate_text))

    # Khi xe rời khỏi camera → chốt kết quả
    left_ids = active_ids_last_frame - active_ids_this_frame

    for l_id in left_ids:
        if l_id in plate_buffer and len(plate_buffer[l_id]) > 0:
            
            # Lấy biển số xuất hiện nhiều nhất (majority voting)
            counter = collections.Counter(plate_buffer[l_id])
            best_plate = counter.most_common(1)[0][0] 
            
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"{best_plate}_{timestamp}.jpg"
            
            # Lưu ảnh lên MinIO
            full_url = save_to_minio(vehicle_images.get(l_id, frame), "full-frames", filename)
            lp_url = save_to_minio(plate_images.get(l_id, frame), "plates", f"lp_{filename}")
            
            # Lưu vector vào Qdrant (chống trùng lặp)
            try:
                vector = extract_vector(vehicle_images.get(l_id, frame))
                qdrant_client.upsert(
                    collection_name="plates",
                    points=[PointStruct(id=l_id, vector=vector.tolist(), payload={"plate": best_plate})]
                )
            except Exception as e:
                print(f"[ERROR] Qdrant Error: {e}")
            
            # Tra cứu phạt nguội + ghi log
            is_violated, error_msg, past_time = check_violation_and_log(best_plate, full_url, lp_url)

            if is_violated:
                time_str = past_time.strftime("%H:%M %d/%m/%Y")
                print(f"--- PHÁT HIỆN XE TRONG DANH SÁCH PHẠT NGUỘI ---")
                print(f"Biển số: {best_plate}")
                print(f"Lỗi vi phạm: {error_msg}")
                print(f"Thời gian vi phạm trước đó: {time_str}")
                print(f"----------------------------------------------")
            else:
                print(f"[OK] Xe {best_plate} không có dữ liệu vi phạm.")

            # Dọn dẹp RAM
            logged_ids.add(l_id)
            if l_id in plate_buffer: del plate_buffer[l_id]
            if l_id in vehicle_images: del vehicle_images[l_id]
            if l_id in plate_images: del plate_images[l_id]

    # Hiển thị video nhận diện
    cv2.imshow("He Thong AI Nhan Dien Bien So", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    active_ids_last_frame = active_ids_this_frame
    ch.basic_ack(delivery_tag=method.delivery_tag)


# ---- Kết nối RabbitMQ và bắt đầu lắng nghe ----
credentials = pika.PlainCredentials('admin', 'admin123')

rabbitmq_conn = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials)
)
channel = rabbitmq_conn.channel()
channel.queue_declare(queue='camera_frames', durable=True)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='camera_frames', on_message_callback=callback)

print("[INFO] Worker AI đang lắng nghe dữ liệu từ RabbitMQ...")
channel.start_consuming()