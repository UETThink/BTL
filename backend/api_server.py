from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import pika
import json
import psycopg2
import re
from psycopg2.extras import RealDictCursor
import time

app = FastAPI(
    title="Hệ thống ALPR API",
    description="Backend API kết nối Frontend, RabbitMQ và PostgreSQL"
)

# Cho phép Frontend gọi API từ domain khác (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình kết nối DB
DB_CONFIG = {
    "dbname": "lpr_db",
    "user": "admin",
    "password": "admin123",
    "host": "localhost",
    "port": "5432"
}

RABBITMQ_HOST = 'localhost'

# Model nhận dữ liệu ảnh từ Camera
class FramePayload(BaseModel):
    camera_id: str
    image_base64: str


def normalize_plate(plate_text: str) -> str:
    """Loại bỏ dấu gạch, chấm, khoảng trắng trong biển số"""
    return re.sub(r'[^A-Za-z0-9]', '', plate_text)

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"[Lỗi Database] Không thể kết nối: {e}")
        raise HTTPException(status_code=500, detail="Lỗi kết nối cơ sở dữ liệu")

def publish_to_rabbitmq(queue_name: str, message: dict):
    """Đẩy message vào queue RabbitMQ"""
    try:
        credentials = pika.PlainCredentials('admin', 'admin123')
        parameters = pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
        
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # persistent message
            )
        )
        connection.close()
    except Exception as e:
        print(f"[Lỗi RabbitMQ] Không thể gửi tin nhắn: {e}")
        raise HTTPException(status_code=500, detail="Lỗi kết nối Message Queue")


# ---- API ENDPOINTS ----

@app.post("/api/upload-frame")
async def upload_frame(payload: FramePayload):
    """Nhận ảnh Base64 từ Camera và đẩy vào RabbitMQ cho Worker xử lý"""
    message = {
        "camera_id": payload.camera_id,
        "image": payload.image_base64,
        "timestamp": time.time()
    }
    publish_to_rabbitmq(queue_name='camera_frames', message=message)
    return {"status": "success", "message": "Đã đưa frame vào hàng đợi xử lý"}

@app.get("/api/logs")
async def get_camera_logs(limit: int = 50):
    """Lấy danh sách lịch sử nhận diện"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = "SELECT * FROM lich_su_camera ORDER BY id ASC LIMIT %s"
        cursor.execute(query, (limit,))
        logs = cursor.fetchall()
        return {"status": "success", "data": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/violations")
async def get_violations():
    """Lấy danh sách xe vi phạm phạt nguội"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = "SELECT * FROM phat_nguoi ORDER BY id ASC"
        cursor.execute(query)
        violations = cursor.fetchall()
        return {"status": "success", "data": violations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/search")
async def search_by_plate(plate: str):
    """Tìm kiếm lịch sử theo biển số xe"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        clean_plate = normalize_plate(plate)
        query = "SELECT * FROM lich_su_camera WHERE bien_so ILIKE %s ORDER BY id ASC"
        cursor.execute(query, (f"%{clean_plate}%",))
        results = cursor.fetchall()
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/stats")
async def get_dashboard_stats():
    """Thống kê tổng quan cho Dashboard"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("SELECT COUNT(*) as total FROM lich_su_camera")
        total_vehicles = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(*) as total FROM lich_su_camera WHERE co_vi_pham = TRUE")
        total_violations_detected = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(*) as total FROM phat_nguoi")
        total_violations_db = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(DISTINCT bien_so) as total FROM lich_su_camera")
        unique_plates = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM lich_su_camera WHERE DATE(thoi_gian_chup) = CURRENT_DATE")
        today_count = cursor.fetchone()["total"]

        return {
            "status": "success",
            "data": {
                "total_vehicles": total_vehicles,
                "total_violations_detected": total_violations_detected,
                "total_violations_db": total_violations_db,
                "unique_plates": unique_plates,
                "today_count": today_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# Serve Frontend qua cùng port 8000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"), media_type="text/html")

app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
