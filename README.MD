# 🚗 Hệ Thống Nhận Diện Biển Số Xe Tự Động (ALPR)

> **Automatic License Plate Recognition** — Hệ thống AI sử dụng kiến trúc 2 giai đoạn dựa trên YOLOv10, tích hợp hạ tầng phân tán với PostgreSQL, MinIO, Qdrant và RabbitMQ.

---

## 📋 Mục Lục

- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt & Chạy Dự Án](#-hướng-dẫn-cài-đặt--chạy-dự-án)
  - [Bước 1: Clone dự án](#bước-1-clone-dự-án)
  - [Bước 2: Cài đặt thư viện Python](#bước-2-cài-đặt-thư-viện-python)
  - [Bước 3: Khởi động hạ tầng Docker](#bước-3-khởi-động-hạ-tầng-docker)
  - [Bước 4: Cấu hình MinIO Storage](#bước-4-cấu-hình-minio-storage)
  - [Bước 5: Khởi động Backend API](#bước-5-khởi-động-backend-api)
  - [Bước 6: Chạy Pipeline AI nhận diện](#bước-6-chạy-pipeline-ai-nhận-diện)
  - [Bước 7: Mở Dashboard trên trình duyệt](#bước-7-mở-dashboard-trên-trình-duyệt)
- [Các Cổng Dịch Vụ](#-các-cổng-dịch-vụ)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)

---

## 🏗 Kiến Trúc Hệ Thống

```
Dashboard (Trình duyệt)
    │
    ├── Bật Webcam / Tải Video
    │        │
    │        ▼
    │   API Server (FastAPI :8000)
    │        │
    │        ▼
    │   RabbitMQ (Message Queue)
    │        │
    │        ▼
    │   main.py — AI Worker (Lắng nghe & xử lý frame)
    │        │
    │        ├── YOLOv10 #1: Phát hiện phương tiện (vehicle_model.pt)
    │        ├── YOLOv10 #2: Phát hiện biển số     (lp_model.pt)
    │        ├── YOLOv10 #3: Nhận diện ký tự OCR   (ocr_model.pt)
    │        │
    │        ├──▶ MinIO        (Lưu ảnh gốc + ảnh biển số)
    │        ├──▶ Qdrant       (Vector DB chống trùng lặp)
    │        └──▶ PostgreSQL   (Lưu lịch sử + danh sách phạt nguội)
    │                 │
    └── Dashboard ◀──┘  (Hiển thị kết quả lên bảng Lịch Sử / Phạt Nguội)
```

**Pipeline xử lý 3 giai đoạn:**

| Giai đoạn | Mô hình | Chức năng |
|-----------|---------|-----------|
| Vehicle Detection | `vehicle_model.pt` | Phát hiện & phân loại phương tiện (Xe máy, Ô tô, Xe tải) |
| LP Detection | `lp_model.pt` | Định vị vùng biển số trong khung hình |
| OCR | `ocr_model.pt` | Nhận diện từng ký tự trên biển số |

---

## 📁 Cấu Trúc Thư Mục

```
BTL-THPTTTNT-36/
│
├── main.py                  # 🧠 Pipeline AI chính (phát hiện + OCR + lưu DB)
├── camera_producer.py       # 📹 Gửi frame từ webcam/video qua API
├── setup_minio.py           # ⚙️ Script tạo bucket trên MinIO
├── check_db.py              # 🔍 Script kiểm tra dữ liệu trong PostgreSQL
├── requirements.txt         # 📦 Danh sách thư viện Python
├── data.yaml                # 📊 Cấu hình dataset cho training
├── testYoloV10.py           # 🧪 Script test mô hình cơ bản
│
├── result_train(pt)/        # 🎯 Trọng số mô hình đã huấn luyện
│   ├── vehicle_model.pt     #    Mô hình phát hiện phương tiện
│   ├── lp_model.pt          #    Mô hình phát hiện biển số
│   └── ocr_model.pt         #    Mô hình nhận diện ký tự
│
├── backend/                 # ⚡ Backend API Server
│   └── api_server.py        #    FastAPI: REST API + Serve Frontend
│
├── frontend/                # 🌐 Giao diện Dashboard
│   ├── index.html           #    Trang chính
│   ├── app.js               #    Logic hiển thị + phân trang
│   ├── style.css            #    Giao diện (Glassmorphism theme)
│   └── particles.js         #    Hiệu ứng hạt nền
│
├── lpr-system/              # 🐳 Hạ tầng Docker
│   ├── docker-compose.yml   #    Cấu hình 4 dịch vụ Docker
│   └── database/
│       └── init.sql          #    Schema + dữ liệu mẫu PostgreSQL
│
├── datasets/                # 📂 Dữ liệu huấn luyện (ảnh + nhãn)
├── training/                # 📝 Script huấn luyện mô hình
└── runs/                    # 📈 Kết quả & log huấn luyện
```

---

## ⚙ Yêu Cầu Hệ Thống

| Yêu cầu | Phiên bản |
|----------|-----------|
| **Python** | >= 3.9 |
| **Docker Desktop** | >= 4.0 (cần chạy Docker Engine) |
| **Git** | Bất kỳ |
| **RAM** | >= 8 GB (khuyến nghị 16 GB) |
| **GPU** | Không bắt buộc (có CUDA sẽ nhanh hơn) |

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### Bước 1: Clone dự án

```bash
git clone <repository-url>
cd BTL-THPTTTNT-36
```

---

### Bước 2: Cài đặt thư viện Python

Tạo môi trường ảo và cài đặt dependencies:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Hoặc kích hoạt trên CMD
.venv\Scripts\activate

# Cài đặt toàn bộ thư viện
pip install -r requirements.txt
```

> ⏳ Quá trình cài đặt mất khoảng 2-5 phút (tải PyTorch, Ultralytics, OpenCV...).

---

### Bước 3: Khởi động hạ tầng Docker

Mở **Docker Desktop** trước, đảm bảo Docker Engine đang chạy, sau đó:

```bash
cd lpr-system
docker-compose up -d
```

Lệnh này sẽ tự động tải image và khởi động **4 dịch vụ**:

| Dịch vụ | Container | Chức năng |
|---------|-----------|-----------|
| **PostgreSQL** | `lpr_postgres` | Cơ sở dữ liệu SQL (lịch sử + phạt nguội) |
| **MinIO** | `lpr_minio` | Object Storage lưu trữ ảnh |
| **Qdrant** | `lpr_qdrant` | Vector Database chống trùng lặp |
| **RabbitMQ** | `lpr_rabbitmq` | Message Queue điều phối luồng ảnh |

Kiểm tra trạng thái các container:

```bash
docker ps
```

> ✅ Tất cả 4 container phải ở trạng thái `Up`.

Quay lại thư mục gốc:

```bash
cd ..
```

---

### Bước 4: Cấu hình MinIO Storage

Chạy script tạo bucket lưu ảnh trên MinIO:

```bash
python setup_minio.py
```

> Script sẽ tạo 2 bucket: `full-frames` (ảnh gốc) và `plates` (ảnh biển số) với quyền đọc public.

---

### Bước 5: Khởi động Backend API

Mở **Terminal 1** và chạy:

```bash
python -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

Kết quả mong đợi:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> 💡 **Giữ terminal này mở.** Backend API phục vụ Dashboard web và nhận frame ảnh từ Camera.

---

### Bước 6: Khởi động AI Worker

Mở **Terminal 2** (giữ Terminal 1 đang chạy), kích hoạt lại môi trường ảo và chạy:

```bash
# Kích hoạt môi trường ảo (nếu chưa)
.\.venv\Scripts\Activate.ps1

# Chạy AI Worker
python main.py
```

Kết quả mong đợi:

```
[INFO] Đang khởi tạo hệ thống và tải các mô hình AI...
[INFO] Worker AI đang lắng nghe dữ liệu từ RabbitMQ...
```

> 🧠 `main.py` là **AI Worker** — nó kết nối vào RabbitMQ và **lắng nghe** frame ảnh được gửi từ Dashboard web. Worker sẽ tự động xử lý khi có dữ liệu: Phát hiện xe → Cắt biển số → OCR đọc ký tự → Upload ảnh lên MinIO → Kiểm tra trùng lặp qua Qdrant → Ghi kết quả vào PostgreSQL.

---

### Bước 7: Mở Dashboard và sử dụng

Truy cập địa chỉ sau trên trình duyệt:

```
http://localhost:8000
```

**Cách sử dụng Camera/Video trên Dashboard:**

1. Kéo xuống phần **"Giám Sát Camera AI"**
2. Nhấn nút **▶ Play** để bật Webcam trực tiếp, hoặc nhấn nút **🎬 Film** để tải video từ máy tính
3. Luồng hình ảnh sẽ được gửi qua API → RabbitMQ → AI Worker (`main.py`) tự động xử lý
4. Kết quả nhận diện biển số sẽ hiển thị trong bảng **"Lịch Sử Nhận Diện"** và **"Tra Cứu Phạt Nguội"**

**Các chức năng trên Dashboard:**

| Trang | Mô tả |
|-------|-------|
| **Tổng Quan** | Thống kê tổng số xe, vi phạm, biển số duy nhất |
| **Camera** | Bật webcam hoặc tải video → gửi frame cho AI Worker xử lý |
| **Lịch Sử** | Bảng lịch sử nhận diện với phân trang (6 dòng/trang) |
| **Phạt Nguội** | Danh sách vi phạm giao thông với phân trang (6 dòng/trang) |
| **Tìm Kiếm** | Tra cứu lịch sử theo biển số xe |

> 📌 **Tóm tắt:** Bạn cần mở **2 terminal** song song:
> - **Terminal 1:** Backend API (`uvicorn`)
> - **Terminal 2:** AI Worker (`main.py`)
>
> Sau đó mở trình duyệt tại `http://localhost:8000` để sử dụng.

---

## 🔌 Các Cổng Dịch Vụ

| Dịch vụ | URL | Ghi chú |
|---------|-----|---------|
| **Dashboard** | http://localhost:8000 | Giao diện web chính |
| **API Server** | http://localhost:8000/api/ | REST API (logs, violations, search, stats) |
| **PostgreSQL** | localhost:5432 | DB: `lpr_db` / User: `admin` / Pass: `admin123` |
| **MinIO Console** | http://localhost:9001 | User: `admin` / Pass: `admin123` |
| **MinIO API** | http://localhost:9000 | S3-compatible API |
| **Qdrant** | http://localhost:6333 | Vector Database dashboard |
| **RabbitMQ** | http://localhost:15672 | User: `admin` / Pass: `admin123` |

---

## 🛠 Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|------------|-----------|
| AI Detection | YOLOv10 (Ultralytics) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Backend API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 |
| Object Storage | MinIO |
| Vector DB | Qdrant |
| Message Queue | RabbitMQ |
| Frontend | HTML5 / CSS3 / JavaScript |
| Container | Docker Compose |

---

> 📌 **Bài tập lớn môn Trí Tuệ Nhân Tạo — BTL-THPTTTNT-36**
> Đại học Công nghệ — Đại học Quốc gia Hà Nội (UET - VNU)
