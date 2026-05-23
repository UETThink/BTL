"""
AI SYSTEM ENGINEERING PROJECT
Module: OCR Character Detection Training
Description: Script dùng để huấn luyện mô hình YOLOv10 nhận diện từng ký tự (chữ/số) trên biển số.
Lưu ý: Dataset yêu cầu 30+ classes (0-9, A-Z).
"""

import os
from ultralytics import YOLO


def main():
    # Cố định thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("[INFO] Khởi tạo mô hình YOLOv10 Nano cho bài toán Character Recognition (OCR)...")
    # Khởi tạo lại một mô hình YOLOv10 gốc mới tinh, KHÔNG dùng lại lp_model.pt của LP
    model = YOLO('yolov10n.pt')

    # Định nghĩa cấu hình huấn luyện (OCR cần ảnh nhỏ hơn nhưng học kỹ hơn)
    dataset_yaml = 'datasets/OCR/OCR/data_ocr.yaml'
    epochs = 50
    batch_size = 16
    image_size = 320  # Kích thước ảnh tối ưu cho ký tự cắt nhỏ

    print(f"[INFO] Bắt đầu huấn luyện OCR với dataset: {dataset_yaml}")
    print(f"[INFO] Tham số: Epochs={epochs}, Batch={batch_size}, ImageSize={image_size}")

    # Thực thi huấn luyện
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project='runs/train',
        name='ocr_detection_model',
        verbose=True
    )

    print("[SUCCESS] Quá trình huấn luyện OCR hoàn tất!")
    print("[INFO] Trọng số tốt nhất (lp_model.pt) được lưu tại: runs/train/ocr_detection_model/weights/lp_model.pt")


if __name__ == '__main__':
    main()