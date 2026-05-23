from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Cố định thư mục làm việc (tránh lỗi đường dẫn của PyCharm)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 2. Khởi tạo mô hình
    model = YOLO('yolov10n.pt')

    # 3. Train mô hình LP
    print("🚀 BẮT ĐẦU TRAIN MÔ HÌNH NHẬN DIỆN BIỂN SỐ...")
    results = model.train(
        data='D:/BTL/BTL-THPTTTNT-36/datasets/LP_detection/LP_detection/data_lp.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        project='runs/train',    # Kết quả lưu vào thư mục runs/train/
        name='lp_detection_model'
    )