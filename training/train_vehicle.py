from ultralytics import YOLO

def main():
    print("[INFO] Bắt đầu quá trình huấn luyện mô hình Nhận Diện Xe Cộ VN...")

    # Load model gốc làm nền tảng
    model = YOLO('yolov10n.pt') 

    # Bắt đầu train
    results = model.train(
        # Trỏ đường dẫn vào thẳng file yaml nằm trong thư mục data
        data='D:/ML1/BTL-THPTTTNT-36/datasets/Vehicle_Detection/data_vehD.yaml', 
        epochs=50,                 
        imgsz=640,                 
        batch=16,                  
        project='runs/train',      
        name='vehicle_detection_model', 
        device='0',
        workers=2               
    )

    print("[INFO] Hoàn tất! Mô hình đã được lưu tại: runs/train/vehicle_detection_model/weights/best.pt")

if __name__ == '__main__':
    main()