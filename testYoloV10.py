
from ultralytics import YOLO
import easyocr
import cv2
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# 1. Khởi tạo mô hình YOLO và Reader OCR
model = YOLO('result_train(pt)/license_detect.pt')
# GPU=False nếu bạn chưa cài CUDA, GPU=True nếu đã cài thành công
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)  # verbose=False để đỡ rác terminal

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 2. Cắt vùng biển số
            # 2. CẮT VÙNG BIỂN SỐ TỪ KHUNG HÌNH GỐC
            plate_img = frame[y1:y2, x1:x2]

            if plate_img.size > 0:
                # --- BẮT ĐẦU THÊM TIỀN XỬ LÝ ẢNH (PREPROCESSING) ---

                # Bước a: Phóng to ảnh lên gấp 2.5 lần để OCR nhìn rõ hơn
                plate_resized = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

                # Bước b: Chuyển ảnh sang trắng đen (Grayscale) để loại bỏ nhiễu màu
                gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)

                # (Tùy chọn) Bước c: Tăng độ tương phản nếu cần
                # gray = cv2.equalizeHist(gray)

                # --- KẾT THÚC TIỀN XỬ LÝ ---

                # 3. Đọc chữ với các thiết lập "ép" EasyOCR đọc chuẩn biển số
                # allowlist: Chỉ cho phép đọc số, chữ cái in hoa và dấu chấm. Loại bỏ hoàn toàn các ký tự nhiễu như @, #, $, %,...
                # paragraph=True: Giúp gom các dòng chữ gần nhau (như biển 2 dòng) tốt hơn.
                ocr_results = reader.readtext(
                    gray,
                    detail=0,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.',
                    paragraph=True
                )

                plate_text = ""
                if len(ocr_results) > 0:
                    plate_text = " ".join(ocr_results).upper()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # In dòng chữ OCR đọc được lên phía trên cái khung
                cv2.putText(frame, f"LP: {plate_text}", (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # ------------------------------------------------------

                # Cập nhật lại cửa sổ hiển thị ảnh để xem ảnh đã qua xử lý trông như nào
                cv2.imshow('Bien so duoc cat (Da xu ly)', gray)
    cv2.imshow('He thong ALPR UET', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()