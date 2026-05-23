

import cv2
import os
import json
import re
import numpy as np
from ultralytics import YOLO

# Đường dẫn model
MODEL_DETECT = 'result_train(pt)/license_detect.pt'
MODEL_OCR = 'result_train(pt)/ocr_dect.pt'

# Các ký tự trong OCR model
CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z']


def validate_plate_format(text):
    """
    Kiểm tra và format lại biển số theo chuẩn VN
    Cân bằng: chấp nhận nhiều format nhưng vẫn chính xác
    """
    if not text or len(text) < 6:
        return None, 0
    
    text = text.upper().strip()
    # Loại bỏ các ký tự đặc biệt thường bị nhận nhầm
    text = text.replace('I', '1').replace('O', '0').replace('.', '').replace('-', '').replace(' ', '')
    text = re.sub(r'[^0-9A-Z]', '', text)
    
    if len(text) < 6:
        return None, 0
    
    # Format chuẩn VN: 2 số + 1 chữ + 4-5 số
    # VD: 30A12345, 51A99999, 75H142599
    
    # Thử format 5 số cuối (8-9 ký tự)
    suffix_match = re.search(r'\d{5}$', text)
    if suffix_match:
        prefix = text[:suffix_match.start()]
        suffix = suffix_match.group()
        prefix_len = len(prefix)
        
        # Prefix: 2-3 số + 1 chữ
        if prefix_len == 3 and re.match(r'^\d{2}[A-Z]$', prefix):
            return f"{prefix}{suffix}", 0.95
        elif prefix_len == 4 and re.match(r'^\d{3}[A-Z]$', prefix):
            return f"{prefix}{suffix}", 0.90
        # Prefix: 2 số + 2 chữ (ít gặp)
        elif prefix_len == 4 and re.match(r'^\d{2}[A-Z]{2}$', prefix):
            return f"{prefix}{suffix}", 0.85
    
    # Thử format 4 số cuối (7-8 ký tự)
    suffix_match = re.search(r'\d{4}$', text)
    if suffix_match:
        prefix = text[:suffix_match.start()]
        suffix = suffix_match.group()
        prefix_len = len(prefix)
        
        # Prefix: 2 số + 1 chữ
        if prefix_len == 3 and re.match(r'^\d{2}[A-Z]$', prefix):
            return f"{prefix}{suffix}", 0.95
        # Prefix: 3 số + 1 chữ  
        elif prefix_len == 4 and re.match(r'^\d{3}[A-Z]$', prefix):
            return f"{prefix}{suffix}", 0.90
        # Prefix: 2 số + 2 chữ
        elif prefix_len == 4 and re.match(r'^\d{2}[A-Z]{2}$', prefix):
            return f"{prefix}{suffix}", 0.85
    
    return None, 0


def validate_plate_fallback(text):
    """Fallback validation - chấp nhận nhiều format hơn"""
    if not text or len(text) < 6:
        return None, 0
    
    text = text.upper().strip()
    # Loại bỏ các ký tự đặc biệt thường bị nhận nhầm
    text = text.replace('I', '1').replace('O', '0').replace('.', '').replace('-', '').replace(' ', '')
    text = re.sub(r'[^0-9A-Z]', '', text)
    
    if len(text) < 6:
        return None, 0
    
    # Kiểm tra có cả số và chữ
    has_digit = bool(re.search(r'\d', text))
    has_letter = bool(re.search(r'[A-Z]', text))
    
    if has_digit and has_letter:
        # Format: 1-3 số + 1-2 chữ + 4-5 số (chuẩn)
        if re.match(r'^\d{1,3}[A-Z]{1,2}\d{4,5}$', text):
            return text, 0.6
        
        # Format: nhiều số + nhiều chữ xen kẽ
        # VD: 27976439V, 51A99999
        if re.match(r'^.*\d+[A-Z].*\d+.*$', text) and len(text) >= 7:
            return text, 0.5
    
    return None, 0


def preprocess_plate(plate_crop):
    """Tiền xử lý ảnh biển số để cải thiện OCR"""
    # Chuyển sang grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()
    
    # Resize nếu ảnh quá nhỏ
    h, w = gray.shape
    if h < 40:
        scale = 40 / h
        new_w = int(w * scale)
        new_h = 40
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Tăng contrast với CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Làm sắc nét
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Giảm nhiễu
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    
    return thresh


def sort_characters_v2(char_data, plate_height, plate_width):
    """
    Sắp xếp ký tự theo đúng thứ tự biển số VN
    - 1 hàng: đọc từ trái qua phải
    - 2 hàng: đọc dòng trên trước, rồi dòng dưới
    
    QUY TẮC QUAN TRỌNG:
    - Dòng trên LUÔN đọc trước dòng dưới (y nhỏ hơn = ở trên ảnh)
    - Trong mỗi dòng: đọc từ trái qua phải (x tăng dần)
    """
    if len(char_data) <= 1:
        return ''.join(c['char'] for c in char_data)
    
    # Loại bỏ các ký tự đặc biệt không phải biển số
    valid_chars = [c for c in char_data if c['char'] not in ['.', '-', ' ']]
    if len(valid_chars) < len(char_data):
        char_data = valid_chars
    
    y_values = [c['y'] for c in char_data]
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    
    # Tính baseline density để xác định 2 hàng
    # 2 hàng nếu: y_range > 25% chiều cao HOẶC có 2 cụm y rời nhau
    is_two_rows = y_range > plate_height * 0.25
    
    if not is_two_rows:
        # 1 hàng: sắp xếp theo x tăng dần (trái qua)
        char_data.sort(key=lambda c: c['x'])
        return ''.join(c['char'] for c in char_data)
    
    # ===== 2 HÀNG =====
    # Xác định ngưỡng y để chia 2 dòng
    # Dòng trên = y NHỎ HƠN (ở trên ảnh), Dòng dưới = y LỚN HƠN (ở dưới ảnh)
    
    # Phương pháp 1: Dùng K-means để phân chia
    y_coords = np.array([[c['y']] for c in char_data])
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(y_coords)
        
        row_top = [c for c, label in zip(char_data, labels) if label == 0]
        row_bottom = [c for c, label in zip(char_data, labels) if label == 1]
        
        # Xác định hàng nào trên, hàng nào dưới bằng trung bình y
        if row_top and row_bottom:
            avg_top = np.mean([c['y'] for c in row_top])
            avg_bottom = np.mean([c['y'] for c in row_bottom])
            
            # Hàng trên phải có y NHỎ HƠN
            if avg_top > avg_bottom:
                row_top, row_bottom = row_bottom, row_top
    except:
        # Fallback: chia theo median
        y_mid = np.median(y_values)
        row_top = [c for c in char_data if c['y'] < y_mid]
        row_bottom = [c for c in char_data if c['y'] >= y_mid]
    
    # Nếu một dòng bị rỗng, đọc hết 1 dòng
    if not row_top:
        row_bottom.sort(key=lambda c: c['x'])
        return ''.join(c['char'] for c in row_bottom)
    if not row_bottom:
        row_top.sort(key=lambda c: c['x'])
        return ''.join(c['char'] for c in row_top)
    
    # Sắp xếp từng hàng theo x tăng dần (trái qua)
    row_top.sort(key=lambda c: c['x'])
    row_bottom.sort(key=lambda c: c['x'])
    
    # ===== QUY TẮC ĐỌC =====
    # Dòng trên LUÔN đọc trước (dù có ít hay nhiều ký tự hơn)
    # Dòng dưới LUÔN đọc SAU (không bao giờ đọc dưới trước)
    
    top_text = ''.join(c['char'] for c in row_top)
    bottom_text = ''.join(c['char'] for c in row_bottom)
    
    # Debug log
    print(f"  [SORT] Top (y<): {top_text} ({len(row_top)}), Bottom (y>): {bottom_text} ({len(row_bottom)})")
    
    # Nối: dòng trên trước, rồi dòng dưới
    return top_text + bottom_text


def sort_characters_simple(boxes, plate_width, plate_height):
    """Sắp xếp ký tự theo đúng thứ tự"""
    if len(boxes) == 0:
        return ""
    
    char_data = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        char = CHARACTERS[cls] if cls < len(CHARACTERS) else '?'
        
        char_data.append({
            'x': (x1 + x2) / 2,
            'y': (y1 + y2) / 2,
            'char': char,
        })
    
    return sort_characters_v2(char_data, plate_height, plate_width)


def recognize_chars(plate_crop, model_ocr):
    """Nhận diện ký tự với nhiều phương pháp"""
    h, w = plate_crop.shape[:2]
    
    # Thử nhiều conf threshold
    best_text = ""
    best_count = 0
    
    for conf_th in [0.3, 0.2, 0.15, 0.1]:
        results = model_ocr(plate_crop, conf=conf_th, verbose=False)
        
        if not results or not results[0].boxes:
            continue
        
        boxes = results[0].boxes.cpu().numpy()
        if len(boxes) == 0:
            continue
        
        char_data = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            char = CHARACTERS[cls] if cls < len(CHARACTERS) else '?'
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            char_data.append({'x': cx, 'y': cy, 'char': char})
        
        text = sort_characters_v2(char_data, h, w)
        
        # Ưu tiên kết quả có nhiều ký tự hơn
        if len(text) > best_count:
            best_text = text
            best_count = len(text)
    
    return best_text


def process_image(img, model_detect, model_ocr):
    """Xử lý ảnh: detect -> recognize -> validate"""
    all_results = []
    
    # Thử với nhiều cấu hình để tìm nhiều biển nhất
    configs = [
        {'conf': 0.25, 'iou': 0.45},
        {'conf': 0.1, 'iou': 0.35},
        {'conf': 0.03, 'iou': 0.2},
    ]
    
    for cfg in configs:
        results = model_detect(img, conf=cfg['conf'], iou=cfg['iou'], verbose=False)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                
                # Bỏ qua boxes quá nhỏ
                if (x2 - x1) < 20 or (y2 - y1) < 15:
                    continue
                
                # Bỏ qua nếu trùng với kết quả đã có
                is_dup = False
                for existing in all_results:
                    ex = existing['bbox']
                    iou = compute_iou([x1, y1, x2, y2], ex)
                    if iou > 0.5:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Crop với padding
                pad = 3
                h, w = img.shape[:2]
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)
                
                plate_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if plate_crop.size == 0:
                    continue
                
                # Nhận diện
                plate_text = recognize_chars(plate_crop, model_ocr)
                
                # Validate format
                valid_text, valid_score = validate_plate_format(plate_text)
                
                # Fallback: thử format rộng hơn
                if not valid_text:
                    valid_text, valid_score = validate_plate_fallback(plate_text)
                
                if valid_text:
                    all_results.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'text': valid_text,
                        'valid_score': valid_score
                    })
    
    return all_results


def compute_iou(box1, box2):
    """Tính IoU giữa 2 boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def test_folder(folder_path):
    """Test trên folder ảnh"""
    print(f"[INFO] Loading detection model: {MODEL_DETECT}")
    model_detect = YOLO(MODEL_DETECT)
    
    print(f"[INFO] Loading OCR model: {MODEL_OCR}")
    model_ocr = YOLO(MODEL_OCR)
    
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(image_exts)]
    
    if not image_files:
        print(f"[ERROR] Khong tim thay anh nao")
        return
    
    print(f"[INFO] Tim thay {len(image_files)} anh\n")
    
    os.makedirs('test_results', exist_ok=True)
    
    all_results = []
    
    for img_file in sorted(image_files):
        img_path = os.path.join(folder_path, img_file)
        print(f"--- Xu ly: {img_file} ---")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        plates = process_image(img, model_detect, model_ocr)
        
        annotated = img.copy()
        image_result = {"image": img_file, "plates": []}
        
        for plate in plates:
            x1, y1, x2, y2 = plate['bbox']
            conf = plate['conf']
            text = plate['text']
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"  Bien so: {text}")
            image_result["plates"].append({"text": text, "confidence": conf})
        
        if not plates:
            print("  Khong phat hien bien so!")
        
        all_results.append(image_result)
        cv2.imwrite(f"test_results/result_{img_file}", annotated)
    
    with open('test_results/results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] Hoan tat!")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_folder('test_image')
