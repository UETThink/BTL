import argparse
import json
import os
import time
from collections import Counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from test_license_plate_final import (
    CHARACTERS,
    MODEL_DETECT,
    MODEL_OCR,
    compute_iou,
    sort_characters_v2,
    validate_plate_format,
    validate_plate_fallback,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện biển số + OCR trên ảnh (1 ảnh hoặc cả folder)"
    )
    parser.add_argument(
        "source",
        help="Đường dẫn 1 file ảnh HOẶC 1 folder chứa ảnh.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="File ảnh xuất (nếu source là ảnh) hoặc folder xuất (nếu source là folder). "
             "Mặc định: ./test_results/",
    )
    parser.add_argument("--no-show", action="store_true", help="Không hiện cửa sổ.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="auto / cuda / cpu (mặc định auto).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Kích thước input YOLO (mặc định 640)."
    )
    parser.add_argument(
        "--half", action="store_true", help="Bật FP16 trên GPU."
    )
    parser.add_argument(
        "--det-conf", type=float, default=0.15, help="Conf detect (mặc định 0.15)."
    )
    parser.add_argument(
        "--ocr-conf", type=float, default=0.25, help="Conf OCR (mặc định 0.25)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Chế độ nhanh: 1 lần detect + 1 lần OCR (kém chính xác hơn).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng → chuyển sang CPU.")
        return "cpu"
    return name


def upscale_plate(crop, target_h=96):
    """Upscale biển số để OCR thấy ký tự rõ hơn."""
    h, w = crop.shape[:2]
    if h >= target_h:
        return crop
    scale = target_h / h
    return cv2.resize(crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)


def _ocr_once(crop, model_ocr, conf, imgsz, device, half):
    """1 lần OCR — trả về (text, num_chars)."""
    results = model_ocr(
        crop, conf=conf, imgsz=imgsz, device=device, half=half, verbose=False
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return "", 0
    boxes = results[0].boxes.cpu().numpy()
    char_data = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        ch = CHARACTERS[cls] if cls < len(CHARACTERS) else "?"
        char_data.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "char": ch})
    h, w = crop.shape[:2]
    return sort_characters_v2(char_data, h, w), len(char_data)


def read_plate(plate_crop, model_ocr, conf, imgsz, device, half, multi=True):
    """OCR với upscale + thử nhiều conf nếu kết quả không hợp lệ."""
    if plate_crop.size == 0:
        return ""
    crop = upscale_plate(plate_crop, target_h=96)

    if not multi:
        text, _ = _ocr_once(crop, model_ocr, conf, imgsz, device, half)
        return text

    best_text = ""
    best_score = -1  # ưu tiên: hợp lệ format > nhiều ký tự
    for c in (conf, 0.15, 0.08):
        text, n = _ocr_once(crop, model_ocr, c, imgsz, device, half)
        if n == 0:
            continue
        valid, vscore = validate_plate_format(text)
        if not valid:
            valid, vscore = validate_plate_fallback(text)
        # Score: ký tự + bonus nếu format chuẩn VN
        score = n + (10 if valid else 0)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def detect_plates(frame, model_detect, conf, imgsz, device, half, fast):
    """Trả về list bbox [x1,y1,x2,y2,conf], dedup bằng IoU."""
    if fast:
        configs = [conf]
    else:
        # Multi-conf: bắt cả biển dễ và biển khó (nghiêng, mờ, xa)
        configs = [conf, max(0.08, conf - 0.1), 0.05]
        # Loại trùng
        configs = sorted(set(round(c, 3) for c in configs), reverse=True)

    found = []
    for c in configs:
        results = model_detect(
            frame,
            conf=c,
            iou=0.45,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )
        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        boxes = results[0].boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) < 20 or (y2 - y1) < 15:
                continue
            score = float(box.conf[0])
            dup = False
            for ex in found:
                if compute_iou([x1, y1, x2, y2], ex[:4]) > 0.5:
                    dup = True
                    break
            if not dup:
                found.append([x1, y1, x2, y2, score])
    return found


def process_one(img, model_detect, model_ocr, args, device):
    h, w = img.shape[:2]
    plates = detect_plates(
        img, model_detect, args.det_conf, args.imgsz, device, args.half, args.fast
    )
    results = []
    for x1, y1, x2, y2, conf in plates:
        # Padding to nhỏ hơn cho biển nhỏ, lớn hơn cho biển to
        pad = max(3, (y2 - y1) // 20)
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        crop = img[y1p:y2p, x1p:x2p]

        raw = read_plate(
            crop, model_ocr, args.ocr_conf, args.imgsz, device, args.half,
            multi=not args.fast,
        )
        valid, score = validate_plate_format(raw)
        if not valid:
            valid, score = validate_plate_fallback(raw)
        results.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "raw": raw,
            "text": valid if valid else raw,
            "valid_score": score,
        })
    return results


def annotate(img, plates):
    out = img.copy()
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        text = p["text"] or "?"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(
            out, text, (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
        )
    return out


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if device == "cuda":
        print(f"[INFO] Dùng GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Dùng CPU.")
        if args.half:
            print("[WARN] --half chỉ có ý nghĩa trên GPU, tắt FP16.")
            args.half = False

    print(f"[INFO] Loading detection: {MODEL_DETECT}")
    model_detect = YOLO(MODEL_DETECT)
    print(f"[INFO] Loading OCR: {MODEL_OCR}")
    model_ocr = YOLO(MODEL_OCR)

    if device == "cuda":
        dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        model_detect(dummy, imgsz=args.imgsz, device=device, half=args.half, verbose=False)
        model_ocr(dummy, imgsz=args.imgsz, device=device, half=args.half, verbose=False)

    # Resolve danh sách ảnh
    src = os.path.abspath(args.source)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Không tìm thấy: {src}")

    if os.path.isdir(src):
        files = sorted(
            os.path.join(src, f)
            for f in os.listdir(src)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not files:
            print(f"[ERROR] Folder {src} không có ảnh.")
            return
        out_dir = args.save or os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(out_dir, exist_ok=True)
        is_folder = True
    else:
        files = [src]
        out_dir = None
        is_folder = False

    print(f"[INFO] Sẽ xử lý {len(files)} ảnh.\n")

    all_json = []
    total_plates = 0
    t0 = time.time()

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"[SKIP] Không đọc được: {path}")
            continue

        t_img = time.time()
        plates = process_one(img, model_detect, model_ocr, args, device)
        dt = (time.time() - t_img) * 1000

        name = os.path.basename(path)
        print(f"--- {name}  ({dt:.0f}ms, {len(plates)} biển)")
        for p in plates:
            tag = p["text"] or "?"
            extra = "" if p["text"] == p["raw"] else f"  (raw: {p['raw']})"
            print(f"   {tag}{extra}  conf={p['conf']:.2f}")
        if not plates:
            print("   (không phát hiện biển nào)")

        total_plates += len(plates)
        annotated = annotate(img, plates)

        # Lưu
        if is_folder:
            cv2.imwrite(os.path.join(out_dir, f"result_{name}"), annotated)
        elif args.save:
            cv2.imwrite(args.save, annotated)
            print(f"[INFO] Đã lưu: {args.save}")

        all_json.append({
            "image": name,
            "plates": [
                {"text": p["text"], "raw": p["raw"], "bbox": p["bbox"], "conf": p["conf"]}
                for p in plates
            ],
        })

        # Hiển thị
        if not args.no_show:
            # Resize nếu ảnh quá lớn
            disp = annotated
            mh, mw = 900, 1400
            h, w = disp.shape[:2]
            if h > mh or w > mw:
                scale = min(mh / h, mw / w)
                disp = cv2.resize(disp, (int(w * scale), int(h * scale)))
            cv2.imshow("LPR Image Test (any key=next, q=quit)", disp)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()

    if is_folder and out_dir:
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Đã ghi JSON: {os.path.join(out_dir, 'results.json')}")

    total_dt = time.time() - t0
    print(f"\n[INFO] {len(files)} ảnh / {total_dt:.1f}s — {total_plates} biển đọc được")

    # Thống kê biển số đếm nhiều nhất
    counter = Counter()
    for item in all_json:
        for p in item["plates"]:
            if p["text"]:
                counter[p["text"]] += 1
    if counter:
        print("[INFO] Top biển số xuất hiện:")
        for txt, c in counter.most_common(10):
            print(f"   {txt:<12}  ({c} lần)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
