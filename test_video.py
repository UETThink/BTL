import argparse
import os
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from test_license_plate_final import (
    CHARACTERS,
    MODEL_DETECT,
    MODEL_OCR,
    sort_characters_v2,
    validate_plate_format,
    validate_plate_fallback,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện biển số + OCR trên video (tối ưu cho real-time GPU)"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Đường dẫn file video. Mặc định '0' = webcam.",
    )
    parser.add_argument("--save", default=None, help="Xuất video kết quả (vd: out.mp4).")
    parser.add_argument("--no-show", action="store_true", help="Chạy không hiện cửa sổ.")
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Xử lý 1/N frame (mặc định 1).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="auto / cuda / cpu (mặc định auto).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước input cho YOLO (mặc định 640). Giảm xuống 480/416 để tăng tốc.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Bật FP16 (chỉ có ý nghĩa trên GPU) — nhanh hơn ~30-50%%.",
    )
    parser.add_argument(
        "--det-conf", type=float, default=0.25, help="Conf threshold detect (mặc định 0.25)."
    )
    parser.add_argument(
        "--ocr-conf", type=float, default=0.25, help="Conf threshold OCR (mặc định 0.25)."
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng → chuyển sang CPU.")
        return "cpu"
    return name


def open_source(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Không tìm thấy video: {source}")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")
    return cap


def read_plate(plate_crop, model_ocr, conf, imgsz, device, half):
    """1 lần OCR cho 1 biển — đủ nhanh cho video."""
    if plate_crop.size == 0:
        return ""
    results = model_ocr(
        plate_crop,
        conf=conf,
        imgsz=imgsz,
        device=device,
        half=half,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return ""
    boxes = results[0].boxes.cpu().numpy()
    char_data = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        ch = CHARACTERS[cls] if cls < len(CHARACTERS) else "?"
        char_data.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "char": ch})
    h, w = plate_crop.shape[:2]
    return sort_characters_v2(char_data, h, w)


def detect_and_read(frame, model_detect, model_ocr, args, device):
    """1 lần detect + 1 lần OCR mỗi biển — tốc độ tối đa."""
    results = model_detect(
        frame,
        conf=args.det_conf,
        iou=0.45,
        imgsz=args.imgsz,
        device=device,
        half=args.half,
        verbose=False,
    )
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes.cpu().numpy()
    plates = []
    h, w = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < 20 or (y2 - y1) < 15:
            continue
        pad = 3
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        plate_crop = frame[y1p:y2p, x1p:x2p]

        raw_text = read_plate(
            plate_crop, model_ocr, args.ocr_conf, args.imgsz, device, args.half
        )

        valid, _ = validate_plate_format(raw_text)
        if not valid:
            valid, _ = validate_plate_fallback(raw_text)

        plates.append({
            "bbox": [x1, y1, x2, y2],
            "text": valid if valid else raw_text,
        })
    return plates


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

    print(f"[INFO] Loading detection model: {MODEL_DETECT}")
    model_detect = YOLO(MODEL_DETECT)
    print(f"[INFO] Loading OCR model: {MODEL_OCR}")
    model_ocr = YOLO(MODEL_OCR)

    # Warmup để CUDA biên dịch kernel
    if device == "cuda":
        dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            model_detect(dummy, imgsz=args.imgsz, device=device, half=args.half, verbose=False)
            model_ocr(dummy, imgsz=args.imgsz, device=device, half=args.half, verbose=False)
        print(f"[INFO] Warmup xong. imgsz={args.imgsz}, half={args.half}")

    cap = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {width}x{height} @ {src_fps:.1f}fps")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (width, height))
        print(f"[INFO] Ghi kết quả vào: {args.save}")

    vote_history = defaultdict(Counter)
    last_results = []

    frame_idx = 0
    t_start = time.time()
    t_window = t_start
    frames_window = 0
    fps_disp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frames_window += 1

        if frame_idx % args.skip == 0:
            plates = detect_and_read(frame, model_detect, model_ocr, args, device)
            for p in plates:
                x1, y1, x2, y2 = p["bbox"]
                key = (x1 // 40, y1 // 40, (x2 - x1) // 40, (y2 - y1) // 40)
                if p["text"]:
                    vote_history[key][p["text"]] += 1
                if vote_history[key]:
                    p["text"] = vote_history[key].most_common(1)[0][0]
            last_results = plates

        for p in last_results:
            x1, y1, x2, y2 = p["bbox"]
            text = p["text"] or "?"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(
                frame, text, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
            )

        # FPS đo trên cửa sổ 1 giây cho mượt
        now = time.time()
        if now - t_window >= 1.0:
            fps_disp = frames_window / (now - t_window)
            t_window = now
            frames_window = 0
        cv2.putText(
            frame, f"FPS: {fps_disp:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

        if writer is not None:
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("LPR Video Test (q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"\n[INFO] {frame_idx} frame / {total:.1f}s = {frame_idx/total:.1f} FPS trung bình")
    print("[INFO] Tổng kết biển số đã đọc:")
    summary = Counter()
    for counter in vote_history.values():
        text, count = counter.most_common(1)[0]
        summary[text] += count
    for text, count in summary.most_common():
        print(f"  {text:<12} ({count} lần)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
