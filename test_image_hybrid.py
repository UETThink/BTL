"""
Pipeline lai: fast-alpr cho DETECTION + PaddleOCR cho OCR.
- fast-alpr: bắt biển chuẩn, kể cả nghiêng/xa
- PaddleOCR: đọc tốt biển VN 2 dòng (text detection + recognition đa dòng)
"""
import argparse
import json
import os
import re
import time
import warnings
from collections import Counter

import cv2
import numpy as np

warnings.filterwarnings("ignore")
os.environ["FLAGS_logging_level"] = "ERROR"

from fast_alpr.default_detector import DefaultDetector  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DETECTOR_CHOICES = [
    "yolo-v9-t-256-license-plate-end2end",
    "yolo-v9-t-384-license-plate-end2end",
    "yolo-v9-t-416-license-plate-end2end",
    "yolo-v9-t-512-license-plate-end2end",
    "yolo-v9-t-640-license-plate-end2end",
    "yolo-v9-s-608-license-plate-end2end",
]

PLATE_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_args():
    p = argparse.ArgumentParser(
        description="LPR hybrid: fast-alpr detect + PaddleOCR read"
    )
    p.add_argument("source", help="File ảnh hoặc folder.")
    p.add_argument("--save", default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"]
    )
    p.add_argument(
        "--detector", default="yolo-v9-s-608-license-plate-end2end",
        choices=DETECTOR_CHOICES,
    )
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection conf threshold.",
    )
    p.add_argument(
        "--tile", type=int, default=1,
        help="Tile detection: chia ảnh thành NxN tiles để bắt biển nhỏ/xa "
             "(1=tắt, 2=2x2, 3=3x3). Mặc định 1.",
    )
    p.add_argument(
        "--tile-overlap", type=float, default=0.25,
        help="Tỉ lệ overlap giữa các tile (mặc định 0.25 = 25%%).",
    )
    return p.parse_args()


def build_providers(device: str):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    if device == "cuda":
        print("[WARN] CUDA không có cho onnxruntime → CPU.")
    return ["CPUExecutionProvider"]


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def nms_dedup(dets, iou_thresh=0.4):
    """dets = list of (x1,y1,x2,y2,conf). Giữ box conf cao nhất khi IoU > thresh."""
    dets = sorted(dets, key=lambda d: -d[4])
    kept = []
    for d in dets:
        if all(iou(d[:4], k[:4]) <= iou_thresh for k in kept):
            kept.append(d)
    return kept


def detect_with_tiling(img, detector, tile_n=1, overlap=0.25):
    """
    Detect biển. Nếu tile_n>1: chạy detect trên full image + NxN tiles overlap,
    rồi NMS dedup. Trả về list of (x1,y1,x2,y2,conf).
    """
    h, w = img.shape[:2]
    all_dets = []

    # Detect trên full image
    for d in detector.predict(img):
        bb = d.bounding_box
        all_dets.append((int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2), float(d.confidence)))

    if tile_n <= 1:
        return nms_dedup(all_dets)

    # Detect trên các tile
    th, tw = h // tile_n, w // tile_n
    oh, ow = int(th * overlap), int(tw * overlap)
    for i in range(tile_n):
        for j in range(tile_n):
            y0 = max(0, i * th - oh)
            y1 = min(h, (i + 1) * th + oh)
            x0 = max(0, j * tw - ow)
            x1 = min(w, (j + 1) * tw + ow)
            tile = img[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            for d in detector.predict(tile):
                bb = d.bounding_box
                all_dets.append((
                    int(bb.x1) + x0, int(bb.y1) + y0,
                    int(bb.x2) + x0, int(bb.y2) + y0,
                    float(d.confidence),
                ))

    return nms_dedup(all_dets)


def upscale_plate(crop, target_h=160):
    """Upscale biển crop để PaddleOCR đọc chính xác hơn."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    if h < target_h:
        scale = target_h / h
        crop = cv2.resize(crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    # Pad trắng xung quanh để OCR không cắt mép
    pad = 12
    crop = cv2.copyMakeBorder(
        crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return crop


def clean_plate_text(text):
    """Bỏ ký tự không phải chữ/số, viết hoa."""
    text = text.upper()
    return "".join(c for c in text if c in PLATE_CHARS)


def ocr_plate(crop, ocr):
    """Đọc 1 biển. PaddleOCR trả về list các vùng text → ghép theo y rồi theo x."""
    if crop.size == 0:
        return "", 0.0

    big = upscale_plate(crop, target_h=160)
    result = ocr.ocr(big, cls=True)
    if not result or not result[0]:
        return "", 0.0

    items = []  # (y_center, x_center, text, conf)
    for line in result[0]:
        box, (text, conf) = line
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        cy = sum(ys) / 4
        cx = sum(xs) / 4
        h_box = max(ys) - min(ys)
        items.append((cy, cx, text, float(conf), h_box))

    if not items:
        return "", 0.0

    # Phân dòng: 2 dòng nếu khoảng cách y > 0.5 * chiều cao box trung bình
    items.sort(key=lambda x: x[0])
    avg_h = np.mean([it[4] for it in items])
    rows = [[items[0]]]
    for it in items[1:]:
        if it[0] - rows[-1][-1][0] > avg_h * 0.5:
            rows.append([it])
        else:
            rows[-1].append(it)

    # Trong mỗi dòng: sort theo x, ghép text
    pieces = []
    confs = []
    for row in rows:
        row.sort(key=lambda x: x[1])
        for cy, cx, text, conf, h in row:
            pieces.append(text)
            confs.append(conf)

    raw = " ".join(pieces)
    cleaned = clean_plate_text(raw)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return cleaned, avg_conf


def annotate(img, items):
    out = img.copy()
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        text = it["text"] or "?"
        label = f"{text}  d={it['det_conf']:.2f} o={it['ocr_conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(
            out, label, (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
        )
    return out


def main():
    args = parse_args()
    providers = build_providers(args.device)
    using_gpu = any(
        (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
        or p == "CUDAExecutionProvider"
        for p in providers
    )
    print(f"[INFO] Device      : {'GPU (CUDA)' if using_gpu else 'CPU'}")
    print(f"[INFO] Detector    : {args.detector}")
    print(f"[INFO] OCR engine  : PaddleOCR (en, angle_cls)")

    print("[INFO] Loading fast-alpr detector...")
    detector = DefaultDetector(
        model_name=args.detector,
        conf_thresh=args.conf,
        providers=providers,
    )

    print("[INFO] Loading PaddleOCR (lần đầu sẽ download model)...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=using_gpu,
        show_log=False,
    )
    print("[INFO] Sẵn sàng.\n")

    src = os.path.abspath(args.source)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if os.path.isdir(src):
        files = sorted(
            os.path.join(src, f) for f in os.listdir(src)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not files:
            print(f"[ERROR] {src} không có ảnh.")
            return
        out_dir = args.save or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_results_hybrid"
        )
        os.makedirs(out_dir, exist_ok=True)
        is_folder = True
    else:
        files = [src]
        out_dir = None
        is_folder = False

    print(f"[INFO] Xử lý {len(files)} ảnh.\n")

    summary = Counter()
    all_json = []
    t0 = time.time()

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"[SKIP] {path}")
            continue

        t1 = time.time()
        detections = detect_with_tiling(img, detector, args.tile, args.tile_overlap)
        items = []
        for x1, y1, x2, y2, det_conf in detections:
            h_img, w_img = img.shape[:2]
            pad = max(3, (y2 - y1) // 20)
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(w_img, x2 + pad); y2p = min(h_img, y2 + pad)
            crop = img[y1p:y2p, x1p:x2p]

            text, ocr_conf = ocr_plate(crop, ocr)
            items.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "det_conf": det_conf,
                "ocr_conf": ocr_conf,
            })
        dt = (time.time() - t1) * 1000

        name = os.path.basename(path)
        print(f"--- {name}  ({dt:.0f}ms, {len(items)} biển)")
        for it in items:
            print(
                f"   {it['text']:<14}  det={it['det_conf']:.2f}  ocr={it['ocr_conf']:.2f}"
            )
            if it["text"]:
                summary[it["text"]] += 1
        if not items:
            print("   (không phát hiện biển)")
        all_json.append({"image": name, "plates": items})

        annotated = annotate(img, items)
        if is_folder:
            cv2.imwrite(os.path.join(out_dir, f"result_{name}"), annotated)
        elif args.save:
            cv2.imwrite(args.save, annotated)
            print(f"[INFO] Đã lưu: {args.save}")

        if not args.no_show:
            disp = annotated
            mh, mw = 900, 1400
            h, w = disp.shape[:2]
            if h > mh or w > mw:
                s = min(mh / h, mw / w)
                disp = cv2.resize(disp, (int(w * s), int(h * s)))
            cv2.imshow("LPR Hybrid  any=next  q=quit", disp)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()

    if is_folder and out_dir:
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Đã ghi JSON: {os.path.join(out_dir, 'results.json')}")

    total = time.time() - t0
    print(f"\n[INFO] {len(files)} ảnh / {total:.1f}s")
    if summary:
        print("[INFO] Biển số đọc được:")
        for t, c in summary.most_common(20):
            print(f"   {t:<14} ({c} lần)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
