"""
Test phát hiện biển số + OCR dùng thư viện fast-alpr (KHÔNG dùng model trained sẵn của project).
Model fast-alpr được train trên dataset cực lớn, đa quốc gia, xử lý tốt biển nghiêng/mờ/xa.
"""
import argparse
import json
import os
import time
from collections import Counter

import cv2
import numpy as np

from fast_alpr import ALPR

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DETECTOR_CHOICES = [
    "yolo-v9-t-256-license-plate-end2end",
    "yolo-v9-t-384-license-plate-end2end",
    "yolo-v9-t-416-license-plate-end2end",
    "yolo-v9-t-512-license-plate-end2end",
    "yolo-v9-t-640-license-plate-end2end",
    "yolo-v9-s-608-license-plate-end2end",  # nặng nhất, chính xác nhất
]

OCR_CHOICES = [
    "global-plates-mobile-vit-v2-model",   # best cho biển nói chung
    "cct-s-v2-global-model",
    "cct-xs-v2-global-model",
    "european-plates-mobile-vit-v2-model",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LPR bằng fast-alpr (detection + OCR pre-trained, không cần model project)"
    )
    p.add_argument("source", help="File ảnh hoặc folder.")
    p.add_argument("--save", default=None, help="File hoặc folder xuất kết quả.")
    p.add_argument("--no-show", action="store_true", help="Chạy headless.")
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="auto / cuda / cpu (mặc định auto).",
    )
    p.add_argument(
        "--detector", default="yolo-v9-s-608-license-plate-end2end",
        choices=DETECTOR_CHOICES,
        help="Model detect (mặc định s-608 = chính xác nhất).",
    )
    p.add_argument(
        "--ocr", default="global-plates-mobile-vit-v2-model",
        choices=OCR_CHOICES,
        help="Model OCR (mặc định mobile-vit-v2 global).",
    )
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection conf threshold (mặc định 0.35).",
    )
    return p.parse_args()


def build_providers(device: str):
    """Trả về list providers cho onnxruntime."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device in ("cuda", "auto"):
        if "CUDAExecutionProvider" in available:
            return [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
        if device == "cuda":
            print("[WARN] CUDA không khả dụng cho onnxruntime → CPU.")
        return ["CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def annotate(img, results):
    out = img.copy()
    for r in results:
        bb = r.detection.bounding_box
        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
        det_conf = r.detection.confidence
        text = (r.ocr.text if r.ocr else "") or "?"
        # OCR confidence có thể là list (per-char) hoặc float
        ocr_conf = r.ocr.confidence if r.ocr else 0
        if isinstance(ocr_conf, list):
            ocr_conf = float(np.mean(ocr_conf)) if ocr_conf else 0.0

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{text}  {ocr_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(
            out, label, (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
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
    print(f"[INFO] Device: {'GPU (CUDA)' if using_gpu else 'CPU'}")
    print(f"[INFO] Detector: {args.detector}")
    print(f"[INFO] OCR     : {args.ocr}")

    print("[INFO] Đang tải model fast-alpr (lần đầu sẽ download)...")
    alpr = ALPR(
        detector_model=args.detector,
        detector_conf_thresh=args.conf,
        detector_providers=providers,
        ocr_model=args.ocr,
        ocr_device="cuda" if using_gpu else "cpu",
    )
    print("[INFO] Sẵn sàng.\n")

    # Resolve danh sách ảnh
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
            os.path.dirname(os.path.abspath(__file__)), "test_results_alpr"
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
        results = alpr.predict(img)
        dt = (time.time() - t1) * 1000

        name = os.path.basename(path)
        print(f"--- {name}  ({dt:.0f}ms, {len(results)} biển)")
        item = {"image": name, "plates": []}
        for r in results:
            text = (r.ocr.text if r.ocr else "") or ""
            ocr_conf = r.ocr.confidence if r.ocr else 0
            if isinstance(ocr_conf, list):
                ocr_conf = float(np.mean(ocr_conf)) if ocr_conf else 0.0
            print(
                f"   {text:<14}  det={r.detection.confidence:.2f}  ocr={ocr_conf:.2f}"
            )
            if text:
                summary[text] += 1
            bb = r.detection.bounding_box
            item["plates"].append({
                "text": text,
                "det_conf": float(r.detection.confidence),
                "ocr_conf": float(ocr_conf),
                "bbox": [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)],
            })
        if not results:
            print("   (không phát hiện biển)")
        all_json.append(item)

        annotated = annotate(img, results)
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
            cv2.imshow("LPR (fast-alpr)  any=next  q=quit", disp)
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
