from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "model",
    "resolution_width",
    "image_path",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "class_id",
    "class_name",
]

_DLL_DIRECTORY_HANDLES = []
_DLL_DIRECTORIES_ADDED = set()


def build_yolo_model_specs(mode="pretrained", weights_dir="chokepoint_data/modeling/yolo_finetuned_weights"):
    pretrained = [
        ("YOLOv5n", "yolov5nu.pt"),
        ("YOLOv8n", "yolov8n.pt"),
        ("YOLOv12n", "paper/yolo12n.pt"),
    ]
    finetuned = [
        ("YOLOv5n-FT", str(Path(weights_dir) / "yolov5n_face.pt")),
        ("YOLOv8n-FT", str(Path(weights_dir) / "yolov8n_face.pt")),
        ("YOLOv12n-FT", str(Path(weights_dir) / "yolov12n_face.pt")),
    ]
    if mode == "pretrained":
        return pretrained
    if mode == "finetuned":
        missing = [path for _, path in finetuned if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing fine-tuned YOLO weights: {missing}")
        return finetuned
    if mode == "both":
        missing = [path for _, path in finetuned if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing fine-tuned YOLO weights: {missing}")
        return pretrained + finetuned
    raise ValueError(f"Unsupported yolo mode: {mode}")


def select_torch_device(requested="auto"):
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def add_cuda_dll_directories():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    prefix = Path(sys.prefix)
    candidate_paths = [
        prefix / "Library" / "bin",
        prefix / "Lib" / "site-packages" / "torch" / "lib",
    ]
    nvidia_root = prefix / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.exists():
        candidate_paths.extend(path / "bin" for path in nvidia_root.iterdir() if path.is_dir())
    for path in candidate_paths:
        resolved = str(path.resolve()) if path.exists() else None
        if resolved and resolved not in _DLL_DIRECTORIES_ADDED:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved))
            _DLL_DIRECTORIES_ADDED.add(resolved)
            current_path = os.environ.get("PATH", "")
            path_parts = [part.lower() for part in current_path.split(os.pathsep) if part]
            if resolved.lower() not in path_parts:
                os.environ["PATH"] = resolved + os.pathsep + current_path


def select_scrfd_ctx(requested="auto"):
    if requested != "auto":
        return int(requested)
    try:
        add_cuda_dll_directories()
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return 0
    except Exception:
        pass
    return -1


def align_to_stride(value, stride=32):
    return int(np.ceil(value / stride) * stride)


def resolution_metadata(width):
    height = int(round(width * 3 / 4))
    label = f"{height}p ({width}x{height})"
    return height, label


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def voc101_ap(gt_df, pred_df, iou_threshold=0.5):
    gt_by_path = {
        row.image_path: [row.x1, row.y1, row.x2, row.y2]
        for row in gt_df.itertuples(index=False)
    }
    if pred_df.empty:
        return 0.0

    pred_sorted = pred_df.sort_values("conf", ascending=False)
    matched = set()
    tp_flags = []
    fp_flags = []
    for row in pred_sorted.itertuples(index=False):
        gt_box = gt_by_path.get(row.image_path)
        if gt_box is None:
            tp_flags.append(0)
            fp_flags.append(1)
            continue
        current_iou = box_iou([row.x1, row.y1, row.x2, row.y2], gt_box)
        if current_iou >= iou_threshold and row.image_path not in matched:
            matched.add(row.image_path)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    precisions = []
    recalls = []
    cum_tp = 0
    cum_fp = 0
    total_gt = len(gt_by_path)
    for tp, fp in zip(tp_flags, fp_flags):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)

    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        candidates = [p for p, r in zip(precisions, recalls) if r >= threshold]
        ap += (max(candidates) if candidates else 0.0) / 101.0
    return float(ap)


def evaluate(gt_df, pred_df, conf_threshold=0.25, iou_threshold=0.5):
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=PREDICTION_COLUMNS)
    gt_by_path = {
        row.image_path: [row.x1, row.y1, row.x2, row.y2]
        for row in gt_df.itertuples(index=False)
    }
    thresholded = pred_df[pred_df["conf"] >= conf_threshold].copy()
    preds_by_path = defaultdict(list)
    for row in thresholded.itertuples(index=False):
        preds_by_path[row.image_path].append([row.x1, row.y1, row.x2, row.y2, row.conf])

    tp = fp = fn = 0
    matched_ious = []
    for image_path, gt_box in gt_by_path.items():
        preds = sorted(preds_by_path.get(image_path, []), key=lambda item: item[4], reverse=True)
        best_iou = 0.0
        for pred in preds:
            best_iou = max(best_iou, box_iou(pred[:4], gt_box))
        if best_iou >= iou_threshold:
            tp += 1
            fp += max(0, len(preds) - 1)
            matched_ious.append(best_iou)
        else:
            fn += 1
            fp += len(preds)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50_voc101": voc101_ap(gt_df, pred_df, iou_threshold),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def sample_split(df, split_name, n, seed):
    split_df = df[df["split"] == split_name].copy()
    if n > 0 and n < len(split_df):
        return split_df.sample(n=n, random_state=seed)
    return split_df


def build_sample(manifest_path, train_n=200, eval_n=400, seed=391, sample_mode="mixed"):
    df = pd.read_csv(manifest_path)
    if "split" not in df.columns and "p1_indoor_split" in df.columns:
        df = df.rename(columns={"p1_indoor_split": "split"})
    if "split" not in df.columns:
        raise ValueError("Manifest must contain either 'split' or 'p1_indoor_split'.")

    if sample_mode == "eval":
        return sample_split(df, "eval", eval_n, seed + 1)
    if sample_mode == "train_dev":
        return sample_split(df, "train_dev", train_n, seed)
    if sample_mode in {"mixed", "all"}:
        train_df = sample_split(df, "train_dev", train_n, seed)
        eval_df = sample_split(df, "eval", eval_n, seed + 1)
        return pd.concat([train_df, eval_df], ignore_index=True)
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def load_resized_bgr(image_path, width):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    original_h, original_w = image.shape[:2]
    height = int(round(original_h * width / original_w))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized, (original_w / width, original_h / height)


def run_yolo_resolution(model_name, weight_path, sample_df, width, predict_conf=0.25, device="auto"):
    from ultralytics import YOLO

    selected_device = select_torch_device(device)
    inference_size = align_to_stride(width)
    model = YOLO(weight_path)
    pred_rows = []
    start = time.perf_counter()
    for idx, row in enumerate(sample_df.itertuples(index=False), start=1):
        image, scale = load_resized_bgr(row.image_path, width)
        if image is None:
            continue
        sx, sy = scale
        result = model.predict(
            source=image,
            imgsz=inference_size,
            conf=predict_conf,
            device=selected_device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, classes):
                pred_rows.append({
                    "model": model_name,
                    "resolution_width": width,
                    "image_path": row.image_path,
                    "x1": float(box[0] * sx),
                    "y1": float(box[1] * sy),
                    "x2": float(box[2] * sx),
                    "y2": float(box[3] * sy),
                    "conf": float(conf),
                    "class_id": int(cls_id),
                    "class_name": result.names.get(int(cls_id), str(cls_id)),
                })
        if idx % 100 == 0:
            print(f"{model_name} {width}px: processed {idx}/{len(sample_df)}")
    elapsed = time.perf_counter() - start
    return pd.DataFrame(pred_rows, columns=PREDICTION_COLUMNS), elapsed


def run_scrfd_resolution(sample_df, width, predict_threshold=0.25, ctx_id="auto"):
    add_cuda_dll_directories()
    from insightface.app import FaceAnalysis

    selected_ctx = select_scrfd_ctx(ctx_id)
    inference_size = align_to_stride(width)
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=selected_ctx, det_size=(inference_size, inference_size), det_thresh=predict_threshold)
    pred_rows = []
    start = time.perf_counter()
    for idx, row in enumerate(sample_df.itertuples(index=False), start=1):
        image, scale = load_resized_bgr(row.image_path, width)
        if image is None:
            continue
        sx, sy = scale
        faces = app.get(image)
        for face in faces:
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
            pred_rows.append({
                "model": "SCRFD",
                "resolution_width": width,
                "image_path": row.image_path,
                "x1": x1 * sx,
                "y1": y1 * sy,
                "x2": x2 * sx,
                "y2": y2 * sy,
                "conf": float(face.det_score),
                "class_id": 0,
                "class_name": "face",
            })
        if idx % 100 == 0:
            print(f"SCRFD {width}px: processed {idx}/{len(sample_df)}")
    elapsed = time.perf_counter() - start
    return pd.DataFrame(pred_rows, columns=PREDICTION_COLUMNS), elapsed


def plot_summary(summary, output_dir):
    if "threshold_rank" in summary.columns:
        summary = summary[summary["threshold_rank"] == summary["threshold_rank"].min()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, group in summary.groupby("model"):
        group = group.sort_values("resolution_width")
        axes[0].plot(group["resolution_width"], group["ap50_voc101"], marker="o", label=model)
        axes[1].plot(group["resolution_width"], group["fps"], marker="o", label=model)
    axes[0].set_title("AP50 vs resolution width")
    axes[0].set_xlabel("Frame width after downscale")
    axes[0].set_ylabel("AP50")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("FPS vs resolution width")
    axes[1].set_xlabel("Frame width after downscale")
    axes[1].set_ylabel("FPS")
    axes[1].grid(True, alpha=0.3)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_resolution_ap50_fps.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="chokepoint_data/modeling/chokepoint_p1_indoor_frontal_manifest.csv")
    parser.add_argument("--output-dir", default="chokepoint_data/modeling")
    parser.add_argument("--train-sample", type=int, default=200)
    parser.add_argument("--eval-sample", type=int, default=400)
    parser.add_argument("--sample-mode", choices=["mixed", "eval", "train_dev", "all"], default="mixed")
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--resolutions", default="640,480,384,320,288,256,240,192,160,128")
    parser.add_argument("--thresholds", default="0.25,0.35,0.45,0.55,0.65,0.70")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="YOLO device: auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--scrfd-ctx", default="auto", help="SCRFD ctx_id: auto, -1 for CPU, 0 for GPU.")
    parser.add_argument("--yolo-mode", choices=["pretrained", "finetuned", "both"], default="pretrained")
    parser.add_argument("--weights-dir", default="chokepoint_data/modeling/yolo_finetuned_weights")
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = build_sample(args.manifest, args.train_sample, args.eval_sample, args.seed, args.sample_mode)
    sample_size = len(sample_df)
    resolutions = [int(value) for value in args.resolutions.split(",") if value.strip()]
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    min_predict_conf = min(thresholds + [args.conf_threshold])
    model_specs = build_yolo_model_specs(args.yolo_mode, args.weights_dir)
    selected_yolo_device = select_torch_device(args.device)
    selected_scrfd_ctx = select_scrfd_ctx(args.scrfd_ctx)
    print(f"Selected YOLO device: {selected_yolo_device}")
    print(f"Selected SCRFD ctx_id: {selected_scrfd_ctx}")

    summary_rows = []
    all_prediction_frames = []
    for width in resolutions:
        resolution_height, resolution_label = resolution_metadata(width)
        for model_name, weight_path in model_specs:
            print(f"Running {model_name} at width={width}")
            predictions, elapsed = run_yolo_resolution(
                model_name,
                weight_path,
                sample_df,
                width,
                predict_conf=min_predict_conf,
                device=args.device,
            )
            for threshold_rank, threshold in enumerate(thresholds, start=1):
                metrics = evaluate(sample_df, predictions, threshold, args.iou_threshold)
                summary_rows.append({
                    "model": model_name,
                    "weights": weight_path,
                    "device": selected_yolo_device,
                    "resolution_width": width,
                    "resolution_height": resolution_height,
                    "resolution_label": resolution_label,
                    "threshold_rank": threshold_rank,
                    "sample_size": sample_size,
                    "sample_mode": args.sample_mode,
                    "predictions_raw": len(predictions),
                    "predictions_at_threshold": int((predictions["conf"] >= threshold).sum()) if not predictions.empty else 0,
                    "elapsed_seconds": elapsed,
                    "ms_per_image": elapsed * 1000.0 / sample_size,
                    "fps": sample_size / elapsed if elapsed > 0 else 0.0,
                    "conf_threshold": threshold,
                    "iou_threshold": args.iou_threshold,
                    **metrics,
                })
            all_prediction_frames.append(predictions)

        print(f"Running SCRFD at width={width}")
        predictions, elapsed = run_scrfd_resolution(
            sample_df,
            width,
            predict_threshold=min_predict_conf,
            ctx_id=args.scrfd_ctx,
        )
        for threshold_rank, threshold in enumerate(thresholds, start=1):
            metrics = evaluate(sample_df, predictions, threshold, args.iou_threshold)
            summary_rows.append({
                "model": "SCRFD",
                "weights": "InsightFace buffalo_l / det_10g.onnx",
                "device": f"ctx_id={selected_scrfd_ctx}",
                "resolution_width": width,
                "resolution_height": resolution_height,
                "resolution_label": resolution_label,
                "threshold_rank": threshold_rank,
                "sample_size": sample_size,
                "sample_mode": args.sample_mode,
                "predictions_raw": len(predictions),
                "predictions_at_threshold": int((predictions["conf"] >= threshold).sum()) if not predictions.empty else 0,
                "elapsed_seconds": elapsed,
                "ms_per_image": elapsed * 1000.0 / sample_size,
                "fps": sample_size / elapsed if elapsed > 0 else 0.0,
                "conf_threshold": threshold,
                "iou_threshold": args.iou_threshold,
                **metrics,
            })
        all_prediction_frames.append(predictions)

    summary = pd.DataFrame(summary_rows)
    predictions_all = pd.concat(all_prediction_frames, ignore_index=True) if all_prediction_frames else pd.DataFrame()

    suffix = f"sample{sample_size}"
    base_name = args.output_prefix or "benchmark_resolution_yolov5n_yolov8n_yolov12n_scrfd"
    summary_path = output_dir / f"{base_name}_{suffix}.csv"
    json_path = output_dir / f"{base_name}_{suffix}.json"
    pred_path = output_dir / f"{base_name}_predictions_{suffix}.csv"
    sample_path = (
        output_dir / f"benchmark_resolution_{suffix}_manifest.csv"
        if not args.output_prefix
        else output_dir / f"{base_name}_{suffix}_manifest.csv"
    )

    summary.to_csv(summary_path, index=False)
    predictions_all.to_csv(pred_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    plot_summary(summary, output_dir)

    print("\nResolution benchmark summary:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved:")
    print(summary_path)
    print(json_path)
    print(pred_path)
    print(output_dir / "benchmark_resolution_ap50_fps.png")


if __name__ == "__main__":
    main()
