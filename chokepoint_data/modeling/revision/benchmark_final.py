from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


MODEL_SPECS = {
    "yolov5n": ("YOLOv5nu-FT", "chokepoint_data/modeling/final_revision/checkpoints/yolo/yolov5nu_face.pt"),
    "yolov8n": ("YOLOv8n-FT", "chokepoint_data/modeling/final_revision/checkpoints/yolo/yolov8n_face.pt"),
    "yolov12n": ("YOLOv12n-FT", "chokepoint_data/modeling/final_revision/checkpoints/yolo/yolov12n_face.pt"),
    "scrfd": ("SCRFD-500M-FT", None),
}
PREDICTION_COLUMNS = [
    "model", "image_path", "resolution_width", "resolution_height",
    "original_width", "original_height", "x1", "y1", "x2", "y2", "conf",
    "class_id", "class_name",
]
SUMMARY_COLUMNS = [
    "split", "model", "resolution_width", "resolution_height", "conf_threshold",
    "iou_threshold", "ap50_vs_derived_boxes", "precision", "recall", "f1",
    "tp", "fp", "fn", "gt_count", "predictions_at_threshold", "fps_median",
    "fps_q1", "fps_q3", "latency_ms_median",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_gpu() -> None:
    torch.cuda.synchronize(0)


class YoloAdapter:
    def __init__(self, label: str, weights: Path, device: str, conf_floor: float, nms_iou: float):
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(40)  # suppress repeated per-image stride adjustment warnings
        self.label = label
        self.weights = weights
        self.model = YOLO(str(weights))
        self.device = device
        self.conf_floor = conf_floor
        self.nms_iou = nms_iou

    def predict(self, image: np.ndarray, width: int, height: int):
        results = self.model.predict(
            source=image,
            imgsz=(height, width),
            conf=self.conf_floor,
            iou=self.nms_iou,
            device=self.device,
            max_det=300,
            half=False,
            verbose=False,
            save=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        return [
            (box.astype(float).tolist(), float(score), int(class_id), str(results[0].names.get(int(class_id), class_id)))
            for box, score, class_id in zip(xyxy, scores, classes)
        ]


class ScrfdAdapter:
    def __init__(self, label: str, config: Path, weights: Path, device: str, conf_floor: float, nms_iou: float):
        from mmdet.apis import init_detector

        self.label = label
        self.weights = weights
        self.model = init_detector(str(config), str(weights), device=device)
        self.conf_floor = conf_floor
        self.nms_iou = nms_iou
        for test_cfg in (self.model.test_cfg, self.model.bbox_head.test_cfg):
            test_cfg.score_thr = conf_floor
            test_cfg.nms.iou_threshold = nms_iou
            test_cfg.max_per_img = 300

    def set_resolution(self, width: int, height: int) -> None:
        pipeline = copy.deepcopy(self.model.cfg.data.test.pipeline)
        multi_scale = next(item for item in pipeline if item["type"] == "MultiScaleFlipAug")
        multi_scale.img_scale = (width, height)
        for transform in multi_scale.transforms:
            if transform["type"] == "Pad":
                transform.pop("size", None)
                transform["size_divisor"] = 32
        self.model.cfg.data.test.pipeline = pipeline

    def predict(self, image: np.ndarray, width: int, height: int):
        from mmdet.apis import inference_detector

        self.set_resolution(width, height)
        results = inference_detector(self.model, image)
        if not results or len(results[0]) == 0:
            return []
        return [
            ([float(value) for value in det[:4]], float(det[4]), 0, "face")
            for det in results[0]
            if float(det[4]) >= self.conf_floor
        ]


def load_manifest(path: Path, expected_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_path", "dataset", "final_split", "x1", "y1", "x2", "y2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if frame["image_path"].duplicated().any():
        raise ValueError(f"{path} contains duplicate image_path values")
    if not frame["final_split"].eq(expected_split).all():
        raise ValueError(f"{path} contains rows outside final_split={expected_split}")
    if frame.empty:
        raise ValueError(f"{path} contains no instances")
    return frame.reset_index(drop=True)


def image_predictions(adapter, frame: pd.DataFrame, width: int, progress_label: str):
    height = round(width * 600 / 800)
    predictions = []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read source image: {row.image_path}")
        original_height, original_width = image.shape[:2]
        for box, score, class_id, class_name in adapter.predict(image, width, height):
            predictions.append({
                "model": adapter.label,
                "image_path": row.image_path,
                "resolution_width": width,
                "resolution_height": height,
                "original_width": original_width,
                "original_height": original_height,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "conf": score, "class_id": class_id, "class_name": class_name,
            })
        if index % 250 == 0 or index == len(frame):
            print(f"{progress_label}: {index}/{len(frame)}", flush=True)
    return predictions


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate(gt_frame: pd.DataFrame, predictions: list[dict], thresholds: list[float], iou_threshold: float):
    ground_truths = defaultdict(list)
    for row in gt_frame.itertuples(index=False):
        ground_truths[row.image_path].append([float(row.x1), float(row.y1), float(row.x2), float(row.y2)])
    pred_by_image = defaultdict(list)
    for item in predictions:
        pred_by_image[item["image_path"]].append(item)

    all_preds = sorted(predictions, key=lambda item: item["conf"], reverse=True)
    matched = defaultdict(set)
    ap_flags = []
    for item in all_preds:
        boxes = ground_truths.get(item["image_path"], [])
        candidates = [
            (box_iou([item["x1"], item["y1"], item["x2"], item["y2"]], box), index)
            for index, box in enumerate(boxes) if index not in matched[item["image_path"]]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        is_true_positive = best_index >= 0 and best_iou >= iou_threshold
        if is_true_positive:
            matched[item["image_path"]].add(best_index)
        ap_flags.append(1 if is_true_positive else 0)

    total_gt = sum(len(boxes) for boxes in ground_truths.values())
    cumulative_tp = np.cumsum(ap_flags, dtype=np.float64)
    cumulative_fp = np.cumsum([1 - value for value in ap_flags], dtype=np.float64)
    if len(cumulative_tp):
        recalls = cumulative_tp / total_gt
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
        ap50 = float(np.mean([
            max(precisions[recalls >= target], default=0.0)
            for target in np.linspace(0.0, 1.0, 101)
        ]))
    else:
        ap50 = 0.0

    rows = []
    for threshold in thresholds:
        tp = fp = 0
        total_predictions = 0
        for image_path, boxes in ground_truths.items():
            kept = sorted(
                (item for item in pred_by_image.get(image_path, []) if item["conf"] >= threshold),
                key=lambda item: item["conf"], reverse=True,
            )
            total_predictions += len(kept)
            image_matched = set()
            for item in kept:
                candidates = [
                    (box_iou([item["x1"], item["y1"], item["x2"], item["y2"]], box), index)
                    for index, box in enumerate(boxes) if index not in image_matched
                ]
                best_iou, best_index = max(candidates, default=(0.0, -1))
                if best_index >= 0 and best_iou >= iou_threshold:
                    image_matched.add(best_index)
                    tp += 1
                else:
                    fp += 1
        fn = total_gt - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "conf_threshold": threshold, "ap50_vs_derived_boxes": ap50,
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "gt_count": total_gt,
            "predictions_at_threshold": total_predictions,
        })
    return rows


def make_speed_sample(test_frame: pd.DataFrame, per_sequence: int, seed: int) -> pd.DataFrame:
    pieces = []
    for index, (sequence, group) in enumerate(test_frame.groupby("dataset", sort=True)):
        if len(group) < per_sequence:
            raise ValueError(f"Sequence {sequence} has {len(group)} frames, fewer than requested {per_sequence}")
        pieces.append(group.sample(n=per_sequence, random_state=seed + index))
    sample = pd.concat(pieces).sort_values(["dataset", "image_path"]).reset_index(drop=True)
    return sample


def measure_speed(adapter, sample: pd.DataFrame, width: int, warmup: int, repeats: int):
    height = round(width * 600 / 800)
    if sample.empty:
        raise ValueError("Speed sample is empty")
    for index in range(warmup):
        row = sample.iloc[index % len(sample)]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read speed-sample image: {row.image_path}")
        adapter.predict(image, width, height)
    sync_gpu()
    fps_values = []
    for repeat in range(repeats):
        sync_gpu()
        start = time.perf_counter()
        for row in sample.itertuples(index=False):
            image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Could not read speed-sample image: {row.image_path}")
            adapter.predict(image, width, height)
        sync_gpu()
        elapsed = time.perf_counter() - start
        fps_values.append(len(sample) / elapsed)
        print(f"speed {adapter.label} {width}px repeat {repeat + 1}/{repeats}: {fps_values[-1]:.3f} FPS", flush=True)
    q1, median, q3 = np.percentile(fps_values, [25, 50, 75]).tolist()
    return {
        "fps_median": float(median), "fps_q1": float(q1), "fps_q3": float(q3),
        "latency_ms_median": 1000.0 / float(median) if median else 0.0,
        "fps_repeats": fps_values,
    }


def resolve_scrfd_weights(explicit: str | None, root: Path) -> Path:
    if explicit:
        return Path(explicit).resolve()
    checkpoint_dir = root / "chokepoint_data" / "modeling" / "final_revision" / "checkpoints" / "scrfd"
    run_record = root / "chokepoint_data" / "modeling" / "final_revision" / "logs" / "scrfd_training_run.json"
    if run_record.is_file():
        record = json.loads(run_record.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and record.get("best_checkpoint"):
            selected = Path(record["best_checkpoint"]).resolve()
            if not selected.is_file():
                raise FileNotFoundError(f"Recorded SCRFD best checkpoint is missing: {selected}")
            expected_sha = record.get("best_checkpoint_sha256")
            if expected_sha and sha256_file(selected) != expected_sha:
                raise ValueError(f"Recorded SCRFD best checkpoint hash mismatch: {selected}")
            return selected
    best_pointer = checkpoint_dir / "best_mAP.pth"
    if best_pointer.is_file():
        return best_pointer
    candidates = sorted(checkpoint_dir.glob("best_mAP_epoch_*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileExistsError(f"Multiple SCRFD best checkpoints found; select one explicitly: {candidates}")
    latest = checkpoint_dir / "latest.pth"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"No fine-tuned SCRFD checkpoint found in {checkpoint_dir}")


def build_adapter(model_key: str, root: Path, device: str, conf_floor: float, nms_iou: float,
                  scrfd_config: Path, scrfd_weights: Path):
    label, relative_weights = MODEL_SPECS[model_key]
    if model_key == "scrfd":
        weights = scrfd_weights
        return ScrfdAdapter(label, scrfd_config, weights, device, conf_floor, nms_iou), weights
    weights = (root / relative_weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing fine-tuned weights: {weights}")
    return YoloAdapter(label, weights, device, conf_floor, nms_iou), weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-locked, final test benchmark for fine-tuned YOLO and SCRFD.")
    parser.add_argument("--dev-manifest", default="chokepoint_data/modeling/final_revision/manifests/dev.csv")
    parser.add_argument("--test-manifest", default="chokepoint_data/modeling/final_revision/manifests/test.csv")
    parser.add_argument("--protocol", default="chokepoint_data/modeling/final_revision/config/final_protocol.yaml")
    parser.add_argument("--out-dir", default="chokepoint_data/modeling/final_revision/benchmark")
    parser.add_argument("--scrfd-config", default="chokepoint_data/modeling/final_revision/config/scrfd_500m_p1.py")
    parser.add_argument("--scrfd-weights", default=None)
    parser.add_argument("--models", default="yolov5n,yolov8n,yolov12n,scrfd")
    parser.add_argument("--resolutions", default="640,480,384,320,288,256,240,192,160,128")
    parser.add_argument("--thresholds", default="0.25,0.35,0.45,0.55,0.65,0.70")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--selection-confidence", type=float, default=0.25)
    parser.add_argument("--min-ap50", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--speed-sample-per-sequence", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dev-only", action="store_true",
                        help="run the development sweep and provisional selection without loading test labels or images")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the locked benchmark protocol")
    if args.device not in {"cuda:0", "0"}:
        raise ValueError("This protocol pins inference to CUDA device 0")
    device = "cuda:0"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = Path.cwd()
    dev_frame = load_manifest(Path(args.dev_manifest), "dev")
    widths = [int(value) for value in args.resolutions.split(",")]
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if args.selection_confidence not in thresholds:
        thresholds.append(args.selection_confidence)
        thresholds.sort()
    model_keys = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(model_keys) - set(MODEL_SPECS))
    if unknown or len(model_keys) != len(set(model_keys)):
        raise ValueError(f"Unknown or repeated model keys: {unknown or model_keys}")
    scrfd_config = Path(args.scrfd_config).resolve()
    scrfd_weights = resolve_scrfd_weights(args.scrfd_weights, root) if "scrfd" in model_keys else Path()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": sha256_file(Path(args.protocol)),
        "dev_manifest": str(Path(args.dev_manifest).resolve()),
        "dev_manifest_sha256": sha256_file(Path(args.dev_manifest)),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "mmcv": importlib.metadata.version("mmcv-full"),
        "mmdet": importlib.metadata.version("mmdet"),
        "opencv": cv2.__version__,
        "models": {},
        "dev_only": bool(args.dev_only),
        "resolutions_width": widths,
        "thresholds": thresholds,
        "confidence_floor": args.conf_floor,
        "nms_iou": args.iou,
        "metric_iou": args.iou,
        "selection": {
            "confidence": args.selection_confidence,
            "minimum_ap50": args.min_ap50,
            "minimum_recall": args.min_recall,
            "rule": "Among candidates satisfying both dev thresholds, choose the smallest input width; ties rank by higher dev AP50, then higher dev recall, then fixed model-key order. No eligible candidate yields no recommendation.",
        },
        "speed": {
            "sample_seed": args.seed,
            "frames_per_test_sequence": args.speed_sample_per_sequence,
            "sample_count": None,
            "warmup_iterations": args.warmup,
            "repeats": args.repeats,
            "timed_region": "OpenCV image read, framework preprocessing/resize, detector inference, and model postprocessing; model load and metric calculation excluded",
        },
        "notes": [
            "Accuracy labels are eye-landmark-derived proxy boxes, not manually annotated face rectangles.",
            "The test manifest is not read until dev candidate metrics and the operating point file have been written.",
            "Speed sample comes from test frames but uses no ground-truth labels and does not select a model or threshold.",
            "Framework-native resize/letterbox is used; YOLO and SCRFD retain model-specific native transforms, with a common target maximum size and common NMS IoU threshold.",
        ],
    }
    if "scrfd" in model_keys:
        run_config["scrfd_config"] = str(scrfd_config)
        run_config["scrfd_config_sha256"] = sha256_file(scrfd_config)
    config_path = out_dir / "run_config.json"

    # The test manifest is validated but no test image or label is read until this dev-only phase ends.
    dev_rows = []
    for model_key in model_keys:
        adapter, weights = build_adapter(model_key, root, device, args.conf_floor, args.iou, scrfd_config, scrfd_weights)
        run_config["models"][adapter.label] = {"weights": str(weights.resolve()), "weights_sha256": sha256_file(weights)}
        for width in widths:
            preds = image_predictions(adapter, dev_frame, width, f"dev {adapter.label} {width}px")
            for metric in evaluate(dev_frame, preds, thresholds, args.iou):
                dev_rows.append({
                    "split": "dev", "model": adapter.label,
                    "resolution_width": width, "resolution_height": round(width * 600 / 800),
                    "iou_threshold": args.iou,
                    "fps_median": None, "fps_q1": None, "fps_q3": None,
                    "latency_ms_median": None, **metric,
                })
        del adapter
        torch.cuda.empty_cache()
    dev_summary = pd.DataFrame(dev_rows, columns=SUMMARY_COLUMNS)
    dev_summary.to_csv(out_dir / "dev_summary.csv", index=False)
    eligible = dev_summary[
        (dev_summary["conf_threshold"] == args.selection_confidence)
        & (dev_summary["ap50_vs_derived_boxes"] >= args.min_ap50)
        & (dev_summary["recall"] >= args.min_recall)
    ].copy()
    if eligible.empty:
        selection = {
            "status": "no_candidate_meets_dev_constraints",
            "confidence": args.selection_confidence,
            "minimum_ap50": args.min_ap50,
            "minimum_recall": args.min_recall,
            "selected_model": None,
            "selected_width": None,
        }
    else:
        key_order = {key: index for index, key in enumerate(model_keys)}
        label_to_key = {MODEL_SPECS[key][0]: key for key in model_keys}
        eligible["_key_order"] = eligible["model"].map(lambda name: key_order[label_to_key[name]])
        chosen = eligible.sort_values(
            ["resolution_width", "ap50_vs_derived_boxes", "recall", "_key_order"],
            ascending=[True, False, False, True], kind="mergesort",
        ).iloc[0]
        selection = {
            "status": "selected_from_dev_before_test",
            "selected_model": str(chosen["model"]),
            "selected_width": int(chosen["resolution_width"]),
            "selected_height": int(chosen["resolution_height"]),
            "confidence": float(args.selection_confidence),
            "dev_ap50_vs_derived_boxes": float(chosen["ap50_vs_derived_boxes"]),
            "dev_precision": float(chosen["precision"]),
            "dev_recall": float(chosen["recall"]),
            "dev_f1": float(chosen["f1"]),
            "minimum_ap50": args.min_ap50,
            "minimum_recall": args.min_recall,
        }
    selection["locked_utc"] = datetime.now(timezone.utc).isoformat()
    if args.dev_only:
        selection["status"] = "provisional_dev_only"
        selection["provisional"] = True
    (out_dir / "operating_point_dev_locked.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print("Dev operating point locked:", json.dumps(selection), flush=True)
    if args.dev_only:
        print("Dev-only run complete; no test labels or images were loaded.", flush=True)
        return

    # Only now load test labels and images for the final benchmark.
    test_frame = load_manifest(Path(args.test_manifest), "test")
    run_config["test_manifest_sha256"] = sha256_file(Path(args.test_manifest))
    speed_sample = make_speed_sample(test_frame, args.speed_sample_per_sequence, args.seed)
    speed_sample[["dataset", "image_path", "final_split"]].to_csv(out_dir / "speed_sample.csv", index=False)
    run_config["speed"]["sample_count"] = int(len(speed_sample))
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    test_rows = []
    prediction_row_count = 0
    prediction_path = out_dir / "predictions.csv.gz"
    with gzip.open(prediction_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for model_key in model_keys:
            adapter, weights = build_adapter(model_key, root, device, args.conf_floor, args.iou, scrfd_config, scrfd_weights)
            speed_by_width = {}
            for width in widths:
                preds = image_predictions(adapter, test_frame, width, f"test {adapter.label} {width}px")
                for item in preds:
                    writer.writerow(item)
                prediction_row_count += len(preds)
                metrics = evaluate(test_frame, preds, thresholds, args.iou)
                speed = measure_speed(adapter, speed_sample, width, args.warmup, args.repeats)
                speed_by_width[width] = speed
                for metric in metrics:
                    test_rows.append({
                        "split": "test", "model": adapter.label,
                        "resolution_width": width, "resolution_height": round(width * 600 / 800),
                        "iou_threshold": args.iou,
                        "fps_median": speed["fps_median"], "fps_q1": speed["fps_q1"],
                        "fps_q3": speed["fps_q3"], "latency_ms_median": speed["latency_ms_median"],
                        **metric,
                    })
            del adapter
            torch.cuda.empty_cache()
    summary = pd.DataFrame(test_rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "summary.csv", index=False)
    run_config["finished_utc"] = datetime.now(timezone.utc).isoformat()
    run_config["test_image_count"] = int(len(test_frame))
    run_config["test_prediction_rows"] = prediction_row_count
    run_config["test_summary_sha256"] = sha256_file(out_dir / "summary.csv")
    run_config["predictions_sha256"] = sha256_file(prediction_path)
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Final test summary: {out_dir / 'summary.csv'}", flush=True)
    print(f"Compressed predictions: {prediction_path}", flush=True)


if __name__ == "__main__":
    main()
