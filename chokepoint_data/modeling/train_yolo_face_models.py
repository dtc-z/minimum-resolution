from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MODEL_SPECS = {
    "yolov5n": {
        "label": "YOLOv5n-FT",
        "weights": "yolov5nu.pt",
        "output": "yolov5n_face.pt",
    },
    "yolov8n": {
        "label": "YOLOv8n-FT",
        "weights": "yolov8n.pt",
        "output": "yolov8n_face.pt",
    },
    "yolov12n": {
        "label": "YOLOv12n-FT",
        "weights": "paper/yolo12n.pt",
        "output": "yolov12n_face.pt",
    },
}


def select_torch_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def train_one_model(model_key: str, args, selected_device: str) -> dict:
    from ultralytics import YOLO

    spec = MODEL_SPECS[model_key]
    output_dir = Path(args.weights_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_weight = output_dir / spec["output"]
    if final_weight.exists() and not args.force:
        return {
            "model": spec["label"],
            "status": "skipped_existing",
            "weights": str(final_weight),
        }

    model = YOLO(spec["weights"])
    run_name = f"{model_key}_face_e{args.epochs}_img{args.imgsz}"
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=selected_device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / run_name))
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    source_weight = best_weight if best_weight.exists() else last_weight
    if not source_weight.exists():
        raise FileNotFoundError(f"Could not locate trained weights for {model_key}: {save_dir}")
    shutil.copy2(source_weight, final_weight)
    return {
        "model": spec["label"],
        "status": "trained",
        "source_weights": str(source_weight),
        "weights": str(final_weight),
        "save_dir": str(save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="chokepoint_data/modeling/yolo_face_dataset/chokepoint_face.yaml")
    parser.add_argument("--models", default="yolov5n,yolov8n,yolov12n")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="chokepoint_data/modeling/yolo_finetune_runs")
    parser.add_argument("--weights-dir", default="chokepoint_data/modeling/yolo_finetuned_weights")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_device = select_torch_device(args.device)
    print(f"Selected YOLO training device: {selected_device}")
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [key for key in model_keys if key not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Available: {sorted(MODEL_SPECS)}")

    results = []
    for model_key in model_keys:
        print(f"\nFine-tuning {model_key} on ChokePoint face boxes")
        results.append(train_one_model(model_key, args, selected_device))

    summary = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": selected_device,
        "results": results,
    }
    summary_path = Path(args.weights_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
