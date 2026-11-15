from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record local runtime and accelerator availability.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "packages": {name: package_version(name) for name in
                     ["torch", "ultralytics", "onnxruntime", "onnxruntime-gpu", "insightface", "mmcv", "mmcv-full", "mmdet", "python-docx", "PyMuPDF", "pandas", "matplotlib", "opencv-python", "pi-heif"]},
        "cuda": {"torch_available": False, "torch_version": None, "torch_cuda_version": None, "devices": []},
        "onnxruntime_providers": [],
        "nvidia_smi": None,
    }
    try:
        import torch
        result["cuda"] = {"torch_available": torch.cuda.is_available(), "torch_version": torch.__version__,
                          "torch_cuda_version": torch.version.cuda,
                          "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as error:
        result["cuda"]["error"] = str(error)
    try:
        import onnxruntime
        result["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except Exception as error:
        result["onnxruntime_error"] = str(error)
    command = shutil.which("nvidia-smi")
    if command:
        try:
            run = subprocess.run([command, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15, check=False)
            result["nvidia_smi"] = {"returncode": run.returncode, "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}
        except Exception as error:
            result["nvidia_smi"] = {"error": str(error)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
