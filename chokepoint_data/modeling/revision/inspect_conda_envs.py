from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import sys


names = ["torch", "torchvision", "mmcv", "mmdet", "python-docx", "PyMuPDF", "ultralytics"]
print(sys.executable, sys.version.split()[0])
versions = {}
for name in names:
    module = name.replace("-", "_").lower()
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    try:
        versions[name] = metadata.version(name) if found else None
    except metadata.PackageNotFoundError:
        versions[name] = "module-present-distribution-unknown" if found else None
print(versions)
if importlib.util.find_spec("torch"):
    import torch
    print("cuda:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
