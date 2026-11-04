from pathlib import Path
from PIL import Image, UnidentifiedImageError
import csv
import shutil
import sys

# Configuration
DATA_ROOT = Path("d:/Admin/Download/DSP391m/chokepoint_data/extracted")  # adjust if needed
OUTPUT_ROOT = Path("d:/Admin/Download/DSP391m/chokepoint_data_jpg")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tiff", ".tif"}
QUALITY = 90
OVERWRITE = False  # set True to replace existing .jpg in OUTPUT_ROOT
MANIFEST_CSV = OUTPUT_ROOT / "conversion_manifest.csv"

def convert_image(src_path: Path, dest_path: Path):
    try:
        with Image.open(src_path) as im:
            im_rgb = im.convert("RGB")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            im_rgb.save(dest_path, format="JPEG", quality=QUALITY, optimize=True)
            return True, None
    except UnidentifiedImageError as e:
        return False, f"UnidentifiedImageError: {e}"
    except Exception as e:
        return False, str(e)

def main():
    if not DATA_ROOT.exists():
        print(f"DATA_ROOT not found: {DATA_ROOT}")
        return

    out_root = OUTPUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    # gather input images
    imgs = [p for p in DATA_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"Found {len(imgs)} image files under {DATA_ROOT}")

    manifest_rows = []
    failures = 0

    for src in imgs:
        rel = src.relative_to(DATA_ROOT)
        dest = out_root.joinpath(rel).with_suffix(".jpg")
        status = "skipped"
        err = ""

        if dest.exists() and not OVERWRITE:
            status = "exists"
        else:
            ok, err = convert_image(src, dest)
            if ok:
                status = "converted"
            else:
                status = "failed"
                failures += 1

        # collect basic metadata if conversion succeeded
        width = height = size_bytes = ""
        try:
            if dest.exists():
                with Image.open(dest) as im:
                    width, height = im.size
                size_bytes = dest.stat().st_size
        except Exception:
            pass

        manifest_rows.append({
            "src": str(src),
            "dest": str(dest),
            "status": status,
            "error": err,
            "width": width,
            "height": height,
            "size_bytes": size_bytes
        })

    # write manifest
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["src","dest","status","error","width","height","size_bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    converted = sum(1 for r in manifest_rows if r["status"]=="converted")
    skipped = sum(1 for r in manifest_rows if r["status"] in ("exists","skipped"))
    failed = sum(1 for r in manifest_rows if r["status"]=="failed")
    print(f"Converted: {converted}, Skipped/Exists: {skipped}, Failed: {failed}")
    print(f"Manifest written to: {MANIFEST_CSV}")

if __name__ == "__main__":
    main()
