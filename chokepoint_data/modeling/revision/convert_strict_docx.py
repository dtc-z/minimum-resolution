from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NAMESPACE_MAP = {
    "http://purl.oclc.org/ooxml/wordprocessingml/main":
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/drawingml/main":
        "http://schemas.openxmlformats.org/drawingml/2006/main",
    "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing":
        "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "http://purl.oclc.org/ooxml/officeDocument/customXml":
        "http://schemas.openxmlformats.org/officeDocument/2006/customXml",
    "http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes":
        "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "http://purl.oclc.org/ooxml/officeDocument/extendedProperties":
        "http://schemas.openxmlformats.org/officeDocument/2006/extendedProperties",
    "http://purl.oclc.org/ooxml/officeDocument/math":
        "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "http://purl.oclc.org/ooxml/officeDocument/relationships":
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Strict OOXML namespace URIs to Transitional for python-docx compatibility.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as source_zip, ZipFile(out, "w", compression=ZIP_DEFLATED) as result_zip:
        for item in source_zip.infolist():
            data = source_zip.read(item.filename)
            if item.filename.endswith((".xml", ".rels")):
                for strict, transitional in NAMESPACE_MAP.items():
                    data = data.replace(strict.encode(), transitional.encode())
            result_zip.writestr(item, data)
    print({"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "converted": str(out.resolve()), "converted_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
