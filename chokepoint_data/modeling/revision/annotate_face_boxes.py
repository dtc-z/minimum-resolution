from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV rectangle annotator for the locked manual audit sample.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--annotator", required=True, choices=["A", "B"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sample = pd.read_csv(args.sample).fillna("")
    output = Path(args.out)
    if output.exists():
        annotations = pd.read_csv(output).fillna("")
    else:
        annotations = pd.DataFrame(columns=["dataset", "frame_number", "annotator", "human_occlusion", "human_x1", "human_y1", "human_x2", "human_y2"])
    saved = {(str(r.dataset), str(r.frame_number), str(r.annotator)) for r in annotations.itertuples(index=False)}
    state = {"start": None, "end": None, "dragging": False}
    current = {"index": None, "image": None}

    def mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=(x, y), end=(x, y), dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state.update(end=(x, y), dragging=False)

    title = "Manual face rectangle | drag box; 0-3 occlusion; Enter save; n skip; Esc quit"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    for i, row in sample.iterrows():
        key = (str(row.dataset), str(row.frame_number), args.annotator)
        if key in saved or (args.annotator == "B" and not bool(row.double_annotate)):
            continue
        image = cv2.imread(str(row.image_path))
        if image is None:
            raise OSError(f"Unable to read image: {row.image_path}")
        state.update(start=None, end=None, dragging=False)
        current.update(index=i, image=image)
        occlusion = None
        while True:
            shown = image.copy()
            ih, iw = shown.shape[:2]
            x1, y1, x2, y2 = [int(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
            cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 165, 255), 2)
            if state["start"] and state["end"]:
                cv2.rectangle(shown, state["start"], state["end"], (0, 255, 0), 2)
            cv2.putText(shown, f"{row.dataset} frame={row.frame_number} annotator={args.annotator}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.putText(shown, "Occlusion 0 none | 1 mild | 2 moderate | 3 severe", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
            cv2.imshow(title, shown)
            keycode = cv2.waitKey(30) & 0xFF
            if keycode in (ord("0"), ord("1"), ord("2"), ord("3")):
                occlusion = int(chr(keycode))
            elif keycode in (10, 13):
                if not state["start"] or not state["end"] or occlusion is None:
                    continue
                ax, ay = state["start"]
                bx, by = state["end"]
                record = {"dataset": row.dataset, "frame_number": row.frame_number, "annotator": args.annotator,
                          "human_occlusion": occlusion, "human_x1": max(0, min(ax, bx)),
                          "human_y1": max(0, min(ay, by)), "human_x2": min(iw - 1, max(ax, bx)),
                          "human_y2": min(ih - 1, max(ay, by))}
                annotations = pd.concat([annotations, pd.DataFrame([record])], ignore_index=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                annotations.to_csv(output, index=False)
                saved.add(key)
                break
            elif keycode == ord("n"):
                break
            elif keycode == 27:
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
