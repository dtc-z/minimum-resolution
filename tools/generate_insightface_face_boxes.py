from pathlib import Path
import cv2
import pandas as pd
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'final' / 'automated_audit' / 'sample.csv'
OUT = ROOT / 'final' / 'annotations' / 'assistant_insightface_face_boxes_240.csv'

def main():
    sample = pd.read_csv(SAMPLE)
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                       allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    rows, misses = [], []
    for _, row in sample.iterrows():
        image = cv2.imread(str(row.image_path))
        if image is None:
            misses.append((row.dataset, row.frame_number, 'unreadable'))
            continue
        faces = app.get(image)
        if not faces:
            misses.append((row.dataset, row.frame_number, 'no_face'))
            continue
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = [round(float(v), 1) for v in face.bbox]
        h, w = image.shape[:2]
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        rows.append(dict(dataset=row.dataset, frame_number=int(row.frame_number),
            class_label='attacker', class_label_source='user_requested_not_image_verified',
            target_type='face_box', x1=x1, y1=y1, x2=x2, y2=y2,
            label_source='InsightFace_buffalo_l_SCRFD_detection_machine_generated',
            coordinate_basis='800x600_source_frame', detected_faces=len(faces)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'rows={len(rows)} misses={misses[:10]} output={OUT}')

if __name__ == '__main__':
    main()
