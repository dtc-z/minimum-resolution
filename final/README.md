# Final paper package

This folder contains the final English DOCX, Vietnamese Markdown translation and reviewable benchmark artifacts.

- `Paper_Improved_Final.docx` ? revised English manuscript based on the IEEE A4 conference template.
- `IEEE_Conference_Template_A4_2024.docx` ? official A4 Word template linked from the review.
- `Paper_Improved_Final_VI.md` ? Vietnamese translation; references retain their original bibliographic form. Vietnamese chart images are under `figures/vi/`.
- `Paper_Review_Checklist.md` ? reviewer-response status.
- `benchmark/` ? compact benchmark summary, development-locked operating point, speed sample, run configuration and progress record.
- `config/final_protocol.yaml` ? immutable pre-test protocol; its SHA-256 matches `benchmark/run_config.json`.
- `annotations/manual_boxes_240.csv` ? user-drawn test boxes, stored by sequence/frame without local image paths.
- `benchmark/manual_box_audit/` ? per-frame proxy agreement and four-model manual-box results.
- `automated_audit/` ? separate SCRFD-10GF model proposals for 240 frames. These are not human annotations. Overlay images and contact sheets are kept locally and excluded from the GitHub push because they contain ChokePoint face frames.
- `generate_vi_figures.py` ? recreates the Vietnamese charts from `benchmark/summary.csv`.

The full detector prediction archive and model checkpoints remain in `chokepoint_data/modeling/final_revision/`; they are not copied into this compact paper package. The author block in the DOCX still needs the user's edits. User-drawn boxes were evaluated separately from the full landmark-proxy sweep. All 240 manual boxes exactly match the proxy coordinates, so this annotation set does not provide an independent estimate of proxy-label bias. The SCRFD proposal run is a separate automatic detector comparison against the manual reference boxes.

Windows held the source DOCX open during the previous packaging step. The DOCX copy in this folder was opened and structurally checked; the original copy remains under `docs/`.
