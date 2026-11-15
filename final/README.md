# Final paper package

This folder contains the rewritten English paper, its editable text source, refreshed figures and reviewable benchmark artifacts.

- `Paper_Rewritten_Final.docx`: current English manuscript filled into the IEEE A4 2024 template, with a balanced two-author block, template placeholders removed, and bordered tables.
- `../results/paper_v2.docx`: updated review copy with the v1 title, nine main sections, and an explicit column break aligning the second author.
- `Paper_Rewritten_Final.md`: text and tables extracted from the current manuscript for review.
- `Paper_Rewritten_Final_VI.md`: Vietnamese translation of the current nine-section manuscript for content review.
- `rebuild_from_2024_template.py`: copies the template's namespace-compatible working copy and inserts manuscript content using its paragraph and table styles.
- `build_rewritten_paper.py`: shortcut to rebuild the template-based manuscript.
- `figures/rewritten/`: current paper figures in PNG and PDF formats.

- `Paper_Improved_Final.docx`: revised English manuscript based on the IEEE A4 conference template.
- `IEEE_Conference_Template_A4_2024.docx`: official A4 Word template linked from the review.
- `Paper_Improved_Final_VI.md`: Vietnamese translation; references retain their original bibliographic form. Vietnamese chart images are under `figures/vi/`.
- `Paper_Review_Checklist.md`: reviewer-response status.
- `benchmark/`: compact benchmark summary, development-locked operating point, speed sample, run configuration and progress record.
- `config/final_protocol.yaml`: immutable pre-test protocol; its SHA-256 matches `benchmark/run_config.json`.
- `annotations/boxes_only_240.csv`: user-designated secondary reference boxes for 240 test frames. Their coordinates exactly match the accompanying InsightFace proposal file, so comparisons against them are not independent human-annotation validation. `annotations/manual_boxes_240.csv` is superseded.
- `annotations/pseudo_boxes_yunet_scrfd_240.csv`: machine-generated interim boxes for the same frames; flagged cases are identified for review and these are not ground truth.
- `annotations/ai_attacker_person_boxes_240.csv`: machine-generated whole-person boxes from YOLO26n for the 240 frames. `attacker` is the user-requested class and is not image-verified; these boxes are not face boxes and are not used by the face-detection paper's metrics.
- `benchmark/manual_box_audit_latest/`: latest per-frame proxy agreement and four-model results computed from `boxes_only_240.csv`.
- `benchmark/manual_box_audit/`: historical audit against the superseded annotation file; do not use its values for the paper.
- `benchmark/pseudo_box_audit/`: exploratory detector agreement against machine-generated pseudo-boxes; not human ground truth.
- `automated_audit/`: separate SCRFD-10GF model proposals for 240 frames. These are not human annotations. Overlay images and contact sheets are kept locally and excluded from the GitHub push because they contain ChokePoint face frames.
- `automated_audit_yunet/`: supplementary YuNet/SCRFD proposal comparison and review flags; face overlays/contact sheets are excluded because they contain ChokePoint frames.
- `generate_vi_figures.py`: recreates the Vietnamese charts from `benchmark/summary.csv`.

The full detector prediction archive and model checkpoints remain in `chokepoint_data/modeling/final_revision/`; they are not copied into this compact paper package. The author block uses two equal columns; author names are bolded while the template font sizes and page margins are retained. Against the 240-frame user-designated reference boxes, proxy IoU is mean 0.625 (median 0.625; IQR 0.593 to 0.654), with 239/240 pairs at IoU at least 0.50. Their coordinates exactly match the accompanying InsightFace proposals, so this secondary comparison is descriptive and is not independent validation of those proposals. The earlier `manual_boxes_240.csv` and its audit outputs are superseded. A separate 240-frame YuNet/SCRFD pseudo-label sensitivity check uses machine-generated references and is not human-ground-truth evaluation. The `attacker` role in the whole-person artifact is a user assumption and is not measured by this face-detection paper; those boxes cannot be substituted for face boxes in the detector evaluation.

Windows held the source DOCX open during the previous packaging step. The DOCX copy in this folder was opened and structurally checked; the original copy remains under `docs/`.
