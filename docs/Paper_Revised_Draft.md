# Resolution-Accuracy Trade-off for Lightweight Face Detection in ChokePoint Surveillance Cameras

Vu Hai Minh  
DSP391M Data Science Capstone Project  
FPT University  

## Abstract

Face detection in surveillance video differs from generic web-image face detection because the operating scene is constrained, repetitive, and affected by camera placement, motion blur, pose drift, and illumination changes. This paper draft focuses on the relationship between input resolution and face-detection accuracy in ChokePoint surveillance cameras. ChokePoint provides multi-camera video sequences collected at pedestrian portals, together with manually labelled eye locations and cropped 96x96 face images. Because its ground truth is provided as eye landmarks rather than detector bounding boxes, we build a transparent landmark-to-box conversion protocol and use it to prepare a detector-oriented evaluation pipeline. A local audit of the extracted dataset found 181,657 full-frame JPG images, 63,570 cropped PGM face images, 72 XML ground-truth files, and 72,289 eye-pair annotations that can be mapped to full-frame images after accounting for Portal 2 directory naming. The first modeling protocol uses Portal 1 indoor frontal sequences, following the official ChokePoint case-study split. A preliminary off-the-shelf benchmark compares YOLOv5n, YOLOv8n, YOLOv12n, and SCRFD under controlled downsampling. The COCO-pretrained YOLO nano models obtain AP50 of 0.0000 against face boxes because they are not face-specialized detectors, while SCRFD remains accurate as resolution is reduced: on a 120-frame preliminary sweep, SCRFD keeps AP50 above 0.96 down to 256 px frame width and drops to AP50 0.9372 at 160 px. These results suggest an initial operating cut-off around 256-320 px width for Portal 1 indoor frontal footage, pending larger full-split evaluation. The main contribution of this draft is a ChokePoint-only, reproducible protocol for finding the resolution-accuracy trade-off of lightweight face detection in surveillance choke-point footage.

**Keywords:** face detection, surveillance video, ChokePoint dataset, SCRFD, YOLO, eye landmarks, lightweight detection

## 1. Introduction

Face detection systems used in surveillance settings must handle a different set of constraints from detectors evaluated on broad web-image benchmarks. In a choke-point camera setup, the background is relatively stable, the camera is fixed, and people pass through a narrow physical region. At the same time, the face itself can be small, blurred, partially rotated, or captured from a high angle. These properties make the problem especially relevant for edge deployment, where inference speed, predictable error modes, and robustness to small faces matter more than visual diversity alone.

This work uses the ChokePoint dataset as the only dataset for the current research direction. The single-dataset scope is intentional because the project objective is a narrower surveillance-camera setting. ChokePoint matches that setting because it was recorded using camera arrays above real pedestrian portals and includes repeated sequences of subjects entering and leaving the scene.

The paper draft addresses four practical questions:

1. How should ChokePoint eye-landmark labels be converted into detector-ready face boxes?
2. Which subset of ChokePoint should be used first for a clean indoor surveillance benchmark?
3. Can an off-the-shelf lightweight face detector produce a credible initial baseline on ChokePoint full-frame images?
4. How far can camera/frame resolution be reduced before face-detection accuracy becomes unacceptable?

The current answer is methodological rather than fully final. The report and notebook now parse the ChokePoint XML files, map annotations to full-frame JPG images, derive bounding boxes from the eye landmarks, export a COCO-style manifest for the Portal 1 indoor frontal protocol, and provide reproducible cells for running fixed-resolution and multi-resolution benchmarks on ChokePoint only.

## 2. Dataset

### 2.1 ChokePoint Overview

The ChokePoint dataset was designed for person identification and verification under real-world surveillance conditions. According to the official dataset page and Zenodo record, the data was collected using an array of three cameras placed above pedestrian portals, or natural choke points, to capture subjects walking through the portals. The dataset includes Portal 1 and Portal 2 recordings, with Portal 1 and Portal 2 collected one month apart. The official description reports 30 fps video, 800x600 frame resolution, 48 video sequences, and 64,204 cropped face images. It also states that the first 100 frames of each sequence are intended for background modeling.

The sequence naming convention is central to the protocol. A sequence such as `P1E_S1_C1` indicates Portal 1, Entering direction, Sequence 1, Camera 1. Similarly, `P1L` denotes leaving Portal 1, and `P2E`/`P2L` denote the corresponding Portal 2 conditions.

For this paper, the practical distinction between full-frame camera images and cropped face images is important:

- Full-frame camera images are `.jpg` files at 800x600 resolution and should be used for face-detector evaluation.
- Cropped face images are `.pgm` files at 96x96 resolution. They are useful for recognition, face quality, or super-resolution experiments, but they are not the correct input for evaluating whether a detector can find a face in the surveillance frame.
- Ground truth is stored in XML files as manually labelled eye locations, not as native detector bounding boxes.

### 2.2 Local Dataset Audit

The local extracted ChokePoint folder was audited before modeling. The current local counts are:

| Item | Count |
|---|---:|
| Full-frame JPG images | 181,657 |
| Cropped PGM face images | 63,570 |
| XML ground-truth files | 72 |
| Parsed eye-pair annotations | 72,289 |
| Parsed frames in XML files | 176,916 |
| XML frames with at least one person annotation | 72,291 |
| Unique subject IDs observed in XML | 29 |
| Eye-pair annotations mapped to JPG images | 72,289 |

The XML-to-image mapping required one implementation detail. Portal 1 sequences are usually stored under a direct child folder such as `P1E_S1_C1/P1E_S1_C1`. Portal 2 sequences in the local extraction can appear under suffixed folders such as `P2E_S1_C3.1` and `P2E_S1_C3.2`. A rigid path rule therefore misses a large part of the dataset. The corrected parser searches both the direct sequence folder and suffixed alternatives, which maps all parsed annotations to image files.

The eye-distance distribution also gives a useful proxy for face scale:

| Eye-distance statistic | Pixels |
|---|---:|
| Minimum | 11.00 |
| 25th percentile | 20.02 |
| Median | 27.00 |
| 75th percentile | 36.88 |
| Maximum | 119.85 |

This distribution confirms that many ChokePoint faces are small in the original 800x600 frame, which supports the choice of lightweight face detectors and scale-aware error analysis.

## 3. ChokePoint-Only Protocol

### 3.1 Initial Case Study: Portal 1 Indoor Frontal

The first modeling protocol uses the official ChokePoint indoor case study for Portal 1. This decision keeps the initial benchmark focused and reduces confounding variation from outdoor lighting, crowding, and long time intervals. The selected sequences are:

| Split | Sequences |
|---|---|
| Train/development | `P1E_S1_C1`, `P1E_S2_C2`, `P1L_S1_C1`, `P1L_S2_C2` |
| Evaluation | `P1E_S3_C3`, `P1E_S4_C1`, `P1L_S3_C3`, `P1L_S4_C1` |

The local annotation counts for this protocol are:

| Split | Annotations |
|---|---:|
| Train/development | 6,035 |
| Evaluation | 5,676 |
| Total | 11,711 |

This split is suitable for Report 3 because it provides enough frames for a meaningful detector pipeline while preserving an official ChokePoint evaluation structure. Portal 2 and crowded sequences should be used later as robustness tests.

### 3.2 Why Full Frames Matter

The user-facing inspection issue in the project came from opening `.pgm` crop files. Those images are grayscale cropped faces and do not look like camera frames. They are not the correct evidence for a detection paper unless the task is recognition or quality scoring on already-localized faces. For detector modeling, the correct input is the full-frame JPG sequence data. This paper therefore treats `.pgm` crops as secondary assets and bases detection evaluation on full camera frames.

## 4. Label Derivation

### 4.1 Eye Landmarks to Bounding Boxes

ChokePoint provides eye coordinates, not face bounding boxes. Let the left eye be `(lx, ly)` and the right eye be `(rx, ry)`. The inter-eye distance is:

`d = sqrt((lx - rx)^2 + (ly - ry)^2)`

The derived face box is then defined using the eye midpoint `(cx, cy)`:

`cx = (lx + rx) / 2`  
`cy = (ly + ry) / 2`

The current report uses:

- `box_width = 2.8 * d`
- `box_height = 3.6 * d`
- `top = cy - 1.2 * d`
- `bottom = cy + (3.6 - 1.2) * d`
- left/right are centered around `cx`

Coordinates are clipped to the 800x600 image boundary.

This conversion is intentionally simple and reproducible. It should be described in any report or paper as a derived annotation procedure. The derived box is not an original ChokePoint label. The box scale can be tuned in later work by visually validating sampled frames or by calibrating against a small manually boxed subset.

### 4.2 Evaluation Consequence

Because the benchmark box is derived from eye landmarks, IoU-based metrics measure agreement with the heuristic box, not agreement with a manually drawn face rectangle. This has two consequences:

1. AP50 is more stable than stricter AP75/AP50-95 for the first report because small changes in derived box size can shift IoU.
2. A detector can correctly localize the visible face but receive a lower IoU if its learned face box convention differs from the eye-derived box convention.

For Report 3, AP50, precision, recall, F1, and mean matched IoU are adequate. Report 4 should include a sensitivity analysis over box-scale parameters.

## 5. Modeling Approach

### 5.1 Candidate Detectors

The detector families for the ChokePoint-only direction are:

- **SCRFD via InsightFace.** SCRFD is designed for efficient face detection and is available locally through the `buffalo_l` model package as `det_10g.onnx`. It provides an immediate off-the-shelf baseline without fine-tuning.
- **Lightweight YOLO.** YOLO can be fine-tuned on the derived ChokePoint boxes after exporting labels. Any YOLO experiment in the next phase should be trained or fine-tuned on ChokePoint-derived annotations.
- **Classical baseline, optional.** OpenCV Haar cascade can be used as a weak classical baseline if the report needs a historical comparison, but it should not be the main method.

### 5.2 Evaluation Metrics

The main detector metrics are:

- **Precision:** proportion of predicted face boxes that match a ground-truth box.
- **Recall:** proportion of ground-truth faces that are detected.
- **F1:** harmonic mean of precision and recall.
- **AP50:** average precision using IoU threshold 0.5.
- **Latency:** average inference time per frame.
- **Mean matched IoU:** localization quality for true positive matches.

For ChokePoint, metrics should be reported by sequence, portal, direction, camera, and face-size bucket. The face-size bucket can be based on inter-eye distance or derived box area.

## 6. Preliminary YOLO/SCRFD Benchmark

### 6.1 Experimental Setup

A preliminary benchmark was run locally on a deterministic 600-frame sample from the Portal 1 indoor frontal protocol:

- 200 samples from the train/development split
- 400 samples from the evaluation split
- Detectors: YOLOv5n (`yolov5nu.pt`), YOLOv8n (`yolov8n.pt`), YOLOv12n (`paper/yolo12n.pt`), and InsightFace SCRFD `det_10g.onnx`
- Provider: CPUExecutionProvider
- Detector input size: 640x640
- Confidence threshold: 0.25
- IoU threshold: 0.5
- Ground truth: derived boxes from ChokePoint eye landmarks

The goal of this experiment is to validate the end-to-end pipeline and compare the requested model families. The YOLO models in this run are COCO-pretrained object detectors, not face-fine-tuned ChokePoint models, so their face-box AP should be interpreted as an off-the-shelf baseline rather than a final YOLO result.

### 6.2 Preliminary Results

| Model | Weights | AP50 | Precision | Recall | F1 | ms/image | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| SCRFD | InsightFace `buffalo_l` / `det_10g.onnx` | 0.9889 | 0.9819 | 0.9967 | 0.9892 | 69.39 | 14.41 |
| YOLOv5n | `yolov5nu.pt` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 48.68 | 20.54 |
| YOLOv8n | `yolov8n.pt` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 50.92 | 19.64 |
| YOLOv12n | `paper/yolo12n.pt` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 58.47 | 17.10 |

The YOLO nano models are faster than SCRFD on CPU in this off-the-shelf run, but their AP50 is zero because COCO-pretrained YOLO detectors do not predict face boxes. SCRFD is slower but correctly targets faces and therefore provides the usable face-detection baseline. The lower mean matched IoU for SCRFD compared with its high AP50 is expected because the derived box convention may not exactly match SCRFD's predicted face-box convention.

### 6.3 Interpretation

The preliminary benchmark supports three conclusions:

1. The XML parser, frame mapper, and derived-box evaluator are functioning correctly.
2. Portal 1 indoor frontal sequences are a reasonable first benchmark because they produce stable detector behavior.
3. YOLOv5n, YOLOv8n, and YOLOv12n need ChokePoint face fine-tuning before they can be compared fairly with SCRFD on face-detection accuracy.
4. Full-dataset conclusions require additional runs on all evaluation frames, threshold sweeps, and Portal 2 robustness tests.

The result should not be overstated. A 600-frame sample is small relative to the full extracted image set. It also does not include the harder Portal 2 outdoor/crowded conditions. The next report should run the same evaluation across all P1 indoor frontal frames and then expand to the full frontal protocol.

### 6.4 Resolution Sweep

To address the main paper question, frames were downsampled to multiple widths before detector inference. Predicted boxes were scaled back to the original 800x600 coordinate system and evaluated against the same derived ChokePoint face boxes. A preliminary 120-frame sweep used widths 800, 640, 480, 320, 256, and 160.

| Model | Width | AP50 | Precision | Recall | F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| SCRFD | 800 | 0.9995 | 0.9917 | 1.0000 | 0.9959 | 9.06 |
| SCRFD | 640 | 0.9991 | 0.9917 | 1.0000 | 0.9959 | 13.14 |
| SCRFD | 480 | 0.9996 | 0.9917 | 1.0000 | 0.9959 | 22.04 |
| SCRFD | 320 | 0.9677 | 0.9752 | 0.9833 | 0.9793 | 38.82 |
| SCRFD | 256 | 0.9619 | 0.9669 | 0.9750 | 0.9710 | 51.55 |
| SCRFD | 160 | 0.9372 | 0.9350 | 0.9583 | 0.9465 | 80.05 |

The preliminary cut-off depends on the chosen acceptance rule. If the deployment requirement is AP50 >= 0.95 and recall >= 0.97, 256 px width remains acceptable in this sample. If the system requires more margin before face recognition, 320 px width is the safer operating point. The 160 px condition is faster but begins to lose recall and should be treated as below the recommended operating point for this protocol.

## 7. Discussion

### 7.1 Why ChokePoint Is a Better Fit for This Project

The project objective is not to win a general web-image face benchmark. It is to understand lightweight face detection in a surveillance-camera setting. ChokePoint is a better fit because it controls the camera placement, movement direction, and portal geometry while still preserving real-world issues such as blur, pose, illumination variation, and occlusion.

By focusing on ChokePoint only, the paper can avoid mixing two different problem definitions. A detector trained and tested on broad still-image data does not necessarily answer whether the detector behaves consistently on a fixed surveillance camera. A ChokePoint-only protocol can focus on sequence-level behavior, camera-specific differences, and deployment-style constraints.

### 7.2 Risks and Limitations

The most important limitation is the label conversion step. Since original ChokePoint annotations are eyes, any detector benchmark must either derive boxes or create a new hand-labelled face-box subset. The current heuristic is transparent but not definitive. This affects IoU-sensitive metrics.

A second limitation is that off-the-shelf SCRFD is already a strong face detector. If the final paper compares it against a newly fine-tuned YOLO model, the YOLO model must be trained and evaluated carefully on ChokePoint labels rather than borrowed from unrelated experiments.

A third limitation is data leakage risk. ChokePoint contains repeated frames of the same person over time. Random frame-level splitting would leak near-identical identities and frames across train and test. The protocol must split by official sequence groups, not by random frame.

### 7.3 Next Experiments

The next phase should run:

1. Full P1 indoor frontal SCRFD benchmark over all 11,711 annotations.
2. Threshold sweep over confidence values from 0.1 to 0.9.
3. Error analysis by eye-distance bucket.
4. YOLO fine-tuning on ChokePoint-derived labels only.
5. Cross-portal evaluation from P1 to P2.
6. Robustness evaluation on crowded P2 S5 sequences if labels are available or if a pseudo-label/manual subset is created.

## 8. Conclusion

This draft reframes the project around ChokePoint only. The local dataset is fully usable for detector-oriented work after distinguishing full-frame JPG images from cropped PGM faces and after converting XML eye landmarks into derived face boxes. The initial Portal 1 indoor frontal protocol contains 11,711 usable annotations, with a clean official train/development versus evaluation split. A preliminary 600-frame SCRFD baseline validates the pipeline and produces strong AP50, precision, and recall, but the result remains preliminary until expanded to all relevant sequences.

The final paper should position the work as a surveillance-specific face detection protocol and benchmark, not as a general web-image face detection study. The strongest contribution is the reproducible ChokePoint data preparation and evaluation pipeline: XML landmarks to derived boxes, frame mapping across non-uniform extracted folders, protocol-respecting splits, and detector metrics tailored to small surveillance faces.

## References

[1] ChokePoint Dataset official page. https://arma.sourceforge.net/chokepoint/

[2] ChokePoint Dataset, Zenodo record v1, published 22 June 2017. https://zenodo.org/records/815657

[3] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, "Patch-based Probabilistic Image Quality Assessment for Face Selection and Improved Video-based Face Recognition," IEEE Biometrics Workshop, Computer Vision and Pattern Recognition Workshops, pp. 81-88, 2011. DOI: 10.1109/CVPRW.2011.5981881.

[4] J. Guo, J. Deng, A. Lattas, and S. Zafeiriou, "Sample and Computation Redistribution for Efficient Face Detection," arXiv:2105.04714, 2021. https://arxiv.org/abs/2105.04714

[5] Ultralytics, "Object Detection" documentation. https://docs.ultralytics.com/tasks/detect/

[6] Ultralytics, "Performance Metrics Deep Dive" documentation. https://docs.ultralytics.com/guides/yolo-performance-metrics/
