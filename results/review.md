Review Comments:

======= Review 1 =======

> *** Novelty and originality: Rate the novelty and originality of the ideas or results presented in the paper.
Minor variations on a well investigated subject. (2)

> *** Technical content and scientific rigour: Rate the technical content of the paper (e.g.: completeness of the analysis or simulation study, thoroughness of the treatise, accuracy of the models, etc.), its soundness and scientific rigour.
Valid work but limited contribution. (3)

> *** Quality of presentation: Rate the paper organization, the clearness of text and figures, the completeness and accuracy of references.
Well written. (4)

> *** Relevance and timeliness: Rate the importance and timeliness of the topic addressed in the paper within its area of research.
Good (4)

> *** Strong aspects: Comments to the authors what are the strong aspects of the paper

- The problem is practical. Real surveillance systems often struggle when many faces appear at once.
- The resolution sweep from 640 px to 128 px is systematic and clearly presented.
- The writing is easy to follow. Figures and tables are clear.
- The deployment suggestions are concrete and may help practitioners.

> *** Weak aspects: Comments to the authors what are the weak aspects of the paper

- Novelty is limited. The paper mainly fine-tunes existing models and reports resolution-accuracy-speed results. It does not propose any new method.
- Face boxes are created by a fixed formula from eye landmarks. Models are trained and tested on these same boxes, so the perfect AP50 = 1.000 may partly show agreement with the formula rather than true detection quality.
- ChokePoint is video data. The paper does not clearly explain the train/test split. There is a risk of temporal leakage.
- The comparison is not fair. YOLO models are fine-tuned, but SCRFD is not. This makes YOLO look better for an obvious reason.
- Experiments only use one indoor portal and one laptop GPU. Yet the paper makes broader claims about real-time surveillance.
- FPS reporting misses important details (preprocessing, NMS, variance, etc.). Some reproducibility details are missing.

> *** Submission Policy: Does the paper list the same author(s), title and abstract (minor wording differences in the abstract are ok) in its PDF file and EDAS registration?

Title and abstract match. The second author name is inconsistent (PDF: Cao Van Mai / EDAS: Mai Cao).

======= Review 2 =======

> *** Novelty and originality: Rate the novelty and originality of the ideas or results presented in the paper.
Minor variations on a well investigated subject. (2)

> *** Technical content and scientific rigour: Rate the technical content of the paper (e.g.: completeness of the analysis or simulation study, thoroughness of the treatise, accuracy of the models, etc.), its soundness and scientific rigour.
Valid work but limited contribution. (3)

> *** Quality of presentation: Rate the paper organization, the clearness of text and figures, the completeness and accuracy of references.
Readable, but revision is needed in some parts. (3)

> *** Relevance and timeliness: Rate the importance and timeliness of the topic addressed in the paper within its area of research.
Acceptable (3)

> *** Strong aspects: Comments to the authors what are the strong aspects of the paper

The strong aspects include: The emerging research topic for real-time surveillance applications.
There are no additional comments about the strong aspects of this paper.

> *** Weak aspects: Comments to the authors what are the weak aspects of the paper

1. COMMENT 01:  The authors' presentation does not follow the required template. Please use the template provided at the following link when writing the paper: https://www.ieee.org/conferences/publishing/templates.
2. COMMENT 02: It would be good to use the full name for the contact of the authors. “Dept. of Artificial Intelligence” should be replaced by “Department of Artificial Intelligence”.

3. COMMENT 03:  If the personal contact email such as Gmail should be avoided if the work contact email is available (@fe.edu.vn).

4. COMMENT 04:  The quality of the abstract can be improved by making it more compact. It would be good to minimise detailed technical content and instead focus on the highlights of the research methods, key findings, novelty, scientific and technological contributions, and impacts of the study.

5. COMMENT 05:  The quality of figures should be improved with the enhanced text size and image quality.

6. COMMENT 06:  The writing structure of a paper can be much improved. Sections VII, VIII and IX can be combined into one section “DICUSSIONS AND CONCLUSION”.

It would be optimal to use the traditional academic writing structure, with the following sections: Introduction, Methods/Methodology, Results, Discussion, and Conclusion.

7. COMMENT 07:  It would be necessary to take into account the conclusion drawn from the comparison in the study: "YOLOv5n/YOLOv8n can outperform SCRFD in the high-accuracy operating region".

The YOLO models are fine-tuned on in-domain data, while SCRFD is used off the shelf. It would be good to provide comparisons that either fine-tune SCRFD on the same 6,035 frames or compare against other pretrained face-specific detectors such as YOLOv8-face, YOLOv5-face, RetinaFace, MediaPipe/BlazeFace, YuNet, and MTCNN.

> *** Submission Policy: Does the paper list the same author(s), title and abstract (minor wording differences in the abstract are ok) in its PDF file and EDAS registration?

Yes.  The paper lists the same author(s), title and abstract in its PDF file and EDAS registration.

======= Review 3 =======

> *** Novelty and originality: Rate the novelty and originality of the ideas or results presented in the paper.
Significant original work and novel results. (4)

> *** Technical content and scientific rigour: Rate the technical content of the paper (e.g.: completeness of the analysis or simulation study, thoroughness of the treatise, accuracy of the models, etc.), its soundness and scientific rigour.
Solid work of notable importance. (4)

> *** Quality of presentation: Rate the paper organization, the clearness of text and figures, the completeness and accuracy of references.
Well written. (4)

> *** Relevance and timeliness: Rate the importance and timeliness of the topic addressed in the paper within its area of research.
Good (4)

> *** Strong aspects: Comments to the authors what are the strong aspects of the paper

Practical motivation: The paper directly targets a real-world bottleneck in surveillance pipelines where face recognition services overload during crowd spikes.
Thorough empirical evaluation: The benchmark systematically tests 10 downsampled widths (640px to 128px) and 6 confidence thresholds on physical hardware (NVIDIA GeForce RTX 3050 Laptop GPU) across 5,676 evaluation frames.
Clear deployment insights: The paper demonstrates that COCO-pretrained YOLO models fail completely without fine-tuning (AP50 = 0.000), establishes a clear practical operating band of 240–288 px, and reveals that attention-centric YOLOv12n-FT degrades faster than YOLOv5n-FT and YOLOv8n-FT under severe spatial compression.
Actionable recommendations: Provides concrete deployment guidelines based on specific operational priorities (YOLOv8n-FT for general balance, YOLOv5n-FT for aggressive compression, and SCRFD as a fallback for extreme low resolution).

> *** Weak aspects: Comments to the authors what are the weak aspects of the paper

Derived bounding-box labels: Ground-truth boxes are constructed deterministically from eye landmarks (2.8d x 3.6d) rather than human-annotated face rectangles, which may introduce bias into IoU and localization metrics.
Restricted dataset evaluation: The evaluation is strictly limited to the indoor Portal 1 split of the ChokePoint dataset, lacking cross-dataset validation, outdoor scenarios, or heavy occlusion testing.
Absence of downstream recognition evaluation: The paper focuses exclusively on detector-level metrics (AP50, recall, FPS) and does not measure how downsampled face crops affect downstream face feature extraction or identification accuracy.

> *** Submission Policy: Does the paper list the same author(s), title and abstract (minor wording differences in the abstract are ok) in its PDF file and EDAS registration?