# Prompt thực hiện chỉnh sửa paper sau review

## Vai trò và mục tiêu

Bạn là cộng sự nghiên cứu về thị giác máy tính. Làm việc trong thư mục gốc `DSP391m`, đọc `results/review.md`, `optimizing.md`, `todo.md` và đối chiếu với paper cùng các artifact hiện có. Hoàn thiện một quy trình có thể tái lập để xử lý từng nhận xét của cả ba reviewer, sau đó viết bản paper đã sửa thành DOCX và tự đánh giá lại như reviewer.

Đóng góp của bài là **benchmark thực nghiệm trade-off độ phân giải–độ chính xác–thông lượng của phát hiện khuôn mặt trong ChokePoint Portal 1 indoor frontal**, không phải detector mới. Giới hạn mọi kết luận vào giao thức, các chuỗi và phần cứng được đo. Không tuyên bố đã cải thiện nhận dạng khuôn mặt hay toàn pipeline vì bài hiện chỉ đo detector-level.

## Trạng thái workspace đã xác minh

- Manifest `chokepoint_data/modeling/chokepoint_p1_indoor_frontal_manifest.csv` có 11.711 instance: `train_dev` 6.035, `eval` 5.676. Bốn chuỗi `train_dev`: P1E_S1_C1 (1.382), P1E_S2_C2 (1.135), P1L_S1_C1 (1.799), P1L_S2_C2 (1.719). Bốn chuỗi `eval`: P1E_S3_C3 (929), P1E_S4_C1 (1.531), P1L_S3_C3 (1.205), P1L_S4_C1 (2.011). Audit manifest hiện thấy không có trùng `image_path`, không trùng cặp `dataset/frame_number`, và không có chuỗi nằm ở cả hai split.
- **Lỗi giao thức cần sửa trước khi dùng lại kết quả:** `prepare_yolo_face_dataset.py` đang gán `eval` vào thư mục `val`; YAML đó được dùng trong fine-tune. Vì vậy 5.676 instance eval đã tham gia chọn checkpoint. Các checkpoint/kết quả cũ không phải phép đo test độc lập; không lấy chúng làm kết quả cuối, dù chúng còn hữu ích để tham khảo.
- `run_resolution_benchmark.py` đo một lượt, chưa có warm-up, đồng bộ GPU và lặp nhiều lần. Các FPS trong log cố định và CSV resolution sweep khác nhau; không trộn chúng.
- Workspace có hình PDF vector, script chuẩn bị dữ liệu, script train YOLO, benchmark, DOCX mẫu `docs/conference-template-a4.docx`, và nhiều bản thảo. LaTeX hiện có tiêu đề/metadata khác PDF nộp; không dùng nó làm nguồn sự thật trước khi đồng bộ.
- Người dùng đã chọn **fine-tune SCRFD** làm hướng chính để xử lý so sánh không ngang bằng. Benchmark cuối phải so YOLO fine-tune với SCRFD fine-tune trên cùng train/dev/test. Có thể giữ thêm SCRFD pretrained làm mốc triển khai, nhưng phải ghi riêng trạng thái huấn luyện.
- Môi trường được kiểm tra: Python 3.13.5; PyTorch 2.12.0 không có CUDA; ONNX Runtime chỉ có `CPUExecutionProvider`; thiếu `mmcv`, `mmdet` và `python-docx`. Máy có NVIDIA GeForce RTX 3050 Laptop GPU 4 GB, driver 610.62. Word ProgID có đăng ký trong Windows nhưng gọi COM thất bại trong phiên hiện tại; chưa xác nhận có thể tự động hóa Word. Các tiện ích `pdftotext`, LaTeX và PDF parsing Python cũng không có. Vì vậy môi trường Python hiện tại **không chạy được benchmark GPU như các log cũ, chưa có đường build DOCX tự động và chưa trích xuất được PDF bản nộp**. Không cài gì hoặc tải checkpoint/code trước khi người dùng duyệt. Tài liệu SCRFD chính thức mô tả pipeline huấn luyện qua MMDetection/MMCV và có checkpoint SCRFD riêng: [SCRFD training guide](https://github.com/deepinsight/insightface/tree/master/detection/scrfd). Đề xuất dùng `python-docx` để build DOCX và thêm PDF parser/render khi cần soát paper; chỉ cài sau khi người dùng duyệt.

## Cây thư mục cho quy trình

Đây là cây của các thành phần cần thiết; `extracted`, `paper/datasets`, `models`, checkpoint và `results.zip` là dữ liệu/binary hiện có, không liệt kê hay đọc sâu. `[...]` là thành phần mới cần tạo; không ghi đè artifact cũ.

```text
DSP391m/
├── .agents/                              # metadata công cụ
├── .git/                                 # metadata Git
├── .gitignore
├── optimizing.md                         # định hướng hiện có
├── todo.md                               # checklist review hiện có
├── paper_improve.md                      # hướng dẫn này
├── results.zip                           # benchmark WIDER Face riêng; không dùng cho paper này
├── chokepoint_data/
│   ├── extracted/                        # dataset nguồn, chỉ đọc
│   └── modeling/
│       ├── chokepoint_p1_indoor_frontal_manifest.csv
│       ├── prepare_yolo_face_dataset.py  # sửa để tạo train/dev/test tách biệt theo chuỗi
│       ├── train_yolo_face_models.py     # tái chạy bằng train và dev, không dùng test
│       ├── run_resolution_benchmark.py   # sửa hoặc thay bằng runner có warm-up/repeats
│       ├── run_yolo_scrfd_benchmark.py    # runner cũ, không dùng nguyên trạng làm số cuối
│       ├── ieee_paper_figures_en/        # hình vector hiện có; chỉ dùng sau khi kiểm tra/tái xuất
│       └── final_revision/               # [mới: chỉ chứa artifact của lần sửa này]
│           ├── config/                   # seed, split, ngưỡng, resolutions, model config
│           ├── manifests/                # audit và manifest train/dev/test/speed sample
│           ├── labels/                   # YOLO và SCRFD format từ cùng hộp đã khóa
│           ├── manual_box_audit/         # mẫu, hướng dẫn annotator, nhãn người, kết quả đối chiếu
│           ├── checkpoints/               # checkpoint YOLO-FT, SCRFD-FT và metadata
│           ├── logs/                     # log train, benchmark và môi trường
│           ├── benchmark/                # summary.csv/json, predictions.csv.gz, run_config.json
│           └── figures/                  # PDF vector và PNG độ phân giải cao, sinh từ summary.csv
├── docs/
│   ├── conference-template-a4.docx       # mẫu có sẵn; xác nhận đúng mẫu hội nghị trước khi dùng
│   ├── Paper_IEEE_ChokePoint_Resolution_Benchmark_IEEEtran.tex  # bản cũ, chỉ tham khảo
│   ├── Paper_Improved_Final.docx         # [mới: deliverable DOCX]
│   └── Paper_Review_Checklist.md         # [mới: bảng đối chiếu nhận xét và bằng chứng]
├── paper/
│   ├── datasets/                         # dữ liệu/binary hiện có, không đọc toàn bộ
│   └── yolo12n.pt                        # model, không mở/đọc trực tiếp
├── models/                               # model, không mở/đọc trực tiếp
├── runs/detect/                          # log/checkpoint chạy cũ; không lấy nhầm vào kết quả mới
└── results/
    ├── paper_v1.pdf                      # bản đã nộp
    └── review.md                         # ba review gốc
```

Các script trong `final_revision` chưa tồn tại. Tạo hoặc sửa script trước khi gọi lệnh tương ứng. Mọi đường dẫn dữ liệu nguồn phải giữ nguyên; dùng manifest, đọc ảnh theo lô khi huấn luyện/đánh giá và ưu tiên hardlink thay vì sao chép dataset.

## Bước 1 — Chỉnh, tổ chức và chạy lại phần kỹ thuật

### 1.1. Khóa giao thức và kiểm tra split

1. Tạo `config/final_protocol.yaml` ghi seed, model khởi tạo, epochs, image size, batch, workers, optimizer, augmentations, confidence, IoU, resolutions, cách resize/letterbox, hardware/software, quy tắc chọn checkpoint và quy tắc chọn operating point. Lưu hash/phiên bản manifest và checkpoint nguồn; không sửa config sau khi chạy test.
2. Tạo `audit_manifest.py` để kiểm tra cột bắt buộc, ảnh thiếu theo `has_image`, duplicate path/frame, sequence giao nhau, session/camera và frame liền kề trùng qua split. Xuất `manifests/split_audit.csv/json`. Giữ nguyên split theo nhóm chuỗi; không random split theo frame.
3. Tạo validation nội bộ chỉ từ bốn chuỗi `train_dev`. Đề xuất giữ P1E_S2_C2 (1.135 instance) làm dev và dùng ba chuỗi còn lại làm train (4.900 instance); ghi quyết định này vào config trước khi train. Bốn chuỗi eval (5.676 instance) là test khóa kín, không dùng cho early stopping, chọn epoch, threshold, resolution hay cấu hình triển khai.
4. Tạo manifest/YAML ba phần `train`, `val`, `test`. Không trỏ `val` vào `eval`. Kiểm tra lại số ảnh/instance/chuỗi trước khi chạy; dừng nếu ảnh thiếu hoặc nhóm trùng. Các checkpoint cũ đã chọn bằng eval không được coi là checkpoint cuối.

### 1.2. Kiểm định hộp suy ra từ mắt

1. Ghi rõ phép dựng hiện tại: với khoảng cách hai mắt `d` và trung điểm `(cx, cy)`, `x1/x2 = cx ± 1.4d`, `y1 = cy − 1.2d`, `y2 = cy + 2.4d`; cắt tọa độ vào ảnh nguồn 800×600. Đây là **hộp suy ra từ landmark mắt**, không phải rectangle mặt do người gán.
2. Tạo tập manual audit phân tầng theo bốn chuỗi eval, kích thước mặt (`eye_distance_px`) và mức che khuất. Đề xuất tối thiểu 240 instance nếu đủ mẫu; ghi seed, quy tắc chọn và strata. Tạo giao diện gán rectangle nhẹ bằng OpenCV (đã có trong môi trường), hướng dẫn annotator thống nhất về biên mặt/che khuất; gán kép ít nhất 20% mẫu nếu có annotator thứ hai, rồi adjudicate bất đồng. Không đọc hoặc duyệt toàn bộ dataset thủ công.
3. Khóa model và operating point từ dev trước khi mở nhãn manual audit. So sánh hộp suy ra với hộp người gán bằng IoU/phân bố sai lệch và báo AP50/recall trên tập manual audit riêng. Báo thêm độ nhạy khi co/giãn hộp quanh tâm theo hệ số 0.9/1.0/1.1. Không chỉnh công thức để tối ưu điểm test.
4. Trong mọi CSV, bảng, caption và văn bản, gọi metric trên nhãn tự động là **AP50/recall so với hộp suy ra từ mốc mắt**. Không diễn giải AP50=1.000 là phát hiện mặt hoàn hảo theo ground truth người gán. Giải thích AP50 được ưu tiên vì vị trí biên hộp suy ra không chính xác như box thủ công; trình bày giới hạn localization.

### 1.3. Huấn luyện lại YOLO và fine-tune SCRFD

1. Tái chạy YOLOv5n/YOLOv8n/YOLOv12n trên cùng 4.900 train instance, dùng 1.135 dev instance để chọn checkpoint/epoch. Cố định seed và config; đưa output vào `final_revision/checkpoints` và log vào `logs`. Bật ghi lại toàn bộ tham số. Dùng project/run directory mới và force train để không lấy nhầm checkpoint cũ.
2. Cài/checkout pipeline SCRFD chính thức trong **môi trường riêng đã được duyệt**; pin commit, phiên bản MMDetection/MMCV/PyTorch/CUDA. Lấy checkpoint SCRFD dạng huấn luyện tương thích, kiểm tra nguồn và giấy phép. ONNX `buffalo_l/det_10g` dùng cho inference không mặc nhiên là checkpoint fine-tune được.
3. Chuyển đúng cùng các ảnh/box train và dev sang format SCRFD; không đưa eval vào config, sampler hay validation. Fine-tune SCRFD bằng train/dev giống YOLO về phân hoạch dữ liệu; có thể giữ loss/augmentation riêng theo kiến trúc nhưng phải ghi rõ để diễn giải phép so sánh là cùng dữ liệu, không giả vờ cùng recipe. Xuất checkpoint, config, seed, log, best epoch và lệnh chạy.
4. Nếu không tìm thấy checkpoint/pipeline SCRFD hợp lệ hoặc không thể chạy trên môi trường đã duyệt, **dừng trước khi đổi baseline**. Báo rõ trở ngại và hỏi người dùng có muốn dùng baseline mặt pretrained khác hay chấp nhận mô tả SCRFD off-the-shelf như so sánh triển khai có giới hạn. Không gọi SCRFD pretrained là fine-tuned.

### 1.4. Benchmark cuối, đo tốc độ và biểu đồ

1. Chỉ sau khi khóa model/config/threshold từ dev mới chạy test. Dùng toàn bộ bốn chuỗi eval, 5.676 instance, ở 10 độ rộng `[640, 480, 384, 320, 288, 256, 240, 192, 160, 128]`, giữ đúng tỷ lệ ảnh. Tính threshold sensitivity ở `[0.25, 0.35, 0.45, 0.55, 0.65, 0.70]`, IoU=0.50. Tách rõ AP50, recall, precision, F1, FP/FN, và kết quả manual audit.
2. Đo tốc độ riêng trên một speed sample cố định, seed 391, phân tầng theo bốn chuỗi eval (đề xuất 150 frame/chuỗi). Chạy warm-up, đồng bộ GPU trước/sau vùng tính thời gian, lặp ít nhất 5 lần, báo median và IQR. Đo vòng đọc ảnh + resize + detector (YOLO NMS/hậu xử lý hoặc SCRFD hậu xử lý tương ứng); loại thời gian load model và metric khỏi FPS. Không gọi đây là tốc độ end-to-end recognition. Giữ cùng GPU/provider, luồng đọc, kích thước và quy tắc đo cho mọi model; nếu cần đo inference-only thì ghi thành cột riêng.
3. Ghi model/GPU/VRAM, CPU, RAM, Windows, Python, PyTorch, Ultralytics, CUDA/cuDNN, InsightFace, ONNX Runtime provider, driver, pretrained source, seed, epochs, imgsz, batch, workers, resize/letterbox, confidence, IoU, warm-up, số lần lặp, thời gian đọc/tiền/hậu xử lý. Lưu `run_config.json` và log từng lượt.
4. Chọn operating point theo ngưỡng đã định trước: AP50≥0.95 và recall≥0.95 tại confidence 0.25; quyết định cấu hình khuyến nghị chỉ dựa trên dev. Trên test, báo kết quả của lựa chọn đã khóa và threshold sweep như phân tích độ nhạy, không chọn cấu hình tối ưu từ test. Nếu đề xuất ngưỡng khác, chọn trên dev và nêu rõ.
5. Tạo **một summary CSV chuẩn duy nhất** từ lần benchmark cuối. Sinh tất cả bảng, hình, abstract, caption, khuyến nghị và deployment config từ CSV/JSON này. Không trộn FPS của log cũ với CSV cũ; không dùng số liệu 600-frame/CPU hoặc `results.zip` WIDER Face. Lưu dự đoán cuối dạng `csv.gz` nếu cần để kiểm toán, tránh tạo thêm bản prediction lớn không cần thiết.
6. Tái xuất PDF vector và PNG ≥300 dpi. Tăng chữ trục/legend/caption, thống nhất font/đơn vị; xem hình ở kích thước một cột và hai cột. Gắn nguồn số liệu và phiên bản run cho từng figure. Chỉ giữ nhận xét như YOLOv12 suy giảm nhanh hơn hoặc vùng 240–288 px nếu kết quả test cuối thực sự xác nhận.

### 1.5. Lệnh chạy mục tiêu

Các lệnh dưới đây là giao diện đề xuất cho script mới; implement và kiểm tra cú pháp trước khi chạy. Không dùng lệnh train cũ với YAML hiện tại.

```powershell
python chokepoint_data/modeling/revision/audit_manifest.py `
  --manifest chokepoint_data/modeling/chokepoint_p1_indoor_frontal_manifest.csv `
  --out chokepoint_data/modeling/final_revision/manifests

python chokepoint_data/modeling/revision/prepare_dataset.py `
  --manifest chokepoint_data/modeling/chokepoint_p1_indoor_frontal_manifest.csv `
  --dev-sequence P1E_S2_C2 --seed 391 `
  --out chokepoint_data/modeling/final_revision

python chokepoint_data/modeling/train_yolo_face_models.py `
  --data chokepoint_data/modeling/final_revision/config/yolo_data.yaml `
  --models yolov5n,yolov8n,yolov12n --epochs 20 --imgsz 640 `
  --seed 391 --device 0 --force `
  --project chokepoint_data/modeling/final_revision/checkpoints/yolo_runs

python chokepoint_data/modeling/revision/train_scrfd_p1.py `
  --config chokepoint_data/modeling/final_revision/config/scrfd_config.py `
  --work-dir chokepoint_data/modeling/final_revision/checkpoints/scrfd

python chokepoint_data/modeling/revision/benchmark_final.py `
  --manifest chokepoint_data/modeling/final_revision/manifests/test.csv `
  --models all --resolutions 640,480,384,320,288,256,240,192,160,128 `
  --thresholds 0.25,0.35,0.45,0.55,0.65,0.70 --iou 0.50 `
  --speed-sample 600 --warmup 50 --repeats 5 --seed 391 `
  --out chokepoint_data/modeling/final_revision/benchmark

python chokepoint_data/modeling/revision/evaluate_manual_boxes.py `
  --annotations chokepoint_data/modeling/final_revision/manual_box_audit/annotations.csv `
  --predictions chokepoint_data/modeling/final_revision/benchmark/predictions.csv.gz `
  --out chokepoint_data/modeling/final_revision/manual_box_audit/summary.json

python chokepoint_data/modeling/revision/plot_final_benchmark.py `
  --summary chokepoint_data/modeling/final_revision/benchmark/summary.csv `
  --out chokepoint_data/modeling/final_revision/figures
```

Lệnh benchmark có thể điều chỉnh theo CLI thực tế; bảo đảm accuracy dùng đủ 5.676 test instance, còn speed sample dùng một tập cố định. Không chạy training/benchmark GPU khi PyTorch báo `torch.cuda.is_available() == False` hoặc ONNX Runtime thiếu CUDA provider nếu SCRFD cần chạy trên GPU.

## Bước 2 — Viết lại paper thành DOCX

1. Đối chiếu trực tiếp PDF đã nộp và EDAS. Giữ nguyên tiêu đề PDF: **“Minimum-Resolution Face Detection for Real-Time Surveillance: A Benchmark of YOLO and SCRFD”**. Nếu tiêu đề/tên/tác giả trong EDAS khác, dừng và hỏi người dùng; không tự chọn giữa “Cao Van Mai” và “Mai Cao”. Dùng đúng thứ tự tác giả, tên đầy đủ, `Department of Artificial Intelligence`, FPT University; dùng email `@fe.edu.vn` nếu tác giả xác nhận địa chỉ đó. Không tự tạo email.
2. Xác nhận với yêu cầu hội nghị rằng `docs/conference-template-a4.docx` là đúng mẫu và đúng khổ giấy. Nếu ban tổ chức cung cấp template riêng, ưu tiên template đó. IEEE Author Center có Word/LaTeX conference templates: [Authoring Tools and Templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/).
3. Dùng một nguồn nội dung chuẩn và tạo `docs/Paper_Improved_Final.docx`. Hiện chưa xác nhận được Word COM trong phiên này; xin duyệt cài `python-docx` và PDF parser/render cần thiết trước khi dùng builder tự động, hoặc yêu cầu người dùng cung cấp môi trường Office chạy được. Không dùng bản DOCX/LaTeX cũ nguyên xi vì khác tiêu đề, metadata, số liệu và cấu trúc. Abstract ngắn, lần lượt nêu động cơ/phạm vi, protocol/dataset, 1–2 kết quả đã xác nhận, đóng góp benchmark và giới hạn tác động.
4. Tổ chức theo cấu trúc truyền thống: **I. Introduction; II. Methodology** (Related Work, Dataset/Split, Derived Labels, Models/Implementation, Metrics); **III. Results; IV. Discussion and Conclusion**, với tiểu mục Limitations/Future Work và Conclusion. Cấu trúc này gộp các mục VII–IX theo yêu cầu reviewer, không lặp kết luận.
5. Bàn luận kết quả so YOLO-FT và SCRFD-FT trên cùng dữ liệu. Nếu có SCRFD off-the-shelf, đánh nhãn rõ là kết quả pretrained riêng. Dùng cách diễn đạt giới hạn vào protocol: ví dụ “trong giao thức và hộp nhãn này, các cấu hình YOLO fine-tuned đạt…”. Không kết luận kiến trúc YOLO nói chung vượt SCRFD. Giải thích novelty vừa phải: benchmark/sweep và hướng dẫn operating point, không phải phương pháp detector mới.
6. Ghi rõ split theo bốn chuỗi, số instance từng chuỗi, split train/dev/test, công thức hộp, manual audit, trạng thái huấn luyện từng model, metric, hardware/software, protocol FPS và tiêu chí chọn operating point. Bàn luận phạm vi chỉ Portal 1 indoor frontal và một GPU; chưa đánh giá outdoor, nhiều camera, occlusion rộng, nhận dạng danh tính hoặc latency toàn pipeline.
7. Chỉ dùng kết quả từ `final_revision/benchmark/summary.csv` và manual audit đã khóa. Rà References: trích nguồn SCRFD trong thân bài, kiểm tra mọi citation có mục tài liệu tương ứng và ngược lại. Sửa đơn vị/viết tắt affiliation, thuật ngữ, caption, hình ở kích thước in; xóa toàn bộ hướng dẫn mẫu còn sót.

## Bước 3 — Đóng vai reviewer và kiểm cuối

Tạo `docs/Paper_Review_Checklist.md`, mỗi dòng có mã reviewer, yêu cầu, vị trí đã sửa trong DOCX, bằng chứng (bảng/hình/log/config) và trạng thái. Kiểm từng mục:

- R1: giải thích split video và nguy cơ temporal leakage; kiểm tra không chia ngẫu nhiên frame; công khai chuỗi; làm rõ YOLO fine-tune/SCRFD fine-tune; manual-label bias; giới hạn một portal/GPU; đo FPS và chi tiết tái lập.
- R2: đúng template; ghi đầy đủ Department; email công việc nếu có; abstract cô đọng; hình đọc được khi in; gộp Sections VII–IX vào phần Discussion and Conclusion; xử lý so sánh high-accuracy YOLO/SCRFD bằng phép so sánh đã fine-tune ngang dữ liệu.
- R3: nêu rõ hộp từ landmark không phải rectangle chuẩn; giới hạn ChokePoint Portal 1 indoor; công khai chưa đo downstream recognition; giữ nhận xét về dải độ phân giải và model chỉ nếu CSV cuối xác nhận.
- Submission policy: title, author order/name, abstract và metadata phải khớp PDF/EDAS; mọi thông tin chưa xác minh là blocker, không tự suy đoán.
- Cross-check số: abstract ↔ bảng ↔ hình ↔ caption ↔ conclusion ↔ `summary.csv`; kiểm tra không lẫn CSV cũ, WIDER Face, checkpoint cũ hoặc FPS log cũ.
- Mở DOCX trong Word, xuất PDF kiểm tra phân trang, hai cột, công thức, hình, bảng, font, References và hướng dẫn mẫu còn sót. Xem từng hình ở kích thước in. Lưu checklist và bản DOCX cuối; chỉ đánh dấu xong khi mọi yêu cầu P0/P1 có bằng chứng và không còn blocker metadata.

## Tiêu chí hoàn tất

1. Có audit split tái lập, train/dev/test tách theo chuỗi; test chưa tham gia huấn luyện/chọn checkpoint.
2. Có manual box audit và chỉ số được gọi đúng theo loại nhãn.
3. YOLO và SCRFD đều được fine-tune trên train/dev hợp lệ, benchmark chung test, một protocol tốc độ đã chuẩn hóa; toàn bộ bảng/hình sinh từ cùng run cuối.
4. Có DOCX theo template được hội nghị xác nhận, giữ nguyên title, metadata đúng EDAS, cấu trúc đáp ứng review và không phóng đại claim.
5. Có checklist reviewer với bằng chứng cho từng nhận xét; DOCX đã mở và kiểm trực quan.
6. Nếu còn thiếu package, checkpoint, mã huấn luyện hoặc thông tin tác giả/template, nêu chính xác phần thiếu và xin duyệt trước khi cài/tải hoặc chốt thông tin; sau khi cài phải kiểm tra lại phiên bản, CUDA/GPU provider và lệnh import trước khi chạy.
