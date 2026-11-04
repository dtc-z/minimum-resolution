# Đề xuất tối ưu bản thảo sau review

## Định hướng biên tập

Nên định vị bài là **một benchmark thực nghiệm về trade-off độ phân giải–độ chính xác–thông lượng của phát hiện khuôn mặt trong giao thức ChokePoint Portal 1 indoor frontal**. Không trình bày nó như một detector mới: điểm mạnh hiện có là câu hỏi triển khai thực tế, sweep có hệ thống và khuyến nghị cụ thể. Các vấn đề khoa học cần xử lý trực tiếp theo review là mô tả split/rủi ro rò rỉ theo thời gian và phép so sánh chưa ngang bằng.

Thứ tự làm việc nên là: (1) xác minh và mô tả split theo nhóm chuỗi; (2) bổ sung baseline khuôn mặt công bằng hơn và kiểm định nhãn; (3) chốt số liệu/tuyên bố; (4) mới hoàn thiện hình, bố cục và metadata. Chỉ cần chạy lại nếu việc kiểm tra split phát hiện rò rỉ hoặc khi có thực nghiệm mới.

## Workspace hiện có

- [paper_v1.pdf](results/paper_v1.pdf) là bản nộp gốc; [review.md](results/review.md) chứa ba review.
- [Manifest giao thức Portal 1](chokepoint_data/modeling/chokepoint_p1_indoor_frontal_manifest.csv) có 11.711 nhãn, chia theo tám chuỗi: 6.035 train/dev và 5.676 eval. Nhóm chuỗi tách biệt nhau, nhưng bản v1 không nêu tên các chuỗi trong bài.
- [Script chuẩn bị dữ liệu](chokepoint_data/modeling/prepare_yolo_face_dataset.py), [script fine-tune](chokepoint_data/modeling/train_yolo_face_models.py) và [tóm tắt lần train GPU](chokepoint_data/modeling/yolo_finetuned_full_gpu_weights/training_summary.json) cho thấy YOLO đã được fine-tune 20 epoch trên ảnh 640 px.
- [CSV resolution sweep full-eval](chokepoint_data/modeling/benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv) có 240 dòng: 4 detector × 10 độ rộng × 6 ngưỡng. [Cấu hình triển khai Report 4](chokepoint_data/modeling/report4_deployment_config.json) đề xuất YOLOv8n-FT 320×240 làm mặc định.
- Có các hình PDF vector trong [ieee_paper_figures_en](chokepoint_data/modeling/ieee_paper_figures_en/). Đây là nguồn tốt để chỉnh lại kích thước chữ thay vì phóng to ảnh raster.
- Workspace có nhiều bản thảo khác nhau: [bản Markdown sơ bộ](docs/Paper_Revised_Draft.md), các DOCX và [nguồn LaTeX IEEEtran](docs/Paper_IEEE_ChokePoint_Resolution_Benchmark_IEEEtran.tex). Bản LaTeX dùng lớp hội nghị IEEE nhưng vẫn viết “Dept.”, giữ các Mục VII–IX riêng, có tiêu đề khác PDF nộp và abstract còn dài. Chọn một nguồn làm bản chủ sau khi đối chiếu metadata EDAS.
- [results.zip](results.zip) chứa benchmark WIDER Face riêng (3.226 ảnh, 39.112 khuôn mặt; YOLO11/YOLO12 và SCRFD ở các kích thước 160/320/640). Đây không phải phép so sánh bổ sung trên ChokePoint nên không giải quyết nhận xét về công bằng của benchmark chính. Model checkpoint và dataset lớn không cần đọc để lập kế hoạch chỉnh sửa.

## Việc cần làm trước khi viết lại kết quả

### 1. Xác minh và mô tả split theo chuỗi

Manifest cho thấy bốn chuỗi train/dev là P1E_S1_C1, P1E_S2_C2, P1L_S1_C1, P1L_S2_C2; bốn chuỗi eval là P1E_S3_C3, P1E_S4_C1, P1L_S3_C3, P1L_S4_C1. Đưa danh sách cùng số frame/nhãn mỗi chuỗi vào phần Dataset/Experimental Setup. Nêu rõ split theo chuỗi thay vì chia ngẫu nhiên frame, rồi kiểm tra không có frame hoặc đoạn video trùng giữa hai nhóm. Nếu xác nhận split hiện tại tách biệt thì mô tả rõ trong paper; chạy lại chỉ khi audit phát hiện rò rỉ thực sự.

### 2. Bổ sung đối chứng và kiểm định hộp nhãn

Phép so sánh hiện đặt YOLO đã fine-tune cạnh SCRFD tiền huấn luyện dùng nguyên trạng. Tốt nhất là fine-tune SCRFD theo cùng train/validation; nếu không khả thi, thêm ít nhất một detector khuôn mặt chuyên biệt đã tiền huấn luyện và mô tả rõ khác biệt về dữ liệu huấn luyện. Khi báo cáo, chỉ kết luận về các cấu hình và giao thức cụ thể, không suy ra ưu thế kiến trúc YOLO.

ChokePoint cung cấp mốc mắt chứ không cung cấp hộp khuôn mặt chuẩn. Hộp hiện tại có kích thước 2.8d × 3.6d, với biên trên/dưới lần lượt là cy − 1.2d và cy + 2.4d. Gán nhãn thủ công một tập con phân tầng để kiểm tra công thức này; đồng thời báo cáo độ nhạy theo kích thước hộp. Gắn nhãn rõ mọi chỉ số là so với hộp suy ra cho đến khi có đối chiếu thủ công.

### 3. Khóa giao thức tốc độ

Benchmark hiện đo vòng đọc ảnh/resize/call detector, còn nạp mô hình và tính metric nằm ngoài thời gian. Với YOLO, predict bao gồm các bước nội bộ của thư viện như tiền xử lý và NMS. Đây là thông lượng detector trong giao thức chạy này, không phải tốc độ toàn pipeline nhận dạng. Thêm warm-up, đồng bộ GPU, nhiều lượt đo và độ phân tán (median/IQR hoặc mean/std); báo cáo cấu hình phần cứng/phần mềm và các bước được tính vào FPS.

Không trộn hai lần đo tốc độ hiện có. Log benchmark cố định ghi khoảng 71.3, 75.6, 53.8 và 44.6 FPS tại 640×480; CSV resolution sweep ghi khoảng 62.8, 65.0, 51.1 và 42.3 FPS cho YOLOv5n-FT, YOLOv8n-FT, YOLOv12n-FT và SCRFD. Chọn số liệu từ lần chạy theo giao thức đã chuẩn hóa và dùng cùng nguồn đó cho bảng, hình, abstract và khuyến nghị.

## Cấu trúc bài đề xuất

| Phần | Nội dung nên đưa vào |
|---|---|
| **I. Introduction** | Vấn đề vận hành và câu hỏi nghiên cứu; xác định đóng góp là benchmark/protocol thực nghiệm, không phải detector mới. Tóm lược related work trong phần này hoặc một tiểu mục ngắn. |
| **II. Methodology** | Dataset và phạm vi; danh sách chuỗi train/validation/test; cách chuyển landmark thành hộp; detector và trạng thái fine-tune; thiết lập thực nghiệm, metrics và phép đo FPS. |
| **III. Results** | Kết quả full resolution; đường cong resolution–accuracy/recall/FPS; độ nhạy threshold; điểm vận hành theo tiêu chí đã định trước. Mỗi bảng/hình lấy từ cùng lần chạy cuối. |
| **IV. Discussion and Conclusion** | Diễn giải trade-off; khuyến nghị triển khai có điều kiện; giới hạn về nhãn, portal, GPU và việc chưa đo recognition; hướng mở rộng; kết luận ngắn. Có thể đặt Limitations/Future Work và Conclusion thành tiểu mục để đáp ứng yêu cầu gộp VII–IX. |

## Cách trình bày các kết quả đã có

CSV hiện tại tại threshold 0.25 cho thấy ở 640×480: YOLOv5n-FT đạt AP50/recall/FPS ≈ 1.000/1.000/62.8; YOLOv8n-FT ≈ 1.000/1.000/65.0; YOLOv12n-FT ≈ 1.000/1.000/51.1; SCRFD ≈ 0.988/0.996/42.3. Đây là số liệu theo hộp suy ra từ mốc mắt; SCRFD cũng chưa fine-tune trên cùng dữ liệu nên cần trình bày giới hạn so sánh. Chỉ tính lại nếu audit split hoặc thực nghiệm bổ sung yêu cầu.

Theo đúng tiêu chí hiện có (AP50 ≥ 0.95 và recall ≥ 0.95 ở threshold 0.25), CSV sweep cho các điểm cắt sau:

| Detector | Bề rộng nhỏ nhất hiện đạt tiêu chí | AP50 | Recall | FPS |
|---|---:|---:|---:|---:|
| YOLOv5n-FT | 240 px | 0.9504 | 0.9505 | 62.8 |
| YOLOv8n-FT | 288 px | 0.9801 | 0.9813 | 62.5 |
| YOLOv12n-FT | 288 px | 0.9505 | 0.9524 | 46.6 |
| SCRFD | 256 px | 0.9541 | 0.9660 | 80.2 |

Đây là số liệu theo hộp suy ra từ mốc mắt và cần tính lại sau khi tách validation. Nếu được xác nhận, kết luận có thể giữ theo cách giới hạn: vùng 240–288 px là vùng vận hành quan sát được trong giao thức Portal 1 indoor frontal; YOLOv8n-FT ở 320×240 là cấu hình mặc định bảo thủ hơn điểm cắt tối thiểu, còn 240×180 với YOLOv5n-FT sát ngưỡng nên ít dư địa. SCRFD giảm suy giảm chậm hơn ở độ phân giải rất thấp, nhưng dưới 192 px không đạt ngưỡng AP50/recall 0.95; vì thế chỉ gọi là bộ phát hiện tương đối bền hơn trong vùng thấp, không khuyến nghị như cấu hình nhận dạng đáng tin cậy ở đó.

Diễn đạt so sánh một cách có điều kiện: “Trong giao thức và nhãn suy ra này, YOLO sau fine-tune cho kết quả AP50/thông lượng cao hơn ở một số điểm vận hành so với SCRFD dùng nguyên trạng.” Không viết rằng YOLO nói chung tốt hơn SCRFD. Kết quả không đo embedding, nhận dạng danh tính hay độ trễ end-to-end; vì vậy chỉ nêu tiềm năng giảm tải phát hiện, không khẳng định đã cải thiện toàn hệ thống nhận diện.

## Sửa trình bày và thông tin nộp bài

- Viết abstract cô đọng theo bốn ý: động cơ, giao thức/phạm vi, 1–2 kết quả chính đã được kiểm chứng lại, đóng góp thực nghiệm. Bỏ danh sách chi tiết toàn bộ độ rộng/ngưỡng; nhấn mạnh đây là benchmark cho Portal 1 indoor frontal.
- Dùng “Department of Artificial Intelligence” đầy đủ. Dùng email @fe.edu.vn cho tác giả nếu có email công việc. Xác minh tên và thứ tự tác giả trực tiếp với EDAS: Review 1 báo lệch “Cao Van Mai/Mai Cao”, trong khi Review 2 nói thông tin khớp.
- Giữ nguyên tiêu đề PDF bản nộp cho tới khi đối chiếu EDAS. Các bản nháp hiện có nhiều tiêu đề khác nhau; tiêu đề bản nộp là “Minimum-Resolution Face Detection for Real-Time Surveillance: A Benchmark of YOLO and SCRFD”.
- Tăng cỡ chữ trục/legend và kiểm tra chất lượng ở kích thước in. PDF vector trong chokepoint_data/modeling/ieee_paper_figures_en có thể dùng làm nguồn tái xuất.
- Dùng đúng mẫu hội nghị được ban tổ chức yêu cầu, giữ kiểu IEEE conference trong bản cuối và xóa hướng dẫn mẫu còn sót. Trang IEEE Author Center cung cấp công cụ/mẫu hội nghị cho Word và LaTeX: [Authoring Tools and Templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/).
- Rà soát References: bản PDF v1 cần nguồn gốc SCRFD vì đây là baseline chính; nguồn đó đã có trong bản LaTeX nháp nên có thể chuyển sang bản chủ. Kiểm tra mọi tài liệu tham khảo được trích dẫn đúng trong nội dung.
