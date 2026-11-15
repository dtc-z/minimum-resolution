# TODO chỉnh sửa paper theo review

Đối chiếu trực tiếp [paper_v1.pdf](results/paper_v1.pdf), [review.md](results/review.md) và các artifact hiện có. Mức ưu tiên: **P0** cần xử lý trước khi chốt kết quả; **P1** cần sửa trong bản thảo nộp lại; **P2** hoàn thiện để tăng độ tái lập và độ tin cậy.

## P0 — Làm rõ train/test split và kiểm tra rò rỉ theo thời gian

- [x] Review 1 nêu paper chưa giải thích rõ train/test split và có nguy cơ rò rỉ theo thời gian. Bổ sung mô tả cách chia dữ liệu vào phần Dataset/Experimental Setup; tránh chỉ ghi tổng số frame.
- [x] **Ghi đầy đủ danh sách chuỗi của từng tập.** Manifest hiện cho thấy:
  - Train/dev: P1E_S1_C1 (1.382 nhãn), P1E_S2_C2 (1.135), P1L_S1_C1 (1.799), P1L_S2_C2 (1.719).
  - Eval: P1E_S3_C3 (929), P1E_S4_C1 (1.531), P1L_S3_C3 (1.205), P1L_S4_C1 (2.011).
  
  Tổng số lần xuất hiện nhãn tương ứng là 6.035 và 5.676. Ghi rõ split hiện tại được tạo theo chuỗi, không chia ngẫu nhiên từng frame liền kề.
- [x] Kiểm tra manifest để xác nhận không có cùng frame/chuỗi nằm ở cả hai phía; trình bày rõ kết quả kiểm tra trong paper. Nếu xác nhận các nhóm chuỗi tách biệt như manifest hiện thể hiện, mô tả đầy đủ giao thức; chỉ điều chỉnh split và chạy lại nếu việc kiểm tra phát hiện rò rỉ thực sự.

## P0 — Làm phép so sánh bộ phát hiện công bằng hơn

- [x] Hiện YOLO được fine-tune trên nhãn suy ra từ ChokePoint, còn SCRFD dùng trọng số tiền huấn luyện buffalo_l. Vì vậy không quy chênh lệch cho kiến trúc hay kết luận YOLO nói chung vượt SCRFD.
- [x] Ưu tiên một trong hai hướng Reviewer 2 nêu: (a) fine-tune SCRFD trên đúng tập train và quy trình tách nhóm như YOLO; hoặc (b) bổ sung một hay nhiều bộ phát hiện khuôn mặt chuyên biệt đã tiền huấn luyện, đánh giá trên cùng eval, ví dụ YOLOv8-face/YOLOv5-face, RetinaFace, YuNet, MTCNN hoặc MediaPipe/BlazeFace.
- [x] Nếu giữ SCRFD nguyên trạng, mô tả kết quả là so sánh giữa **YOLO fine-tune theo miền** và **SCRFD tiền huấn luyện dùng nguyên trạng**. Nêu rõ dữ liệu huấn luyện đã biết của từng mô hình nếu có, đồng thời gọi đây là so sánh triển khai chứ không phải phép đối chứng huấn luyện ngang bằng.

## P0 — Kiểm định nhãn và diễn giải AP50

- [x] Ghi công thức đầy đủ: với khoảng cách hai mắt d và trung điểm (cx, cy), hộp hiện dùng x = cx ± 1.4d, y trên = cy − 1.2d, y dưới = cy + 2.4d, sau đó cắt tọa độ vào biên ảnh 800×600. Nêu rõ đây là hộp suy ra, không phải nhãn mặt chữ nhật gốc.
- [ ] Bổ sung một tập con được gán nhãn thủ công thành hình chữ nhật khuôn mặt, lấy mẫu phân tầng theo chuỗi, kích thước mặt và mức che khuất. So sánh hộp thủ công với hộp suy ra, rồi báo cáo AP/recall trên cả hai cách gán nhãn hoặc báo cáo độ nhạy theo kích thước hộp. (Con mo: chua co nhan mat doc lap; toa do tep 240 khung hinh trung InsightFace, khong the dung lam xac thuc thu cong.)
- [x] Trong toàn bài, gọi chỉ số là “AP50 so với hộp suy ra từ mốc mắt”; không diễn giải AP50 = 1.000 như phát hiện mặt hoàn hảo theo nhãn mặt do người đánh dấu. Giải thích vì sao AP50 được ưu tiên hơn AP75/AP50:95 trong giao thức này.

## P1 — Siết chặt phép đo tốc độ và chi tiết tái lập

- [x] Làm rõ FPS hiện tại được đo thế nào. Mã benchmark tính thời gian theo vòng xử lý frame, gồm đọc ảnh/resize và gọi detector; thời gian nạp mô hình và tính metric nằm ngoài vòng này. Với YOLO, lời gọi predict bao gồm tiền/hậu xử lý của thư viện; không có số đo riêng cho các thành phần hoặc cho pipeline nhận dạng ở bước sau.
- [ ] Ghi rõ GPU/VRAM, CPU, RAM, phiên bản hệ điều hành, Python, PyTorch, Ultralytics, CUDA/cuDNN, InsightFace/ONNX Runtime và driver; thêm trọng số khởi tạo, seed, epochs, imgsz, batch, workers, threshold, IoU và chính sách letterbox/resize. (Con mo: log benchmark thieu CPU, RAM, OS, Python, cuDNN, phien ban InsightFace/ONNX Runtime va driver; khong suy ra tu may hien tai.)
- [x] Thêm warm-up, đồng bộ GPU trước/sau khoảng đo, lặp nhiều lượt và báo cáo trung vị/độ lệch hoặc khoảng phân vị. Nếu chỉ có thể báo FPS tổng, ghi rõ FPS gồm đọc ảnh, giảm mẫu, inference và NMS/hậu xử lý nào; không gọi đó là thông lượng toàn hệ thống nhận dạng.
- [x] **Đối soát FPS giữa các lần chạy trước khi cập nhật bảng.** Bản v1 và log benchmark cố định ghi khoảng 71.3/75.6/53.8/44.6 FPS ở 640×480 cho YOLOv5n-FT/YOLOv8n-FT/YOLOv12n-FT/SCRFD. CSV của resolution sweep khác lần đo: khoảng 62.8/65.0/51.1/42.3 FPS ở cùng bề rộng. Không trộn hai lần đo trong một bảng; dùng cùng giao thức đo và một nguồn CSV cho toàn bộ bảng/hình.
- [x] Nêu rõ tiêu chí chọn điểm vận hành (AP50, recall và confidence threshold). Trình bày sweep 6 ngưỡng như phân tích độ nhạy và phân biệt với cấu hình được dùng cho khuyến nghị triển khai.

## P1 — Sửa bố cục, tóm tắt và hình

- [ ] Re-export DOCX to PDF and confirm the final page count is at most 6. The current PDF is 5 pages but predates the latest grayscale marker and line-style update.
- [x] Viết lại abstract ngắn, theo thứ tự: vấn đề và phạm vi → giao thức/dataset → kết quả nổi bật sau khi chạy lại → đóng góp thực nghiệm và tác động có giới hạn. Bỏ liệt kê quá nhiều chiều rộng/ngưỡng/cấu hình; không gọi đây là phương pháp phát hiện mới.
- [x] Tổ chức lại paper_v2 thành chín mục chính như version1 để tiện đối chiếu, nhưng dùng nội dung, số liệu và phân tích cập nhật; không khôi phục các kết luận cũ đã bị thay bằng benchmark mới.
- [x] Làm lại 4 hình tổng hợp ở bề rộng in một cột; giữ bảng màu cho bản mềm, gán marker và kiểu nét riêng cho từng model để bản trắng đen vẫn phân biệt được; dùng thang sáng cividis cho heatmap. Đã kiểm tra mô phỏng grayscale.

## P1 — Giới hạn tuyên bố và thống nhất metadata

- [x] Giới hạn mọi kết luận vào **Portal 1, indoor frontal, các chuỗi và phần cứng đã đo**. Không suy rộng thành hiệu năng giám sát ngoài trời/đa camera hoặc hệ thống thời gian thực nói chung nếu chưa có đánh giá tương ứng.
- [x] Bài chỉ đo detector-level AP50/recall/FPS, chưa đo nhận dạng sau crop. Không khẳng định giảm độ phân giải đã cải thiện độ chính xác nhận dạng hay ngăn quá tải toàn pipeline; diễn đạt là tiềm năng giảm tải detector, hoặc bổ sung thử nghiệm end-to-end nếu muốn đưa ra kết luận hệ thống.
- [ ] Xác minh tên tác giả thứ hai trực tiếp với hồ sơ EDAS và tác giả, vì Review 1 ghi “Cao Van Mai / Mai Cao” là không nhất quán còn Review 2 nói thông tin khớp. Không tự chọn một biến thể. (Email công việc có trong artifact; chưa thể xác minh hồ sơ EDAS từ workspace.)
- [ ] Giữ tiêu đề PDF bản nộp trừ khi EDAS cho thấy tiêu đề khác cần dùng. Workspace hiện có nhiều tiêu đề khác nhau trong DOCX/Markdown/LaTeX; đồng bộ tiêu đề, thứ tự tác giả, email và tóm tắt với hồ sơ nộp cuối. (Tiêu đề, tác giả, email và abstract trong DOCX/Markdown đã đồng bộ; đối chiếu EDAS còn mở.)
- [x] Đổi “Dept. of Artificial Intelligence” thành “Department of Artificial Intelligence”. Thay Gmail của tác giả bằng email công việc @fe.edu.vn nếu tác giả có địa chỉ đó; không tự tạo địa chỉ.
- [x] Khôi phục danh mục theo paper_v1 ở các vị trí [1]–[5], thêm tài liệu mới ở [6]–[10], và chỉ bỏ mục tham khảo cuối cùng [11] theo yêu cầu mới nhất.

## P2 — Rà soát tính nhất quán của artifact

- [x] Chọn một nguồn văn bản chuẩn. Không lấy số liệu sơ bộ 600-frame/CPU từ docs/Paper_Revised_Draft.md lẫn với benchmark full-eval GPU.
- [x] Đồng bộ abstract, bảng số liệu, caption, hình, deployment config và kết luận với cùng CSV sau khi chạy lại.
- [x] Tách rõ results.zip: CSV/JSON trong ZIP thuộc một thử nghiệm WIDER Face riêng, không phải benchmark ChokePoint của bài này; chỉ đưa vào bài nếu chủ động mở rộng phạm vi và mô tả thiết lập mới.
