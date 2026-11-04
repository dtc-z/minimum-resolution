# Bản dịch tiếng Việt của nội dung review

*Ghi chú: Các ký tự lỗi mã hóa trong bản review gốc đã được khôi phục theo ngữ cảnh. Tên mô hình, tên riêng và ký hiệu chỉ số được giữ nguyên.*

## Review 1

### Tính mới và tính nguyên gốc

Những biến thể nhỏ của một chủ đề đã được nghiên cứu kỹ. **(2)**

### Nội dung kỹ thuật và tính nghiêm cẩn khoa học

Công trình có giá trị, nhưng đóng góp còn hạn chế. **(3)**

### Chất lượng trình bày

Trình bày tốt. **(4)**

### Mức độ phù hợp và tính cập nhật

Tốt. **(4)**

### Điểm mạnh

- Bài toán có tính thực tiễn. Các hệ thống giám sát thực tế thường gặp khó khăn khi nhiều khuôn mặt xuất hiện cùng lúc.
- Khảo sát độ phân giải từ 640 px đến 128 px có hệ thống và được trình bày rõ ràng.
- Bài viết dễ theo dõi. Hình và bảng rõ ràng.
- Các đề xuất triển khai cụ thể và có thể hữu ích cho người thực hành.

### Điểm yếu

- Tính mới còn hạn chế. Bài báo chủ yếu fine-tune các mô hình hiện có rồi báo cáo kết quả theo độ phân giải, độ chính xác và tốc độ; bài chưa đề xuất phương pháp mới.
- Các hộp khuôn mặt được tạo bằng một công thức cố định từ mốc mắt. Mô hình được huấn luyện và kiểm thử trên chính các hộp này, nên AP50 hoàn hảo bằng 1.000 có thể phần nào phản ánh mức độ khớp với công thức thay vì chất lượng phát hiện khuôn mặt thực sự.
- ChokePoint là dữ liệu video. Bài báo chưa giải thích rõ cách chia tập huấn luyện/kiểm thử, nên có nguy cơ rò rỉ thông tin theo thời gian.
- Phép so sánh chưa công bằng. Các mô hình YOLO được fine-tune nhưng SCRFD thì không, khiến YOLO có lợi thế rõ ràng.
- Thực nghiệm chỉ dùng một cổng trong nhà và một GPU laptop, trong khi bài báo đưa ra nhận định rộng hơn về giám sát thời gian thực.
- Báo cáo FPS còn thiếu các chi tiết quan trọng như tiền xử lý, NMS, độ biến thiên và một số thông tin cần thiết để tái lập kết quả.

### Chính sách nộp bài

PDF có liệt kê cùng tác giả, tiêu đề và tóm tắt như hồ sơ đăng ký EDAS không? (Có thể có khác biệt nhỏ về câu chữ trong tóm tắt.)

Tiêu đề và tóm tắt trùng khớp. Tên tác giả thứ hai không nhất quán (**PDF: Cao Van Mai / EDAS: Mai Cao**).

## Review 2

### Tính mới và tính nguyên gốc

Những biến thể nhỏ của một chủ đề đã được nghiên cứu kỹ. **(2)**

### Nội dung kỹ thuật và tính nghiêm cẩn khoa học

Công trình có giá trị, nhưng đóng góp còn hạn chế. **(3)**

### Chất lượng trình bày

Dễ đọc, nhưng một số phần cần được chỉnh sửa. **(3)**

### Mức độ phù hợp và tính cập nhật

Chấp nhận được. **(3)**

### Điểm mạnh

Điểm mạnh gồm chủ đề nghiên cứu mới nổi về ứng dụng giám sát thời gian thực. Không có nhận xét bổ sung về điểm mạnh của bài báo.

### Điểm yếu

1. **NHẬN XÉT 01:** Cách trình bày của tác giả chưa theo mẫu bắt buộc. Vui lòng sử dụng mẫu được cung cấp tại liên kết sau khi viết bài: https://www.ieee.org/conferences/publishing/templates.
2. **NHẬN XÉT 02:** Nên ghi đầy đủ tên đơn vị liên hệ của tác giả. Cụm “Dept. of Artificial Intelligence” nên được đổi thành “Department of Artificial Intelligence”.
3. **NHẬN XÉT 03:** Nên tránh dùng email cá nhân như Gmail nếu có email công việc (@fe.edu.vn).
4. **NHẬN XÉT 04:** Có thể cải thiện tóm tắt bằng cách viết cô đọng hơn. Nên giảm chi tiết kỹ thuật và tập trung vào điểm nổi bật của phương pháp, kết quả chính, tính mới, đóng góp khoa học/công nghệ và tác động của nghiên cứu.
5. **NHẬN XÉT 05:** Cần nâng chất lượng hình bằng cách tăng cỡ chữ và cải thiện chất lượng ảnh.
6. **NHẬN XÉT 06:** Có thể cải thiện đáng kể cấu trúc bài viết. Có thể gộp các Mục VII, VIII và IX thành một mục “THẢO LUẬN VÀ KẾT LUẬN”. Cấu trúc học thuật truyền thống được khuyến nghị gồm: Giới thiệu, Phương pháp/Phương pháp luận, Kết quả, Thảo luận và Kết luận.
7. **NHẬN XÉT 07:** Cần tính đến kết luận rút ra từ phép so sánh trong nghiên cứu: “YOLOv5n/YOLOv8n có thể vượt SCRFD trong vùng vận hành ưu tiên độ chính xác cao”.

   Các mô hình YOLO được fine-tune trên dữ liệu cùng miền, trong khi SCRFD được dùng nguyên trạng. Nên bổ sung phép so sánh bằng cách fine-tune SCRFD trên cùng 6.035 frame hoặc đối chiếu với các bộ phát hiện khuôn mặt chuyên biệt đã được tiền huấn luyện khác, chẳng hạn YOLOv8-face, YOLOv5-face, RetinaFace, MediaPipe/BlazeFace, YuNet và MTCNN.

### Chính sách nộp bài

PDF có liệt kê cùng tác giả, tiêu đề và tóm tắt như hồ sơ đăng ký EDAS không? (Có thể có khác biệt nhỏ về câu chữ trong tóm tắt.)

Có. Bài báo liệt kê cùng tác giả, tiêu đề và tóm tắt trong PDF và hồ sơ đăng ký EDAS.

## Review 3

### Tính mới và tính nguyên gốc

Công trình có tính nguyên gốc đáng kể và đưa ra kết quả mới. **(4)**

### Nội dung kỹ thuật và tính nghiêm cẩn khoa học

Công trình vững chắc và có tầm quan trọng đáng kể. **(4)**

### Chất lượng trình bày

Trình bày tốt. **(4)**

### Mức độ phù hợp và tính cập nhật

Tốt. **(4)**

### Điểm mạnh

- **Động cơ thực tiễn:** Bài báo tập trung trực tiếp vào nút thắt thực tế trong các quy trình giám sát, nơi dịch vụ nhận diện khuôn mặt có thể bị quá tải khi số người tăng đột biến.
- **Đánh giá thực nghiệm kỹ lưỡng:** Benchmark khảo sát có hệ thống 10 chiều rộng ảnh sau khi giảm mẫu, từ 640 px đến 128 px, cùng 6 ngưỡng tin cậy trên phần cứng thực (GPU NVIDIA GeForce RTX 3050 Laptop), với 5.676 frame đánh giá.
- **Thông tin hữu ích cho triển khai:** Bài báo cho thấy YOLO tiền huấn luyện trên COCO thất bại hoàn toàn nếu không fine-tune (AP50 = 0.000), xác định vùng vận hành thực tiễn 240–288 px, đồng thời cho thấy YOLOv12n-FT, vốn tập trung vào attention, suy giảm nhanh hơn YOLOv5n-FT và YOLOv8n-FT khi độ phân giải không gian bị nén mạnh.
- **Khuyến nghị có thể áp dụng:** Đưa ra hướng dẫn triển khai cụ thể theo ưu tiên vận hành: YOLOv8n-FT để cân bằng trong trường hợp thông thường, YOLOv5n-FT khi cần giảm độ phân giải mạnh, và SCRFD làm phương án dự phòng khi độ phân giải cực thấp.

### Điểm yếu

- **Nhãn hộp bao suy diễn:** Ground truth được tạo một cách xác định từ các mốc mắt (2.8d × 3.6d), thay vì từ hình chữ nhật khuôn mặt do người gán nhãn. Cách này có thể gây thiên lệch cho IoU và các chỉ số định vị.
- **Phạm vi dữ liệu đánh giá hạn chế:** Đánh giá chỉ dùng phân hoạch Portal 1 trong nhà của ChokePoint, chưa có kiểm chứng trên tập dữ liệu khác, bối cảnh ngoài trời hoặc tình huống che khuất nhiều.
- **Thiếu đánh giá nhận diện ở bước sau:** Bài báo chỉ xét chỉ số ở mức bộ phát hiện (AP50, recall, FPS), chưa đo tác động của khuôn mặt giảm độ phân giải lên trích xuất đặc trưng hoặc độ chính xác nhận dạng.

### Chính sách nộp bài

PDF có liệt kê cùng tác giả, tiêu đề và tóm tắt như hồ sơ đăng ký EDAS không? (Có thể có khác biệt nhỏ về câu chữ trong tóm tắt.)

*Bản review gốc không ghi câu trả lời cho câu hỏi này.*
