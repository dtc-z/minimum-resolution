from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "chokepoint_data" / "modeling"
OUT = ROOT / "DSP391M_Final_Project_Report.ipynb"

MANIFEST_CSV = MODELING / "chokepoint_p1_indoor_frontal_manifest.csv"
FIXED_CSV = MODELING / "benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv"
RESOLUTION_CSV = MODELING / "benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv"
FIG_DIR = MODELING / "ieee_paper_figures_en"

GROUP = "Nhóm 13"
STUDENT = "Vũ Hải Minh - HE194098"
SUPERVISOR = "Cao Văn Mai"
SUBMISSION_DATE = "18/7/2026"
TITLE_VI = (
    "Đánh giá điểm cắt độ phân giải cho phát hiện khuôn mặt thời gian thực "
    "trên camera an ninh"
)
TITLE_EN = "Minimum-Resolution Face Detection for Real-Time Surveillance: A ChokePoint Benchmark of YOLO and SCRFD"


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" for _ in headers]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_data_summary(manifest: pd.DataFrame) -> str:
    split = manifest.groupby("p1_indoor_split").agg(
        frames=("image_path", "count"),
        subjects=("person_id", "nunique"),
        median_face_width=("box_w", "median"),
        median_eye_distance=("eye_distance_px", "median"),
    ).round(2)
    rows = []
    for split_name, row in split.loc[["train_dev", "eval"]].iterrows():
        rows.append([
            split_name,
            int(row["frames"]),
            int(row["subjects"]),
            fmt(row["median_face_width"], 2),
            fmt(row["median_eye_distance"], 2),
        ])
    rows.append([
        "Tổng",
        len(manifest),
        manifest["person_id"].nunique(),
        fmt(manifest["box_w"].median(), 2),
        fmt(manifest["eye_distance_px"].median(), 2),
    ])
    return table(
        ["Split", "Số frame", "Số subject", "Median bbox width", "Median eye distance"],
        rows,
    )


def build_fixed_table(fixed: pd.DataFrame) -> str:
    rows = []
    order = ["YOLOv5n", "YOLOv8n", "YOLOv12n", "YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]
    indexed = fixed.set_index("model")
    for model in order:
        row = indexed.loc[model]
        rows.append([
            model,
            fmt(row["precision"]),
            fmt(row["recall"]),
            fmt(row["f1"]),
            fmt(row["ap50_voc101"]),
            fmt(row["fps"], 1),
            int(row["fp"]),
            int(row["fn"]),
        ])
    return table(["Model", "Precision", "Recall", "F1", "AP50", "FPS", "FP", "FN"], rows)


def build_cutoff_table(resolution: pd.DataFrame) -> str:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    rows = []
    for model in ["YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]:
        group = primary[primary["model"].eq(model)].sort_values("resolution_width")
        valid = group[(group["ap50_voc101"] >= 0.95) & (group["recall"] >= 0.95)]
        if valid.empty:
            rows.append([model, "Không đạt", "-", "-", "-", "-"])
            continue
        best = valid.iloc[0]
        rows.append([
            model,
            f"{int(best['resolution_width'])}x{int(best['resolution_height'])}",
            fmt(best["ap50_voc101"]),
            fmt(best["recall"]),
            fmt(best["f1"]),
            fmt(best["fps"], 1),
        ])
    return table(["Model", "Điểm cắt nhỏ nhất", "AP50", "Recall", "F1", "FPS"], rows)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_CSV)
    fixed = pd.read_csv(FIXED_CSV)
    resolution = pd.read_csv(RESOLUTION_CSV)

    data_summary = build_data_summary(manifest)
    fixed_table = build_fixed_table(fixed)
    cutoff_table = build_cutoff_table(resolution)

    cells = [
        md_cell(f"""
# BÁO CÁO TỔNG KẾT ĐỒ ÁN (FINAL PROJECT REPORT)
## Môn học: DSP391m - Data Science Capstone Project

**Tên đề án:** {TITLE_VI}  
**Tên tiếng Anh:** {TITLE_EN}

**Nhóm thực hiện:** {GROUP}  
- {STUDENT} (Trưởng nhóm)

**Giảng viên hướng dẫn:** {SUPERVISOR}  
**Ngày nộp:** {SUBMISSION_DATE}  
**Hà Nội, Năm 2026**
"""),
        md_cell("""
## TÓM TẮT (ABSTRACT)

Dự án nghiên cứu bài toán phát hiện khuôn mặt trong video camera an ninh dưới ràng buộc thời gian thực. Động lực thực tế xuất phát từ hiện tượng hệ thống nhận diện có thể bị nghẽn khi số lượng khuôn mặt trong khung hình tăng đột ngột, làm chi phí detect và nhận diện tăng nhanh. Vì vậy, mục tiêu của đồ án không chỉ là tìm mô hình có độ chính xác cao nhất, mà là xác định **điểm cắt độ phân giải**: mức giảm độ phân giải nhỏ nhất có thể sử dụng để tăng throughput nhưng vẫn giữ chất lượng phát hiện ở mức chấp nhận được.

Dữ liệu được sử dụng là ChokePoint, tập trung vào giao thức Portal 1 indoor frontal vì bối cảnh gần với camera giám sát trong nhà và có điều kiện ánh sáng tương đối ổn định. Do nhãn gốc của ChokePoint là vị trí hai mắt, dự án xây dựng protocol chuyển đổi landmark mắt thành bounding box khuôn mặt, sau đó dùng bbox này để fine-tune và đánh giá YOLOv5n, YOLOv8n, YOLOv12n cùng SCRFD. Benchmark được thực hiện trên toàn bộ eval split gồm 5,676 frame, với nhiều mức độ phân giải từ 640x480 xuống 128x96 và nhiều confidence threshold.

Kết quả cho thấy YOLO pretrained trên COCO không phù hợp trực tiếp cho face-box detection, nhưng sau fine-tune, YOLOv5n-FT, YOLOv8n-FT và YOLOv12n-FT đều đạt AP50 rất cao ở full resolution. YOLOv8n-FT là lựa chọn cân bằng nhất cho triển khai: tại 320x240, mô hình vẫn đạt AP50 xấp xỉ 0.990 và recall xấp xỉ 0.997 trong khi giảm 75% số pixel so với 640x480. Đồ án đề xuất 320x240 làm cấu hình mặc định, 288x216 làm ngưỡng tối thiểu hợp lý cho YOLOv8n-FT trong điều kiện tương tự ChokePoint Portal 1.

**Từ khóa (Keywords):** face detection, surveillance camera, ChokePoint, YOLO, SCRFD, resolution scaling, real-time inference.
"""),
        md_cell("""
## 1. GIỚI THIỆU (INTRODUCTION)

### 1.1. Bối cảnh và Động lực (Motivation)

Trong hệ thống nhận diện khuôn mặt trên camera an ninh, face detection là bước đầu tiên và có ảnh hưởng trực tiếp đến toàn bộ pipeline phía sau. Nếu detector bỏ sót khuôn mặt, module recognition không còn dữ liệu để trích xuất embedding. Nếu detector quá chậm, hàng đợi frame tăng và hệ thống có thể mất tính thời gian thực. Vấn đề này trở nên rõ rệt khi nhiều người cùng xuất hiện trong khung hình, đặc biệt tại cửa ra vào, sảnh văn phòng hoặc khu vực kiểm soát ra vào.

Một hướng tối ưu thực dụng là giảm độ phân giải đầu vào của detector. Tuy nhiên, giảm quá mạnh sẽ làm mất chi tiết khuôn mặt và làm recall giảm. Vì vậy, dự án tập trung tìm mối tương quan giữa độ phân giải, độ chính xác và throughput, từ đó đề xuất cấu hình vận hành có cơ sở thực nghiệm.

### 1.2. Phát biểu bài toán (Problem Statement)

**Đầu vào:** frame camera từ ChokePoint Portal 1, nhãn mắt thủ công, mô hình YOLO/SCRFD và danh sách mức độ phân giải cần benchmark.  
**Đầu ra:** độ chính xác phát hiện khuôn mặt, FPS, false positive/false negative và khuyến nghị độ phân giải triển khai.  
**Câu hỏi nghiên cứu:** độ phân giải có thể giảm đến đâu trước khi AP50 và recall không còn đạt ngưỡng chấp nhận được?

### 1.3. Đóng góp chính

- Xây dựng protocol ChokePoint-only để chuyển eye landmarks thành bbox khuôn mặt phục vụ detector benchmark.
- Fine-tune YOLOv5n, YOLOv8n và YOLOv12n trên bbox suy ra từ ChokePoint.
- Benchmark YOLO fine-tuned và SCRFD trên full eval split gồm 5,676 frame.
- Phân tích trade-off giữa độ phân giải, AP50, recall, false negative và FPS.
- Đề xuất cấu hình deployment cho camera an ninh thời gian thực.
"""),
        md_cell("""
## 2. TỔNG QUAN NGHIÊN CỨU (RELATED WORKS)

Các nghiên cứu face detection truyền thống như Viola-Jones chứng minh khả năng phát hiện khuôn mặt thời gian thực bằng đặc trưng Haar-like và cascade classifier. Tuy nhiên, các hệ thống hiện đại thường dùng deep learning detector vì khả năng tổng quát tốt hơn trước thay đổi về pose, scale và ánh sáng.

YOLO là họ detector một bước, tối ưu cho tốc độ suy luận và phù hợp với bài toán real-time. Trong đồ án này, YOLOv5n, YOLOv8n và YOLOv12n được chọn vì có kích thước nano, phù hợp với ràng buộc tốc độ của camera an ninh. Tuy nhiên, YOLO pretrained trên COCO không phải face detector chuyên biệt, nên cần fine-tune trên bbox khuôn mặt.

SCRFD là face detector chuyên biệt, được thiết kế theo hướng phân bổ lại sample và computation để tăng hiệu quả phát hiện khuôn mặt. Vì SCRFD đã được huấn luyện cho face detection, mô hình này đóng vai trò baseline mạnh để so sánh với YOLO fine-tuned.

ChokePoint là dataset video-based face recognition được thu ở các portal có camera giám sát. Dataset này phù hợp với mục tiêu đồ án vì có bối cảnh gần với camera an ninh thực tế, có nhãn mắt thủ công và có nhiều sequence camera để đánh giá ảnh hưởng của scale/pose theo thời gian.
"""),
        md_cell(f"""
## 3. CHUẨN BỊ DỮ LIỆU & EDA (DATA PREPARATION & EXPLORATION)

### 3.1. Thu thập và Làm sạch dữ liệu

Dữ liệu ChokePoint được tải và giải nén vào thư mục `chokepoint_data`. Đồ án sử dụng các ảnh full-frame từ Portal 1, bao gồm hai hướng di chuyển `P1E` và `P1L`. Các file XML ground truth chứa vị trí mắt trái/mắt phải theo từng frame. Vì mục tiêu benchmark là face detection, nhãn mắt được chuyển thành bbox khuôn mặt theo một công thức nhất quán dựa trên khoảng cách hai mắt.

Tập dữ liệu sau khi lọc theo giao thức Portal 1 indoor frontal có thống kê:

{data_summary}

Các bước tiền xử lý chính:

- kiểm tra frame có tồn tại trên ổ đĩa;
- map XML annotation với đường dẫn ảnh;
- loại các bản ghi không có ảnh hoặc thiếu landmark;
- tạo bbox từ eye distance;
- chia dữ liệu theo `train_dev` và `eval`;
- export sang định dạng YOLO để fine-tune detector.
"""),
        code_cell("""
from pathlib import Path
import json
import pandas as pd
from IPython.display import display

ROOT = Path(".")
MODELING_ROOT = ROOT / "chokepoint_data" / "modeling"
MANIFEST_CSV = MODELING_ROOT / "chokepoint_p1_indoor_frontal_manifest.csv"
FIXED_CSV = MODELING_ROOT / "benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv"
RESOLUTION_CSV = MODELING_ROOT / "benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv"

manifest_df = pd.read_csv(MANIFEST_CSV)
fixed_df = pd.read_csv(FIXED_CSV)
resolution_df = pd.read_csv(RESOLUTION_CSV)

print("Manifest:", manifest_df.shape)
print("Fixed benchmark:", fixed_df.shape)
print("Resolution benchmark:", resolution_df.shape)

display(
    manifest_df.groupby("p1_indoor_split").agg(
        frames=("image_path", "count"),
        subjects=("person_id", "nunique"),
        median_box_w=("box_w", "median"),
        median_eye_distance=("eye_distance_px", "median"),
    ).round(2)
)
"""),
        md_cell("""
### 3.2. Phân tích Khám phá (EDA Insights)

![Data distribution](chokepoint_data/modeling/report3_figures/report3_01_phan_bo_du_lieu.png)

![Face size distribution](chokepoint_data/modeling/report3_figures/report3_02_phan_bo_kich_thuoc_mat.png)

Các insight EDA quan trọng:

- Split `train_dev` và `eval` có quy mô tương đối cân bằng, giúp benchmark ổn định hơn so với sample nhỏ.
- Median bbox width khoảng 87-90 px ở ảnh gốc 800x600. Khi frame bị downsample xuống 320x240, median face width cũng giảm còn khoảng 35-36 px; khi xuống 240x180, median face width chỉ còn khoảng 26-27 px.
- Do bài toán phụ thuộc mạnh vào face scale, độ phân giải frame không nên được xem độc lập; cần xét kích thước khuôn mặt sau downsampling.
"""),
        md_cell("""
## 4. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

### 4.1. Kiến trúc hệ thống

Pipeline nghiên cứu gồm các bước:

1. Tải và giải nén ChokePoint.
2. Parse XML ground truth để lấy vị trí mắt.
3. Chuyển eye landmarks thành bbox khuôn mặt.
4. Export dữ liệu sang định dạng YOLO.
5. Fine-tune YOLOv5n, YOLOv8n và YOLOv12n trên `train_dev`.
6. Benchmark YOLO fine-tuned và SCRFD trên `eval`.
7. Lặp benchmark theo nhiều độ phân giải và threshold.
8. Phân tích AP50, recall, precision, F1, false positive, false negative và FPS.

### 4.2. Lựa chọn Mô hình

- **YOLOv5n:** baseline YOLO nano ổn định, tốc độ tốt.
- **YOLOv8n:** phiên bản YOLO nano hiện đại hơn, được kỳ vọng cân bằng tốt giữa speed và accuracy.
- **YOLOv12n:** mô hình mới hơn được đưa vào để kiểm tra khả năng thay thế trong cùng kích thước nano.
- **SCRFD:** face detector chuyên biệt, dùng làm baseline mạnh không cần fine-tune trên ChokePoint.

### 4.3. Metric đánh giá

Một detection được tính là true positive nếu IoU với bbox ground truth chưa match đạt ít nhất 0.50. Các metric chính:

- **AP50:** chất lượng detection ở ngưỡng IoU 0.50.
- **Recall:** tỷ lệ khuôn mặt được phát hiện, đặc biệt quan trọng cho recognition pipeline.
- **Precision:** tỷ lệ detection đúng trong các bbox dự đoán.
- **F1:** cân bằng precision và recall.
- **FPS/ms per image:** throughput và độ trễ suy luận.
- **FP/FN:** phân tích lỗi vận hành, trong đó FN thường nghiêm trọng hơn vì làm mất khuôn mặt trước bước recognition.
"""),
        code_cell("""
training_summary_path = MODELING_ROOT / "yolo_finetuned_full_gpu_weights" / "training_summary.json"
if training_summary_path.exists():
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    display(training_summary)
else:
    print("Không tìm thấy training_summary.json")
"""),
        md_cell("""
## 5. KẾT QUẢ VÀ THẢO LUẬN (RESULTS & DISCUSSION)

### 5.1. Đánh giá mô hình ở 640x480

Kết quả fixed benchmark trên toàn bộ eval split:

""" + fixed_table + """

Kết quả này cho thấy YOLO pretrained trên COCO không thể dùng trực tiếp cho face-box benchmark. Sau fine-tune, cả ba YOLO nano đều đạt AP50 = 1.000 ở full resolution. Trong nhóm này, YOLOv8n-FT có throughput tốt nhất và không tạo FP/FN trong benchmark fixed. SCRFD có precision/recall cao nhưng FPS thấp hơn tại 640x480.
"""),
        code_cell("""
display(
    fixed_df[["model", "fps", "precision", "recall", "f1", "ap50_voc101", "tp", "fp", "fn"]]
    .sort_values("model")
    .round(4)
)
"""),
        md_cell("""
### 5.2. Đường cong độ phân giải - độ chính xác

![AP50 vs resolution](chokepoint_data/modeling/ieee_paper_figures_en/fig01_ap50_vs_resolution.png)

![Recall vs resolution](chokepoint_data/modeling/ieee_paper_figures_en/fig02_recall_vs_resolution.png)

Các đường cong cho thấy vùng 480-640 px gần như an toàn cho YOLO fine-tuned. Khi giảm xuống 320x240, YOLOv8n-FT vẫn giữ AP50 và recall rất cao. Dưới 288x216, recall của YOLO bắt đầu giảm rõ hơn, đặc biệt ở 240x180 và thấp hơn. SCRFD giữ tốt hơn ở vùng độ phân giải rất thấp, nhưng không vượt trội về tốc độ ở full resolution.
"""),
        md_cell(f"""
### 5.3. Điểm cắt độ phân giải

Tiêu chí chấp nhận trong đồ án là **AP50 >= 0.95** và **Recall >= 0.95**.

{cutoff_table}

![Minimum acceptable width](chokepoint_data/modeling/ieee_paper_figures_en/fig06_minimum_acceptable_width.png)

**Kết luận vận hành:** YOLOv8n-FT tại 320x240 là cấu hình mặc định nên dùng vì có safety margin cao hơn 288x216. Nếu hệ thống chịu tải cao, 288x216 là mức tối thiểu hợp lý. YOLOv5n-FT có thể đạt tiêu chí tại 240x180, nhưng mức này sát ngưỡng và dễ rủi ro khi ánh sáng, pose hoặc mật độ khuôn mặt xấu hơn.
"""),
        md_cell("""
### 5.4. Trade-off accuracy và throughput

![Accuracy FPS tradeoff](chokepoint_data/modeling/ieee_paper_figures_en/fig02_accuracy_fps_tradeoff.png)

Biểu đồ trade-off cho thấy mục tiêu deployment không phải luôn là FPS cao nhất. Một cấu hình tốt cần giữ recall cao trước, sau đó mới tối ưu throughput. YOLOv8n-FT ở 320x240 giảm 75% số pixel so với 640x480 nhưng vẫn duy trì chất lượng detection cao, phù hợp với hệ thống camera an ninh cần xử lý liên tục.

![Effective face scale](chokepoint_data/modeling/ieee_paper_figures_en/fig05_effective_face_scale.png)

Phân tích theo effective face scale xác nhận nguyên nhân giảm accuracy: khi khuôn mặt sau downsampling còn quá ít pixel, detector khó duy trì localization ổn định. Điều này gợi ý rằng hệ thống thực tế nên theo dõi kích thước mặt trung vị trong frame, thay vì chỉ cố định một độ phân giải duy nhất.
"""),
        code_cell("""
primary = resolution_df[resolution_df["conf_threshold"].eq(0.25)].copy()
primary["resolution"] = primary["resolution_width"].astype(int).astype(str) + "x" + primary["resolution_height"].astype(int).astype(str)

deployment_view = primary[
    primary["model"].isin(["YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"])
    & primary["resolution_width"].isin([640, 480, 320, 288, 256, 240, 192, 160])
][["model", "resolution", "fps", "precision", "recall", "f1", "ap50_voc101", "fp", "fn"]]

display(deployment_view.sort_values(["model", "resolution"], ascending=[True, False]).round(4))
"""),
        md_cell("""
### 5.5. Threshold và error modes

![False negative rate](chokepoint_data/modeling/ieee_paper_figures_en/fig08_false_negative_rate.png)

![False positives per 100 frames](chokepoint_data/modeling/ieee_paper_figures_en/fig09_false_positives_per_100_frames.png)

Threshold không nên được chọn độc lập với độ phân giải. Ở độ phân giải thấp, tăng threshold có thể giảm false positive nhưng làm false negative tăng. Với face recognition pipeline, false negative thường nghiêm trọng hơn vì khuôn mặt bị loại trước khi recognition diễn ra. Do đó, cấu hình production nên ưu tiên recall và giám sát FN proxy trong runtime.

### 5.6. Nhận xét về YOLOv12n

YOLOv12n-FT đạt accuracy tốt ở độ phân giải cao, nhưng trong benchmark này chưa cho thấy trade-off tốt hơn YOLOv5n-FT và YOLOv8n-FT. Mô hình giảm recall mạnh ở low resolution và FPS thấp hơn YOLOv8n-FT. Đây là một hướng nghiên cứu tiếp theo: cần thử thêm hyperparameter, augmentation, epoch, seed và export/runtime backend trước khi kết luận chắc chắn về năng lực của YOLOv12n trong bài toán face detection độ phân giải thấp.
"""),
        md_cell("""
## 6. KẾT LUẬN & ĐỀ XUẤT (CONCLUSION & RECOMMENDATIONS)

### 6.1. Kết luận

Dự án đã hoàn thành mục tiêu xây dựng benchmark có khả năng trả lời câu hỏi thực tế: có thể giảm độ phân giải camera đến đâu mà face detection vẫn đủ tốt cho pipeline recognition. Kết quả thực nghiệm trên ChokePoint Portal 1 indoor frontal cho thấy:

- YOLO cần fine-tune để trở thành face detector phù hợp với ChokePoint.
- YOLOv8n-FT là mô hình khuyến nghị chính nhờ cân bằng tốt giữa AP50, recall và FPS.
- 320x240 là cấu hình mặc định phù hợp cho triển khai thời gian thực.
- 288x216 là ngưỡng tối thiểu hợp lý với YOLOv8n-FT trong điều kiện tương tự tập benchmark.
- SCRFD vẫn có giá trị như face-specific baseline và fallback ở low resolution.

### 6.2. Đề xuất triển khai

Luồng production nên detect ở 320x240 bằng YOLOv8n-FT, sau đó scale bbox về frame gốc để crop khuôn mặt cho module recognition. Khi tải hệ thống tăng mạnh, có thể hạ xuống 288x216 tạm thời. Khi khuôn mặt quá nhỏ hoặc scene khác đáng kể ChokePoint, hệ thống nên tự tăng resolution hoặc chuyển sang fallback detector.

### 6.3. Hạn chế và hướng phát triển

- Benchmark mới đánh giá detection, chưa đo end-to-end recognition accuracy.
- Ground truth bbox là bbox suy ra từ landmark mắt, không phải bbox thủ công.
- Dataset tập trung vào Portal 1 indoor frontal, cần kiểm tra thêm các bối cảnh khác.
- Cần export TensorRT/ONNX để đo latency production thật.
- Cần stress test với nhiều khuôn mặt trong một frame, đúng với tình huống gây quá tải ban đầu.
"""),
        code_cell("""
def choose_runtime_profile(face_count: int, queue_ms: float) -> dict:
    if queue_ms > 150 or face_count >= 12:
        return {
            "detector": "YOLOv8n-FT",
            "resolution": "288x216",
            "reason": "High load: reduce detector input while keeping the minimum acceptable profile.",
        }
    if face_count <= 4 and queue_ms < 60:
        return {
            "detector": "YOLOv8n-FT",
            "resolution": "480x360",
            "reason": "Low load: use extra resolution margin.",
        }
    return {
        "detector": "YOLOv8n-FT",
        "resolution": "320x240",
        "reason": "Default balanced profile.",
    }

for face_count, queue_ms in [(2, 30), (8, 80), (15, 180)]:
    print(face_count, "faces,", queue_ms, "ms queue ->", choose_runtime_profile(face_count, queue_ms))
"""),
        md_cell("""
## TÀI LIỆU THAM KHẢO & PHỤ LỤC (REFERENCES & APPENDICES)

### Tài liệu tham khảo

[1] P. Viola and M. J. Jones, "Robust Real-Time Face Detection," *International Journal of Computer Vision*, vol. 57, no. 2, pp. 137-154, 2004.  
[2] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," 2016.  
[3] J. Guo, J. Deng, A. Lattas, and S. Zafeiriou, "Sample and Computation Redistribution for Efficient Face Detection," 2021.  
[4] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, "Patch-based Probabilistic Image Quality Assessment for Face Selection and Improved Video-based Face Recognition," 2011.  
[5] ChokePoint Dataset, DOI: `10.5281/zenodo.815657`.

### Phụ lục A - Artifact bàn giao

- Notebook Report 2: `DSP391M_Report2_template.ipynb`
- Notebook Report 3: `DSP391M_Report3_template.ipynb`
- Notebook Report 4: `DSP391M_Report4_template.ipynb`
- Notebook Final Report: `DSP391M_Final_Project_Report.ipynb`
- Paper IEEE DOCX: `Paper_IEEE_ChokePoint_Resolution_Benchmark_Revised.docx`
- YOLO dataset: `chokepoint_data/modeling/yolo_face_dataset`
- Weights YOLO fine-tuned: `chokepoint_data/modeling/yolo_finetuned_full_gpu_weights`
- Benchmark CSV: `chokepoint_data/modeling/benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv`
- Resolution sweep CSV: `chokepoint_data/modeling/benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv`
- Figures: `chokepoint_data/modeling/ieee_paper_figures_en`

### Phụ lục B - Ghi chú tái lập

Để tái lập benchmark, chạy lần lượt:

1. `prepare_yolo_face_dataset.py`
2. `train_yolo_face_models.py`
3. `run_yolo_scrfd_benchmark.py`
4. `run_resolution_benchmark.py`

Các script đã được lưu trong `chokepoint_data/modeling`.
"""),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
