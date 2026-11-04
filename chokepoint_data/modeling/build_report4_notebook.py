from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "chokepoint_data" / "modeling"
OUT = ROOT / "DSP391M_Report4_template.ipynb"

FIXED_CSV = MODELING / "benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv"
RESOLUTION_CSV = MODELING / "benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv"
CONFIG_JSON = MODELING / "report4_deployment_config.json"

PROJECT_TITLE = (
    "Minimum-Resolution Face Detection for Real-Time Surveillance: "
    "A ChokePoint Benchmark of YOLO and SCRFD"
)
STUDENT = "Vũ Hải Minh - HE194098"
GROUP = "Nhóm 13"
DUE_DATE = "18/7/2026"
SUPERVISOR = "Cao Văn Mai"


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" for _ in headers]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def fixed_summary_table(fixed: pd.DataFrame) -> str:
    order = ["YOLOv5n", "YOLOv8n", "YOLOv12n", "YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]
    rows = []
    for _, row in fixed.set_index("model").loc[order].reset_index().iterrows():
        rows.append([
            row["model"],
            fmt(row["precision"]),
            fmt(row["recall"]),
            fmt(row["f1"]),
            fmt(row["ap50_voc101"]),
            fmt(row["fps"], 1),
            int(row["fp"]),
            int(row["fn"]),
        ])
    return markdown_table(
        ["Model", "Precision", "Recall", "F1", "AP50", "FPS", "FP", "FN"],
        rows,
    )


def cutoff_table(resolution: pd.DataFrame) -> tuple[str, list[dict]]:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    rows: list[list[object]] = []
    records: list[dict] = []
    for model in ["YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]:
        group = primary[primary["model"].eq(model)].sort_values("resolution_width")
        valid = group[(group["ap50_voc101"] >= 0.95) & (group["recall"] >= 0.95)]
        if valid.empty:
            rows.append([model, "Không đạt", "-", "-", "-", "-"])
            continue
        best = valid.iloc[0]
        record = {
            "model": model,
            "resolution_width": int(best["resolution_width"]),
            "resolution_height": int(best["resolution_height"]),
            "ap50": float(best["ap50_voc101"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "fps": float(best["fps"]),
        }
        records.append(record)
        rows.append([
            model,
            f"{record['resolution_width']}x{record['resolution_height']}",
            fmt(record["ap50"]),
            fmt(record["recall"]),
            fmt(record["f1"]),
            fmt(record["fps"], 1),
        ])
    return markdown_table(
        ["Model", "Điểm cắt nhỏ nhất", "AP50", "Recall", "F1", "FPS"],
        rows,
    ), records


def selected_resolution_table(resolution: pd.DataFrame) -> str:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    selected = primary[
        primary["model"].isin(["YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"])
        & primary["resolution_width"].isin([640, 320, 288, 256, 240, 192, 160])
    ].copy()
    selected["resolution"] = selected["resolution_width"].astype(int).astype(str) + "x" + selected["resolution_height"].astype(int).astype(str)
    rows = []
    for _, row in selected.sort_values(["model", "resolution_width"], ascending=[True, False]).iterrows():
        rows.append([
            row["model"],
            row["resolution"],
            fmt(row["ap50_voc101"]),
            fmt(row["recall"]),
            fmt(row["precision"]),
            fmt(row["fps"], 1),
            int(row["fp"]),
            int(row["fn"]),
        ])
    return markdown_table(
        ["Model", "Resolution", "AP50", "Recall", "Precision", "FPS", "FP", "FN"],
        rows,
    )


def build_config(cutoffs: list[dict]) -> dict:
    return {
        "project_title": PROJECT_TITLE,
        "student": STUDENT,
        "group": GROUP,
        "due_date": DUE_DATE,
        "recommended_default": {
            "detector": "YOLOv8n-FT",
            "frame_width": 320,
            "frame_height": 240,
            "confidence_threshold": 0.25,
            "reason": (
                "320x240 keeps AP50 and recall above 0.99 on the full Portal 1 eval split "
                "while reducing input pixels by 75% compared with 640x480."
            ),
        },
        "minimum_acceptable_by_model": cutoffs,
        "artifacts": {
            "yolov5n_ft": str(MODELING / "yolo_finetuned_full_gpu_weights" / "yolov5n_face.pt"),
            "yolov8n_ft": str(MODELING / "yolo_finetuned_full_gpu_weights" / "yolov8n_face.pt"),
            "yolov12n_ft": str(MODELING / "yolo_finetuned_full_gpu_weights" / "yolov12n_face.pt"),
            "fixed_benchmark": str(FIXED_CSV),
            "resolution_benchmark": str(RESOLUTION_CSV),
            "paper_figures": str(MODELING / "ieee_paper_figures_en"),
            "report3_figures": str(MODELING / "report3_figures"),
        },
    }


def main() -> None:
    fixed = pd.read_csv(FIXED_CSV)
    resolution = pd.read_csv(RESOLUTION_CSV)
    fixed_md = fixed_summary_table(fixed)
    cutoff_md, cutoffs = cutoff_table(resolution)
    selected_md = selected_resolution_table(resolution)

    config = build_config(cutoffs)
    CONFIG_JSON.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    cells = [
        md_cell(f"""
# DSP391m - DATA SCIENCE CAPSTONE PROJECT
## FINAL REPORT: RESULTS INTERPRETATION, DEPLOYMENT & CONCLUSION

**Tên nhóm:** {GROUP}  
**Sinh viên:** {STUDENT}  
**Giảng viên hướng dẫn:** {SUPERVISOR}  
**Đề tài:** {PROJECT_TITLE}  
**Ngày nộp:** {DUE_DATE}

Report 4 tổng hợp kết quả cuối cùng của dự án benchmark phát hiện khuôn mặt trên ChokePoint. Trọng tâm của báo cáo là diễn giải mối quan hệ giữa **độ phân giải đầu vào**, **độ chính xác phát hiện**, và **tốc độ xử lý**, từ đó đề xuất cấu hình triển khai phù hợp cho camera an ninh thời gian thực.
"""),
        md_cell("""
## I. DIỄN GIẢI KẾT QUẢ & TRỰC QUAN HÓA (RESULTS INTERPRETATION)

### 1.1. Bài toán triển khai thực tế

Động lực chính của dự án đến từ một vấn đề vận hành: khi hệ thống nhận diện khuôn mặt chạy trên camera an ninh gặp số lượng khuôn mặt tăng đột ngột trong cùng khung hình, tải suy luận tăng nhanh và có thể làm pipeline bị nghẽn hoặc sập. Vì vậy, thay vì chỉ hỏi mô hình nào chính xác nhất ở ảnh gốc, dự án đặt câu hỏi triển khai:

> Có thể giảm độ phân giải khung hình đến mức nào để tăng tốc xử lý nhưng vẫn giữ chất lượng phát hiện khuôn mặt ở mức chấp nhận được?

ChokePoint phù hợp cho bài toán này vì dữ liệu được thu từ các camera giám sát ở khu vực cổng đi lại, có nhãn mắt thủ công và có ảnh full-frame từ nhiều camera. Trong dự án này, nhãn mắt được chuyển thành bounding box khuôn mặt bằng một protocol nhất quán để benchmark YOLOv5n, YOLOv8n, YOLOv12n và SCRFD.
"""),
        code_cell("""
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path(".")
MODELING_ROOT = ROOT / "chokepoint_data" / "modeling"
FIXED_CSV = MODELING_ROOT / "benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv"
RESOLUTION_CSV = MODELING_ROOT / "benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv"
CONFIG_JSON = MODELING_ROOT / "report4_deployment_config.json"

fixed_df = pd.read_csv(FIXED_CSV)
resolution_df = pd.read_csv(RESOLUTION_CSV)

print("Fixed benchmark:", fixed_df.shape, FIXED_CSV)
print("Resolution benchmark:", resolution_df.shape, RESOLUTION_CSV)
display(fixed_df[["model", "fps", "precision", "recall", "f1", "ap50_voc101", "tp", "fp", "fn"]].round(4))
"""),
        md_cell(f"""
### 1.2. Kết quả benchmark cố định ở 640x480

Benchmark cố định trên toàn bộ eval split gồm **5,676 frame** cho thấy YOLO pretrained trên COCO gần như không dùng được trực tiếp cho bài toán face-box detection, vì class set và phân bố bbox không khớp với khuôn mặt trong ChokePoint. Sau fine-tune trên bbox suy ra từ landmark mắt, các mô hình YOLO nano đạt AP50 rất cao. SCRFD vẫn là baseline chuyên biệt mạnh, nhưng chậm hơn ở full resolution trên cấu hình GPU sử dụng trong thí nghiệm.

{fixed_md}

**Diễn giải chính:** YOLOv8n-FT là cấu hình cân bằng nhất ở 640x480: AP50 = 1.000, recall = 1.000, không có false positive/false negative trong benchmark fixed và đạt khoảng 75.6 FPS. YOLOv12n-FT đạt độ chính xác cao sau fine-tune nhưng throughput thấp hơn, vì vậy chưa phải lựa chọn tốt nhất cho deployment thời gian thực trong project này.
"""),
        md_cell("""
### 1.3. Explainable AI theo hướng sensitivity analysis

Với bài toán object detection, các kỹ thuật như SHAP/feature importance cho dữ liệu bảng không phản ánh trực tiếp quyết định của detector. Vì vậy, report này dùng **sensitivity analysis** để giải thích mô hình:

- **Resolution sensitivity:** khi frame bị downsample, số pixel mô tả khuôn mặt giảm, làm landmark/biên mặt khó phân biệt hơn.
- **Threshold sensitivity:** confidence threshold cao giúp giảm false positive nhưng có thể tăng false negative, đặc biệt ở độ phân giải thấp.
- **Face-scale sensitivity:** yếu tố giải thích quan trọng không chỉ là frame width, mà là kích thước khuôn mặt sau khi downsample.
- **Error-mode analysis:** trong hệ thống nhận diện khuôn mặt, false negative thường nguy hiểm hơn false positive vì khuôn mặt bị mất trước khi đi vào bước recognition.
"""),
        md_cell("""
### 1.4. Đường cong độ phân giải và độ chính xác

![AP50 vs resolution](chokepoint_data/modeling/ieee_paper_figures_en/fig01_ap50_vs_resolution.png)

**Insight:** Các mô hình YOLO sau fine-tune giữ AP50 gần tuyệt đối ở vùng 480-640 px, nhưng bắt đầu rơi mạnh khi frame width xuống dưới khoảng 288-240 px. SCRFD giảm chậm hơn ở vùng rất thấp, cho thấy detector chuyên biệt cho face có khả năng chịu downsampling tốt hơn trong trường hợp thiếu pixel.

![Recall vs resolution](chokepoint_data/modeling/ieee_paper_figures_en/fig02_recall_vs_resolution.png)

**Insight:** Recall là metric cần ưu tiên cho camera an ninh vì một khuôn mặt bị miss sẽ không thể được nhận diện ở bước sau. YOLOv8n-FT ở 320x240 vẫn giữ recall xấp xỉ 0.997, trong khi 288x216 là ngưỡng tối thiểu còn chấp nhận được theo tiêu chí AP50 >= 0.95 và recall >= 0.95.
"""),
        md_cell("""
### 1.5. Trade-off accuracy và throughput

![Accuracy FPS tradeoff](chokepoint_data/modeling/ieee_paper_figures_en/fig02_accuracy_fps_tradeoff.png)

**Insight:** Cấu hình deployment không nên chọn độ phân giải nhỏ nhất một cách máy móc. 240x180 có thể đạt tiêu chí tối thiểu với YOLOv5n-FT nhưng safety margin thấp. 320x240 với YOLOv8n-FT là điểm cân bằng tốt hơn: giảm 75% số pixel so với 640x480 nhưng vẫn giữ AP50 và recall rất cao, đồng thời có throughput tốt hơn.
"""),
        md_cell(f"""
### 1.6. Điểm cắt độ phân giải theo từng mô hình

Tiêu chí điểm cắt trong report này là **AP50 >= 0.95** và **Recall >= 0.95** tại confidence threshold 0.25.

{cutoff_md}

![Minimum acceptable width](chokepoint_data/modeling/ieee_paper_figures_en/fig06_minimum_acceptable_width.png)

**Khuyến nghị vận hành:**  
YOLOv8n-FT tại **320x240** nên được chọn làm cấu hình mặc định vì có biên an toàn tốt hơn 288x216. Nếu bắt buộc phải giảm mạnh tài nguyên, YOLOv8n-FT tại **288x216** là mức tối thiểu hợp lý. YOLOv5n-FT tại **240x180** là mức nhỏ nhất đạt tiêu chí nhưng dễ bị rủi ro khi điều kiện thực tế xấu hơn ChokePoint Portal 1.
"""),
        md_cell("""
### 1.7. Ảnh hưởng của threshold và lỗi phát hiện

![False negative rate](chokepoint_data/modeling/ieee_paper_figures_en/fig08_false_negative_rate.png)

![False positives per 100 frames](chokepoint_data/modeling/ieee_paper_figures_en/fig09_false_positives_per_100_frames.png)

**Insight:** Khi độ phân giải thấp, tăng threshold có thể làm precision nhìn tốt hơn nhưng làm false negative tăng. Với bài toán recognition sau detection, false negative có chi phí cao hơn vì khuôn mặt không được đưa vào bước xác thực. Do đó, threshold nên được chọn cùng với độ phân giải, không nên tối ưu riêng lẻ.

![Effective face scale](chokepoint_data/modeling/ieee_paper_figures_en/fig05_effective_face_scale.png)

**Insight:** Độ phân giải frame chỉ là biến quan sát bên ngoài. Biến trực tiếp chi phối detector là **kích thước khuôn mặt sau downsampling**. Điều này gợi ý một hướng triển khai tốt hơn: hệ thống có thể tự điều chỉnh resolution dựa trên median face size hoặc mật độ khuôn mặt trong frame.
"""),
        code_cell("""
primary = resolution_df[resolution_df["conf_threshold"].eq(0.25)].copy()
primary["resolution"] = primary["resolution_width"].astype(int).astype(str) + "x" + primary["resolution_height"].astype(int).astype(str)

selected = primary[
    primary["model"].isin(["YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"])
    & primary["resolution_width"].isin([640, 320, 288, 256, 240, 192, 160])
].copy()

display(
    selected.sort_values(["model", "resolution_width"], ascending=[True, False])[
        ["model", "resolution", "ap50_voc101", "recall", "precision", "fps", "fp", "fn"]
    ].round(4)
)
"""),
        md_cell(f"""
### 1.8. Bảng chọn lọc cho quyết định triển khai

{selected_md}
"""),
        md_cell("""
## II. ĐÓNG GÓI VÀ TRIỂN KHAI (SYSTEM INTEGRATION & DEPLOYMENT)

### 2.1. Artifact bàn giao

Các artifact chính đã sẵn sàng trong workspace:

- Fine-tuned YOLOv5n: `chokepoint_data/modeling/yolo_finetuned_full_gpu_weights/yolov5n_face.pt`
- Fine-tuned YOLOv8n: `chokepoint_data/modeling/yolo_finetuned_full_gpu_weights/yolov8n_face.pt`
- Fine-tuned YOLOv12n: `chokepoint_data/modeling/yolo_finetuned_full_gpu_weights/yolov12n_face.pt`
- Benchmark cố định full eval: `chokepoint_data/modeling/benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv`
- Benchmark độ phân giải full eval: `chokepoint_data/modeling/benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv`
- Config triển khai tóm tắt: `chokepoint_data/modeling/report4_deployment_config.json`
- Biểu đồ paper/report: `chokepoint_data/modeling/ieee_paper_figures_en` và `chokepoint_data/modeling/report3_figures`
"""),
        code_cell("""
import json
from pathlib import Path

deployment_config = {
    "recommended_default": {
        "detector": "YOLOv8n-FT",
        "weights": str(MODELING_ROOT / "yolo_finetuned_full_gpu_weights" / "yolov8n_face.pt"),
        "frame_width": 320,
        "frame_height": 240,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.50,
        "expected_full_eval": {
            "ap50": 0.9901,
            "recall": 0.9967,
            "precision": 0.9996,
            "fps": 69.0,
        },
    },
    "minimum_viable": {
        "detector": "YOLOv8n-FT",
        "frame_width": 288,
        "frame_height": 216,
        "condition": "Use only when compute pressure is high and the scene is similar to ChokePoint Portal 1 indoor frontal.",
    },
    "fallback_low_resolution": {
        "detector": "SCRFD",
        "condition": "Consider when frames are heavily downsampled, but monitor false positives and latency.",
    },
}

CONFIG_JSON.write_text(json.dumps(deployment_config, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved deployment config:", CONFIG_JSON)
display(deployment_config)
"""),
        md_cell("""
### 2.2. Luồng suy luận đề xuất

Luồng triển khai cho camera an ninh:

1. Nhận frame gốc từ camera.
2. Resize frame về cấu hình mặc định **320x240**.
3. Chạy YOLOv8n-FT để phát hiện bbox khuôn mặt.
4. Scale bbox về tọa độ frame gốc nếu cần crop ở độ phân giải cao hơn.
5. Gửi crop khuôn mặt sang module recognition/verification.
6. Theo dõi số lượng khuôn mặt, FPS, queue length và false-negative proxy để tự động tăng/giảm resolution.

Điểm quan trọng là detection resolution không nhất thiết phải bằng recognition crop resolution. Hệ thống có thể detect ở 320x240 để giảm tải, sau đó crop vùng mặt từ frame gốc nếu cần embedding chất lượng cao hơn.
"""),
        code_cell("""
def choose_runtime_profile(face_count: int, gpu_queue_ms: float) -> dict:
    \"\"\"Simple rule-based profile selector for the proposed deployment pipeline.\"\"\"
    if gpu_queue_ms > 150 or face_count >= 12:
        return {
            "detector": "YOLOv8n-FT",
            "resolution": (288, 216),
            "conf_threshold": 0.25,
            "note": "High load: use minimum acceptable YOLOv8n-FT profile.",
        }
    if face_count <= 4 and gpu_queue_ms < 60:
        return {
            "detector": "YOLOv8n-FT",
            "resolution": (480, 360),
            "conf_threshold": 0.25,
            "note": "Low load: use higher resolution for extra safety margin.",
        }
    return {
        "detector": "YOLOv8n-FT",
        "resolution": (320, 240),
        "conf_threshold": 0.25,
        "note": "Default balanced profile.",
    }

for face_count, queue_ms in [(2, 30), (8, 80), (15, 180)]:
    print(face_count, "faces,", queue_ms, "ms queue ->", choose_runtime_profile(face_count, queue_ms))
"""),
        md_cell("""
### 2.3. Kiểm tra tính sẵn sàng của artifact

Cell dưới đây giúp kiểm tra nhanh các file cần bàn giao trước khi nộp hoặc demo.
"""),
        code_cell("""
required_paths = [
    MODELING_ROOT / "yolo_finetuned_full_gpu_weights" / "yolov5n_face.pt",
    MODELING_ROOT / "yolo_finetuned_full_gpu_weights" / "yolov8n_face.pt",
    MODELING_ROOT / "yolo_finetuned_full_gpu_weights" / "yolov12n_face.pt",
    FIXED_CSV,
    RESOLUTION_CSV,
    MODELING_ROOT / "ieee_paper_figures_en",
    MODELING_ROOT / "report3_figures",
]

artifact_status = pd.DataFrame(
    {
        "path": [str(path) for path in required_paths],
        "exists": [path.exists() for path in required_paths],
        "size_mb": [round(path.stat().st_size / (1024 * 1024), 2) if path.is_file() else None for path in required_paths],
    }
)
display(artifact_status)
"""),
        md_cell("""
## III. KẾT LUẬN & KHUYẾN NGHỊ (CONCLUSION & RECOMMENDATIONS)

### 3.1. Mức độ hoàn thành mục tiêu

Dự án đã hoàn thành mục tiêu chính: xây dựng một protocol benchmark trên ChokePoint để đánh giá mối quan hệ giữa độ phân giải, độ chính xác và throughput của các detector khuôn mặt nhẹ. Kết quả không chỉ cho biết mô hình nào tốt nhất ở full resolution, mà còn xác định được vùng resolution có thể dùng trong triển khai thực tế.

Kết luận chính:

- YOLO pretrained trên COCO không phù hợp để detect face box trực tiếp trên ChokePoint.
- Fine-tune YOLO bằng bbox suy ra từ landmark mắt giúp YOLOv5n, YOLOv8n và YOLOv12n đạt AP50 rất cao ở full resolution.
- YOLOv8n-FT là lựa chọn mặc định tốt nhất cho hệ thống thời gian thực trong project này.
- 320x240 là cấu hình triển khai khuyến nghị vì giữ accuracy cao trong khi giảm đáng kể số pixel xử lý.
- 288x216 là ngưỡng tối thiểu hợp lý cho YOLOv8n-FT; dưới ngưỡng này, recall bắt đầu giảm rõ rệt.
- SCRFD hữu ích như baseline chuyên biệt và fallback ở độ phân giải thấp, nhưng không phải cấu hình nhanh nhất ở full resolution.
"""),
        md_cell("""
### 3.2. Hạn chế của dự án

Các hạn chế cần ghi nhận:

- Ground truth gốc của ChokePoint là landmark mắt, không phải bbox khuôn mặt thủ công. Bbox dùng trong benchmark là bbox suy ra từ eye distance, vì vậy AP50 đo mức khớp với protocol chuyển đổi chứ không phải nhãn bbox gốc.
- Report tập trung vào Portal 1 indoor frontal; kết quả có thể thay đổi ở Portal 2, outdoor-like lighting, hoặc camera angle khác.
- Benchmark hiện đo detection, chưa đánh giá end-to-end recognition accuracy sau khi crop và trích xuất embedding.
- Chưa có stress test với nhiều khuôn mặt đồng thời trong một frame ngoài phân bố ChokePoint.
- YOLOv12n-FT cho kết quả accuracy tốt ở độ phân giải cao nhưng throughput và độ ổn định ở low resolution chưa cạnh tranh với YOLOv5n-FT/YOLOv8n-FT.
"""),
        md_cell("""
### 3.3. Hướng phát triển tiếp theo

Các hướng có giá trị nghiên cứu và triển khai:

- Đánh giá end-to-end face recognition: detection -> crop -> embedding -> verification/identification.
- Chạy thêm các sequence đông người hoặc dữ liệu camera văn phòng thực tế để kiểm tra crowd burst.
- Export YOLOv8n-FT sang ONNX/TensorRT để đo latency triển khai thật.
- Xây dựng adaptive resolution controller dựa trên face count, queue length, GPU memory và median face size.
- Fine-tune thêm YOLOv12n hoặc thử cấu hình hyperparameter khác để giải thích vì sao trade-off hiện tại kém hơn YOLOv5n/YOLOv8n.
- Bổ sung tracking để giảm số lần detect full frame, ví dụ detect mỗi N frame và track bằng ByteTrack/DeepSORT giữa các frame.
"""),
        md_cell("""
## IV. PHỤ LỤC & TÀI LIỆU BÀN GIAO (APPENDICES)

### 4.1. Source code và notebook

- Report 2: `DSP391M_Report2_template.ipynb`
- Report 3: `DSP391M_Report3_template.ipynb`
- Report 4: `DSP391M_Report4_template.ipynb`
- Chuẩn bị YOLO dataset: `chokepoint_data/modeling/prepare_yolo_face_dataset.py`
- Fine-tune YOLO: `chokepoint_data/modeling/train_yolo_face_models.py`
- Benchmark fixed resolution: `chokepoint_data/modeling/run_yolo_scrfd_benchmark.py`
- Benchmark resolution sweep: `chokepoint_data/modeling/run_resolution_benchmark.py`
- Paper IEEE DOCX: `Paper_IEEE_ChokePoint_Resolution_Benchmark_Revised.docx`

### 4.2. Tài liệu tham khảo

- Viola, P., & Jones, M. J. (2004). Robust real-time face detection. *International Journal of Computer Vision*.
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection.
- Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2021). Sample and Computation Redistribution for Efficient Face Detection.
- Wong, Y., Chen, S., Mau, S., Sanderson, C., & Lovell, B. C. (2011). Patch-based probabilistic image quality assessment for face selection and improved video-based face recognition.
- ChokePoint Dataset, DOI: `10.5281/zenodo.815657`.
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
    print(f"Wrote {CONFIG_JSON}")


if __name__ == "__main__":
    main()
