# 06 · Đánh giá, kiểm chứng và danh sách các lần chạy

*Lý do: file câu chuyện, mục 6.7 (kiến trúc đảm bảo gì và không đảm bảo gì) và mục 8 (hạn chế).*

## Mục tiêu

Tạo các bảng kết quả cho luận văn trên ba mặt:

1. **Chẩn đoán**, so với RAD.
2. **Mệnh đề**: điểm mệnh đề có mang thông tin vượt quá việc biết bệnh của ca không.
3. **Kiểm chứng**: các kiểm tra tuỳ chọn và bảng kiểm.

## Đầu vào

| Thứ | Nguồn |
|---|---|
| RAD (mốc so sánh) | `RAD-outputs/skin-rad-20ep/20261007_111918` (đã có) |
| Kiến trúc mệnh đề không có nhãn riêng (mốc cho C2) | `Proposition-outputs/skin-proposition-20ep/20261007_112004` (đã có) |
| Các run mới | `WeakSup/runs/...` (file 04, 05) |
| Mẫu chấm tay | `WeakSup/weak_labels/v<k>/audit/` (file 03) |

## Phần A · Chẩn đoán (dùng lại `K34_Results_Evaluation.ipynb`)

1. Thêm các run Stage 2 vào `PROP_RUNS` theo đúng cách các run cũ đang được khai báo. File dự đoán đã cùng format (khoá `pred_proposition`), nên không cần sửa logic notebook.
2. Chỉ số: top-1/2/3, MRR@k, macro-F1, balanced accuracy, macro-AUC, mAP. Tất cả lấy từ `final_metrics`.
3. So sánh từng run mới với **RAD fused** bằng paired bootstrap (B = 2000, seed 42) và McNemar, như các lần trước.
4. RAD và các run mới cùng đọc **caption gốc**, nên so sánh hợp lệ.

## Phần B · Mệnh đề (`WS_06_eval.ipynb`, dùng hàm trong `proposition/weak/metrics.py`)

| Chỉ số | Cách tính | So với |
|---|---|---|
| **C1** | AUC trong cùng bệnh (file 04, mục 6) | 0,5 (mô hình chỉ biết bệnh) |
| AUC toàn cục | AUC trên mọi ô có nhãn ở test | baseline "chỉ biết bệnh" (file 04, mục 6) |
| **C2a / C2b** | tương quan điểm / attention giữa các mệnh đề | **P1, tính lại bằng cùng hàm** |
| **C3** | `c3.json` của run `frozen` | phải `passed: true` |
| Độ trôi | `drift.json` của các run `unfrozen` | run `frozen` cùng mode tổng hợp |
| Đối chiếu chấm tay | trên các cặp của mẫu chấm tay (lấy từ test): precision và recall của trạng thái `có/không` của mô hình so với `human_label`; đặt cạnh precision của bộ gán nhãn (`audit_metrics.json`) | bộ gán nhãn |

Dòng cuối cho phép tách hai câu hỏi: "mô hình có khớp với bộ gán nhãn không" và "mô hình có đúng theo người chấm không".

## Phần C · Kiểm tra tuỳ chọn (chạy nếu kịp)

| Mã | Cách chạy | Kỳ vọng nếu mô hình đọc đúng dấu hiệu |
|---|---|---|
| **O1** xáo nhãn | Stage 1 với `--shuffle_labels`, rồi tính C1 | C1 tụt về khoảng 0,5 |
| **O2** đổi query | Ở bước inference, thay embedding query của mệnh đề A bằng của mệnh đề B trên cùng ca (vd. "nail pitting" ↔ "erythematous plaques"), đo $s$ thay đổi | $s$ đổi theo nội dung query, không giữ nguyên |
| **O3** chỉ một nguồn | Stage 1 với `--memory image` và `--memory text`, rồi tính C1 và AUC toàn cục | Biết bằng chứng đến từ đâu. Nếu `text` ≈ `both` thì **không** được gọi vết attention trên ảnh là bằng chứng thị giác (file câu chuyện, mục 8, hạn chế 7) |

## Phần D · Danh sách các lần chạy (theo thứ tự ưu tiên)

| # | Run | File | Ghi chú |
|---|---|---|---|
| R1 | **Chính:** nhãn `both` → Stage 1 → `frozen` + `mean` | 04, 05 | bắt buộc |
| R2 | R1 → `frozen` + `weighted` | 05 | rẻ: chỉ đọc lại $S$ |
| R3 | R1 → `frozen` + `group` | 05 | rẻ |
| R4 | R1 → `unfrozen`, λ = 1 | 05 | so độ trôi với R1 |
| R5 | R1 → `unfrozen`, λ = 0 | 05 | xem nhãn bệnh có kéo $s$ về theo bệnh không |
| R6 | Nhãn `text` (không dùng ConceptCLIP) → Stage 1 → `frozen` + `mean` | 03, 04, 05 | ablation bộ gán nhãn |
| R7 | O1 xáo nhãn | 04 | tuỳ chọn |
| R8, R9 | O3 `image` / `text` | 04 | tuỳ chọn |
| R10 | Che tên bệnh cho cả student và RAD | — | để sau, nếu có thời gian |

Mỗi run ghi một dòng vào `WeakSup/runs/RUNS.md`: tên, timestamp, phiên bản nhãn, các cờ, đường dẫn kết quả.

## Phần E · Bảng kết quả cần xuất

1. `results_diagnosis.csv`: các chỉ số chẩn đoán của RAD và R1–R6, kèm cột p-value của bootstrap và McNemar so với RAD.
2. `results_concepts.csv`: C1, AUC toàn cục, C2a, C2b, C3 cho R1–R9, kèm hàng P1 cho C2.
3. `results_audit.csv`: precision của bộ gán nhãn và của mô hình trên mẫu chấm tay.
4. `results_drift.csv`: so sánh R4, R5 với R1.

Chỉ xuất số, không viết diễn giải. Phần diễn giải viết sau, khi đã có đủ kết quả.

## Phần F · Case viewer

Tạo trình xem ca cho R1, đọc từ `checklist/checklist_test.json` và `attention_test.npz`. Giao diện sẽ được bàn riêng khi tới bước này; giai đoạn này chỉ cần đảm bảo dữ liệu đã đủ.

## Xong khi

- Đã có bốn file ở phần E cho ít nhất R1–R5.
- Trí đã xem và chọn những run sẽ đưa vào luận văn.

## Không được làm

- Dùng test để chọn run hay chỉnh tham số.
- Báo cáo chỉ số mệnh đề tính trên nhãn yếu như thể đó là ground truth. Khi ghi kết quả, luôn ghi rõ "so với nhãn yếu" hay "so với nhãn chấm tay".
