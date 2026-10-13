# 05 · Stage 2: tầng quyết định và các chế độ đóng băng

*Lý do: file câu chuyện, mục 6.4 (vì sao đóng băng), 6.5 (Stage 2) và 6.6 (bảng kiểm).*

## Mục tiêu

Học cách kết hợp điểm mệnh đề $s_{ij}$ thành xác suất bệnh. Có **hai chế độ**:

- `frozen`: chế độ chính. Tầng bằng chứng đóng băng hoàn toàn.
- `unfrozen`: chế độ nghiên cứu, để so sánh với `frozen`.

Đầu ra cùng format với RAD/P1, để dùng lại được notebook đánh giá.

## Đầu vào

- `runs/<run_name>/<timestamp>/stage1/`: `best.pt`, `scores_{train,val,test}.npz`, `param_hash.json` (file 04).
- `propositions_final.json` (polarity, category, disease của từng mệnh đề).
- Nhãn bệnh: cột `gt` trong `scores_*.npz`.

## Code mới

| File | Nội dung |
|---|---|
| `proposition/weak/aggregation.py` | `ActiveAggregator(prop_disease, prop_group, num_diseases, mode)` với `mode ∈ {mean, weighted, group}`. Mọi mode chỉ tổng hợp trên **tập tổng hợp $B_d^*$** (định nghĩa bên dưới), theo đúng thứ tự cột của $S$ |
| `proposition/weak/model.py` (thêm) | `DecisionHead(aggregator, calibration)`: polarity → aggregation → `PositiveCalibration` (import từ P1, không sửa) |
| `main_ws_stage2.py` | entrypoint |

### Tập tổng hợp $B_d^*$

$B_d^*$ gồm các mệnh đề quan sát được của bệnh *d* **có ít nhất 1 nhãn (0 hoặc 1) trong train**, tính trên `label_final` sau bước LLM (file 03).

- Mệnh đề quan sát được nhưng chưa có nhãn nào trong train **vẫn có điểm $s$** (để hiển thị trong bảng kiểm), nhưng **không tham gia** tính điểm bệnh, vì điểm của nó chưa từng được dạy.
- Nếu $B_d^*$ của một bệnh bị rỗng, dùng mọi mệnh đề quan sát được của bệnh đó và ghi cờ `fallback=true` cho bệnh đó vào `run_config.json`.
- Ghi danh sách $B_d^*$ ra `aggregation_set.json` (theo từng bệnh: số mệnh đề, theo nhóm, cờ `fallback`).

### Các mode tổng hợp

| `--agg` | Công thức trong từng bệnh *d* | Tham số học |
|---|---|---|
| `mean` (chính) | $z_{id}=\frac{1}{\lvert B_d^*\rvert}\sum_{j\in B_d^*}\tilde s_{ij}$ | không có |
| `weighted` | $z_{id}=\sum_{j\in B_d^*}\mathrm{softmax}_{j\in B_d^*}(a_j)\,\tilde s_{ij}$ | $a_j$ cho từng mệnh đề |
| `group` | như `weighted`, nhưng $a_j=a_{g(j)}$ dùng chung theo nhóm *g* (morphology / location / symptom / history) | $a_g$, 4 tham số |

Sau đó: $\hat y_{id}=\sigma(\gamma_d z_{id}+b_d)$ với $\gamma_d>0$, dùng `PositiveCalibration` của P1.

## Chế độ `--freeze_mode frozen` (chế độ chính)

1. **Không nạp ảnh hay bộ mã hoá.** Chỉ đọc `scores_{train,val,test}.npz`.
2. Tính $\tilde s$ từ $S$ và polarity, rồi train `DecisionHead` bằng BCE với nhãn bệnh. Tham số được học: $\gamma_d, b_d$, cộng thêm $a_j$ hoặc $a_g$ nếu dùng mode có trọng số.
3. Optimizer AdamW, lr `1e-2`, chạy cả batch (full-batch) hoặc batch 256, tối đa 300 epoch, early stopping theo **val macro-F1** với patience 30. Tất cả đặt trong `configs/ws_stage2.yaml`.
4. Chọn mô hình theo **val macro-F1**, cùng quy ước với P1/RAD.
5. **C3: kiểm tra đóng băng.**
   - (a) Tính lại `param_hash` của Stage-1 `best.pt` lúc bắt đầu và lúc kết thúc; hai giá trị phải trùng.
   - (b) Nạp Stage-1 `best.pt` ở `eval()` + `no_grad()`, tính lại $S$ cho test và so với `scores_test.npz`; sai khác tuyệt đối lớn nhất phải ≤ 1e-4. Bước này kiểm tra luôn việc BatchNorm và dropout đã tắt đúng.
   - (c) Ghi SHA-256 của các file $S$ mà Stage 2 đã đọc.

   Ghi cả ba vào `metrics/c3.json` (`passed: true/false`).

## Chế độ `--freeze_mode unfrozen` (chế độ nghiên cứu)

1. Nạp Stage-1 `best.pt` (bộ mã hoá và `EvidenceModel`) cùng một `DecisionHead` mới. Có thể khởi tạo `DecisionHead` từ một run `frozen` cùng mode tổng hợp bằng cờ `--init_head_from <stage2_frozen_dir>`; nên dùng.
2. Huấn luyện end-to-end trên ảnh và caption, giống pipeline của file 04:
   $$\mathcal L=\mathcal L_{dis}+\lambda\,\mathcal L_{concept}$$
   - `--lambda_concept`: mặc định `1.0`. Đặt `0` thì chỉ còn loss bệnh, tức tầng bằng chứng bị nhãn bệnh kéo tự do.
   - Bộ mã hoá ở chế độ `train()`. Số epoch đặt trong config (mặc định 10). Chọn checkpoint theo val macro-F1.
3. Sau khi train, **xuất lại $S$** vào thư mục stage2 (`scores_*.npz`) và tính lại C1, C2 (dùng hàm của file 04). Thêm thống kê **độ trôi** so với Stage 1:
   - trung bình của $\lvert s^{stage2}_{ij}-s^{stage1}_{ij}\rvert$;
   - tương quan giữa $s^{stage1}$ và $s^{stage2}$;
   - mức thay đổi của C1 và C2.

   Ghi vào `metrics/drift.json`.
4. C3 không áp dụng cho chế độ này. Ghi `c3.json` với `"applicable": false`.

> **Giả định cần Trí xác nhận:** chế độ không đóng băng được hiểu là loss bệnh cộng λ lần loss mệnh đề. Chạy λ = 1 và λ = 0 bao được cả hai cách hiểu đã bàn. Nếu Trí chọn cách khác, chỉ cần sửa mục này.

## Bảng kiểm (chung cho cả hai chế độ)

Với mỗi ca test và 3 bệnh có xác suất cao nhất, ghi lại **mọi mệnh đề của bệnh đó trong KB**:

| Loại mệnh đề | Thông tin ghi |
|---|---|
| quan sát được | $s_{ij}$, $\tilde s_{ij}$, polarity, cờ `in_aggregation` (thuộc $B_d^*$ hay không), trạng thái `có` khi $s\ge$ `--status_threshold` (mặc định 0,5), ngược lại là `không`; top-k vết attention (lấy từ `attention_test.npz`, hoặc tính lại nếu chạy `unfrozen`); nhãn yếu; nhãn chấm tay nếu ca nằm trong mẫu chấm tay |
| không quan sát được | trạng thái `không đủ thông tin` |

Mọi dòng đều kèm `sourceExcerpt`. Ghi ra `checklist/checklist_test.json`. Ngưỡng 0,5 chỉ dùng để hiển thị; không dùng ở bất kỳ phép tính nào khác.

## Đầu ra

```text
runs/<run_name>/<timestamp>/stage2_<freeze_mode>_<agg>[_lam<λ>]/
├── checkpoints/best.pt            # frozen: chỉ DecisionHead; unfrozen: toàn bộ mô hình
├── predictions/
│   ├── val_epoch_XXX.npz          # gt, pred_proposition (như P1)
│   └── test_best.npz              # gt, pred_proposition, image_paths (như P1)
├── final_metrics.json             # cùng cấu trúc P1: {"best_epoch", "best_validation_score",
│                                  #   "proposition": final_metrics(gt, pred)}
├── metrics/{metrics_history.csv, c3.json, drift.json (chỉ unfrozen)}
├── scores_*.npz                   # chỉ unfrozen
├── checklist/checklist_test.json
├── run_config.json
└── meta.json
```

Tên khoá `pred_proposition` và cấu trúc `final_metrics.json` phải giữ **đúng như P1**, để `K34_Results_Evaluation.ipynb` đọc được mà không phải sửa. Hàm `final_metrics` lấy từ `proposition.engine.artifacts`.

## Xong khi

- Run `frozen` + `mean` hoàn tất, `c3.json` báo `passed: true`.
- Có `final_metrics.json` và `test_best.npz` đúng format.
- Unit test chạy qua:
  - `ActiveAggregator` chỉ dùng cột active;
  - mode `mean` cho trung bình đúng;
  - $\gamma_d>0$;
  - với chế độ `frozen`, sau một bước optimizer, tham số của bộ mã hoá không đổi.

## Không được làm

- Ở chế độ `frozen`: cập nhật hay tính lại $S$ trong lúc train; để bộ mã hoá ở `train()`.
- Đưa vào phép tổng hợp mệnh đề không quan sát được, hoặc mệnh đề không có nhãn nào trong train (trừ khi dùng fallback).
- Chọn mô hình theo test.
- Đổi tên khoá hay cấu trúc file dự đoán so với P1.
