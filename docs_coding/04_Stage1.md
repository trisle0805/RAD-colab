# 04 · Stage 1: học tầng bằng chứng

*Lý do: file câu chuyện, mục 6.3 (Stage 1) và mục 4 (vì sao mỗi mệnh đề cần tín hiệu riêng).*

## Mục tiêu

Huấn luyện mô hình đọc (ảnh + caption) bằng query của từng mệnh đề để cho ra điểm $s_{ij}$. Mô hình **chỉ** được học từ nhãn yếu, không dùng nhãn bệnh. Sau khi huấn luyện, xuất ma trận $S$ và vết attention, rồi tính C1 và C2.

## Đầu vào

| Thứ | Nguồn |
|---|---|
| CSV, ảnh, KB | `configs/ws.yaml` |
| Mệnh đề quan sát được | `WeakSup/propositions/v2/propositions_final.json` (file 02) |
| Nhãn yếu | `WeakSup/weak_labels/v<k>/labels.npz` (file 03); dùng cột `label_final` |
| Bộ mã hoá | `resnet50` và `emilyalsentzer/Bio_ClinicalBERT`, giống P1 |
| Cấu hình huấn luyện | `configs/skin.yaml` (batch, lr, scheduler) cộng `configs/ws_stage1.yaml` |

## Code mới

| File | Nội dung |
|---|---|
| `proposition/weak/dataset.py` | `CaseIndexedDataset(base)`: bọc `Skin_Train_Dataset` / `Skin_Test_Dataset`, trả thêm `case_id = str(base.img_path_list[index])`. **Không sửa `dataset/`.** |
| `proposition/weak/model.py` | `EvidenceModel(embed_dim, tau_att)`: dùng lại `CosineCrossAttention` và `SharedScorer` (import từ `proposition.models.proposition_model`, không sửa file đó); `forward(query, memory, memory_padding_mask, return_attention)` trả `logits (B, M)`, `support (B, M)`, `attention`. Không có aggregator, calibration hay nhánh nhãn |
| `proposition/weak/pipeline.py` | `WSBatchStream`: giống `EncodedBatchStream` của P1 nhưng (1) chỉ mã hoá query cho **các mệnh đề quan sát được**, theo thứ tự của `propositions_final.json`; (2) không tạo `label_query` và `alignment_prototypes`; (3) trả thêm `case_id` và `weak_labels (B, M)` lấy từ `labels.npz` |
| `proposition/weak/losses.py` | `masked_bce(logits, targets, pos_weight=None)`, với `targets ∈ {1, 0, -1}` |
| `proposition/weak/metrics.py` | `concept_auc`, `within_disease_auc` (C1), `proposition_correlation` (C2), `param_hash` |
| `main_ws_stage1.py` | entrypoint |

Query của mệnh đề là `canonicalDescription`, giống P1. Câu hỏi đã bọc mẫu ở file 02 chỉ dành cho bộ gán nhãn.

## Việc cần làm

### 1. Dữ liệu

- Train dùng `RandomSampler` và `fourier_aug`; val và test dùng `SequentialSampler`, giống P1.
- Assert: mọi `case_id` của cả ba split đều có trong `labels.npz`; thứ tự cột mệnh đề khớp với `propositions_final.json`.
- Caption đưa vào mô hình là **caption gốc** (cột `caption_zh_polish_en`), **không** che tên bệnh. Lý do: để so sánh trực tiếp với RAD (file câu chuyện, mục 8, hạn chế 2).

### 2. Hàm mất mát

```python
mask   = targets >= 0                      # bỏ ∅
loss   = BCEWithLogits(logits[mask], targets[mask].float(), pos_weight=pos_weight)
```

- Chuẩn hoá theo số ô có nhãn trong batch. Batch không có ô nào có nhãn thì bỏ qua và tăng bộ đếm `skipped_batches`.
- `pos_weight`: cờ `--pos_weight none|auto|<số>`, mặc định `none`.
  - `auto` = (số nhãn 0)/(số nhãn 1) trên toàn bộ train, đọc từ survey của file 03.
  - Trí quyết định có bật hay không sau khi xem survey.
- Mỗi epoch log `n_labeled`, `n_pos`, `n_neg`, `skipped_batches`. Assert không có ô `-1` nào nằm trong `mask`.

### 3. Tham số được học và optimizer

- Optimizer **chỉ** gồm: `image_encoder`, `text_encoder` (cả BERT và `mlp_embed`), `EvidenceModel` (`q_proj`, `k_proj`, `v_proj`, scorer).
- Assert: không có `label_branch`, không có PECL, không có tham số calibration hay aggregation nào được tạo ra ở Stage 1.
- AdamW, chia decay / no-decay như P1, dùng lr và scheduler của `configs/skin.yaml`. Số epoch đặt trong `configs/ws_stage1.yaml` (mặc định 20).

### 4. Đánh giá mỗi epoch trên val và chọn checkpoint

Mỗi epoch, tính trên val:

- `val_concept_loss`: masked BCE.
- `val_concept_auc`: AUC trung bình trên các mệnh đề có ít nhất 3 nhãn 1 và ít nhất 3 nhãn 0 trong val.
- `val_c1`: C1 trên val (xem mục 6).

**Chọn checkpoint theo `val_concept_auc`.** Nếu không mệnh đề nào đủ điều kiện thì dùng `val_concept_loss`. Không chọn theo độ chính xác chẩn đoán vì Stage 1 không có chẩn đoán.

Lưu như P1:

- `checkpoints/latest.pt`: để chạy tiếp; gồm optimizer, scheduler, trạng thái RNG.
- `checkpoints/best.pt`: `image_encoder`, `text_encoder`, `evidence_model`, config, epoch, metrics.
- `metrics/metrics_history.csv`.

### 5. Xuất kết quả với `best.pt`

Chạy ở chế độ `eval()` và `torch.no_grad()`:

```text
runs/<run_name>/<timestamp>/stage1/
├── checkpoints/{best.pt, latest.pt}
├── scores_train.npz      # case_id (N,), S (N,M) float32, logits (N,M), gt (N,C)
├── scores_val.npz
├── scores_test.npz
├── attention_test.npz    # top-k (k=10) chỉ số và trọng số bộ nhớ cho mỗi (ca, mệnh đề),
│                         # image_mass, caption_mass;
│                         # full attention cho 20 ca test chọn bằng seed + các ca trong mẫu chấm tay
├── param_hash.json       # SHA-256 trạng thái image_encoder, text_encoder, evidence_model
├── metrics/{metrics_history.csv, c1_c2.json}
├── run_config.json       # tham số, git commit, SHA-256 đầu vào, phiên bản nhãn và mệnh đề
└── meta.json
```

### 6. Định nghĩa C1 và C2 (cài trong `proposition/weak/metrics.py`, tính trên test)

**C1: phân biệt trong cùng bệnh.**

- Với mỗi cặp (mệnh đề *j*, bệnh *d*): lấy các ca test thuộc bệnh *d* có nhãn của *j*. Nếu có ít nhất 3 nhãn 1 và ít nhất 3 nhãn 0, tính AUC của $s_{ij}$ so với $r_{ij}$.
- Báo cáo: số cặp đủ điều kiện, AUC trung bình, AUC trung vị, tỉ lệ cặp có AUC > 0,5.
- Mốc so sánh: một mô hình chỉ biết bệnh cho cùng một điểm cho mọi ca trong cùng bệnh, nên AUC của nó bằng 0,5.

Kèm theo **AUC toàn cục** trên mọi ô có nhãn, so với baseline "chỉ biết bệnh": điểm $b_{ij}$ = tỉ lệ nhãn 1 của mệnh đề *j* trên các ca train cùng bệnh với ca *i*.

**C2: các mệnh đề có còn nhân bản nhau không.**

- (a) **Điểm:** với mỗi bệnh *d*, lấy ma trận $s$ trên các ca test thuộc *d* và các mệnh đề trong $B_d^*$ (định nghĩa ở file 00 / 05). Tính tương quan Pearson giữa các cột, lấy trung bình phần ngoài đường chéo, rồi lấy trung bình theo bệnh.
- (b) **Attention:** với các ca có full attention, lấy bệnh đúng *d* của ca, tính tương quan Pearson giữa bản đồ attention của các mệnh đề trong $B_d^*$, rồi lấy trung bình.
- Chạy **cùng hàm này** trên artifact checklist của P1 (`support` và `full_attention` trong `checklist/` của run P1), giới hạn ở các mệnh đề quan sát được, để có mốc so sánh cùng cách đo.

### 7. Công tắc dành cho file 06 (cài sẵn, mặc định tắt)

| Cờ | Ý nghĩa |
|---|---|
| `--memory both\|image\|text` | che bớt một phần bộ nhớ khi train và khi test (O3); mặc định `both` |
| `--shuffle_labels` | hoán vị ngẫu nhiên (seed 42) các **hàng** của ma trận nhãn train giữa các ca (O1) |
| `--weak_labels_dir` | chọn phiên bản nhãn, hoặc nhãn chạy với `--teacher text\|image` ở file 03 |

## Xong khi

- Có `best.pt`, `scores_*.npz`, `attention_test.npz`, `param_hash.json`.
- Có `c1_c2.json` cho run này **và** C2 tính lại trên P1 bằng cùng hàm.
- Unit test trong `proposition/tests/weak/` chạy qua: masked BCE bỏ qua `-1`; wrapper dataset trả đúng `case_id`; thứ tự cột khớp; C1 cho kết quả 0,5 với điểm hằng số.
- Trí đã xem C1 và C2.

## Không được làm

- Dùng nhãn bệnh dưới bất kỳ hình thức nào trong loss của Stage 1.
- Bật lại PECL hay `label_aux`.
- Tính loss trên ô `-1`.
- Chọn checkpoint theo kết quả test.
- Sửa file trong `dataset/`, `models/`, `engine/`, hay code P1 trong `proposition/` (ngoài thư mục `proposition/weak/`).
