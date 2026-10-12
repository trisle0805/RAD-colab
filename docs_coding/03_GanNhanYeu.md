# 03 · Gán nhãn yếu, khảo sát nhãn và mẫu chấm tay

*Lý do: file câu chuyện, mục 5.2–5.4 (ba nguyên tắc của nhãn yếu, NLI, ConceptCLIP) và mục 6.2 (bước gán nhãn, quy tắc gộp hai nguồn).*

## Mục tiêu

Với mỗi cặp (ca, mệnh đề quan sát được), tạo nhãn yếu $r_{ij}\in\{1, 0, -1\}$, trong đó `-1` nghĩa là không đủ thông tin (∅). Nhãn lấy từ hai nguồn:

- **văn bản:** bộ gán nhãn của file 01, đọc caption; sau đó một LLM kiểm tra lại một phần nhỏ các ô (bước A2b);
- **ảnh:** ConceptCLIP, đọc ảnh.

Sau đó khảo sát thống kê của nhãn và cho Trí chấm tay một mẫu nhỏ.

## Đầu vào

| Thứ | Nguồn |
|---|---|
| CSV train / val / test và ảnh SkinCAP | `configs/ws.yaml` |
| Bộ gán nhãn văn bản | `WeakSup/teacher_text/bioclinicalbert_mednli_v1/` (file 01) |
| Mệnh đề và câu hỏi | `WeakSup/propositions/v1/propositions_final.json` (file 02) |
| ConceptCLIP | `JerrryNie/ConceptCLIP`: model gated, Trí tự chấp nhận điều khoản và đặt HF token trong Colab Secrets; nạp bằng `AutoModel` / `AutoProcessor` với `trust_remote_code=True` |
| LLM kiểm tra lại (bước A2b) | API của Anthropic hoặc OpenAI, chọn bằng cờ `--llm_provider`; API key do Trí đặt trong biến môi trường hoặc Colab Secrets, **không** ghi vào code hay repo |

Gán nhãn cho **cả ba split**: train dùng làm tín hiệu huấn luyện, val dùng chọn checkpoint, test dùng báo cáo.

---

## Phần A · Gán nhãn (notebook `WS_03_weak_labels.ipynb`)

### A1. Che tên bệnh trong caption (chỉ cho bộ gán nhãn)

1. Tự sinh danh sách tên cần che cho 47 bệnh:
   - tên trong CSV;
   - biến thể dấu gạch / khoảng trắng / số nhiều;
   - dạng tính từ hay gặp (psoriatic, eczematous…);
   - viết tắt rõ ràng (BCC, SCC, SJS…).

   Xuất `disease_alias_v1.csv` (`disease_id`, `alias`, `source`) để **Trí duyệt**.
2. Thay mọi alias đã duyệt (không phân biệt hoa thường, khớp nguyên từ) bằng cụm `this skin condition`.
3. **Mô hình chính (file 04) vẫn đọc caption gốc.** Bước che tên chỉ dùng cho bộ gán nhãn.

### A2. Nhãn từ văn bản (NLI)

Với mỗi ca *i*:

1. Tách caption đã che tên thành các câu (regex theo dấu `.`, `!`, `?`). Lưu danh sách câu.
2. Với mỗi mệnh đề quan sát được *j*, mỗi câu hỏi con *h* của *j* (đã bọc mẫu `The patient has {…}.`) và mỗi câu *k*: tính xác suất ba nhãn của cặp (câu *k*, *h*).
3. Gộp theo câu: $p^{ent}_{ih}=\max_k p^{ent}_{ihk}$, $p^{con}_{ih}=\max_k p^{con}_{ihk}$.
4. Gộp theo câu hỏi con, theo nguyên tắc "có một là đủ":

   | Điều kiện | `label_text` |
   |---|---|
   | tồn tại *h* có $p^{ent}_{ih}\ge\tau^{+}_{txt}$ | `1` |
   | không có *h* nào đạt mức trên, **và mọi** *h* có $p^{con}_{ih}\ge\tau^{-}_{txt}$ | `0` |
   | còn lại | `-1` |

   Với mệnh đề `single` hoặc `pattern`, chỉ có một *h* nên bảng trên quy về điều kiện của một câu.
5. Ngưỡng ban đầu: $\tau^{+}_{txt}=0{,}9$, $\tau^{-}_{txt}=0{,}8$, đặt trong config. Ngưỡng của nhãn 0 thấp hơn vì ở bộ thử 30 ca (file 01), các dự đoán contradiction đều đúng còn nhãn 0 thì rất hiếm. Có thể chỉnh sau khi chấm tay; khi đó tạo phiên bản mới.
6. Lưu nhãn này vào `label_text_nli`. Nhãn văn bản cuối cùng `label_text` là kết quả sau bước A2b; nếu A2b tắt thì `label_text = label_text_nli`.

Chạy theo lô (batch 256, fp16, `max_length=128`) và lưu kết quả dở dang theo từng khối ca lên Drive, để chạy tiếp được nếu Colab ngắt. Ước lượng: khoảng 2.500 ca × ~450 mệnh đề × vài câu hỏi con × ~3 câu, tức vài triệu cặp. Trên T4 mất vài chục phút đến khoảng 1–2 giờ.

### A2b. LLM kiểm tra lại nhãn văn bản (cờ `--llm_verify`, mặc định bật)

*Lý do: ở bộ thử của file 01, kiểu lỗi chính của NLI là phủ định nằm giữa câu ("but not", "without", "sparing", "unaffected"). Lỗi này tạo ra nhãn 1 sai. LLM đóng vai **người kiểm tra lại**, không phải bộ gán nhãn chính.*

**1. Những ô được gửi đi** (tính trên kết quả A2):

| Nhóm | Điều kiện | Cờ | Mặc định |
|---|---|---|---|
| A | `label_text_nli = 1` **và** câu căn cứ (`best_sentence_idx`) chứa từ phủ định | `--llm_group_a` | bật |
| B | `label_text_nli = -1`, không có câu hỏi con nào $p^{ent}\ge\tau^{+}_{txt}$, và $0{,}5\le \max p^{con}<\tau^{-}_{txt}$ | `--llm_group_b` | bật |
| C | `label_text_nli = -1` và $0{,}5\le \max p^{ent}<\tau^{+}_{txt}$ | `--llm_group_c` | **tắt** |

Danh sách từ phủ định (khớp nguyên từ, không phân biệt hoa thường; đặt trong config): `no, not, without, absent, absence, sparing, spared, spares, unaffected, negative, free of, lack, lacks, lacking, none, neither, nor, denies, never`.

Các ô còn lại **không gọi** LLM và giữ nguyên `label_text_nli`.

**2. Chạy thử trước khi gọi API (bắt buộc).** In ra số ô của từng nhóm theo split, số ca có ít nhất một ô được chọn (tức số lần gọi) và ước lượng số token. Dừng lại để Trí xác nhận rồi mới gọi API thật.

**3. Cách gọi.**

- Gom theo ca: **một lần gọi cho mỗi ca**. Đầu vào là caption đã che tên bệnh (A1) và danh sách các mệnh đề được chọn của ca đó. Mỗi mệnh đề gửi kèm `proposition_id` và danh sách câu hỏi con (dạng cụm danh từ, chưa bọc mẫu).
- `temperature = 0`; yêu cầu trả JSON theo schema bên dưới; nếu JSON lỗi thì thử lại tối đa 3 lần.
- Lưu cache theo khoá (`case_id`, `proposition_id`, `prompt_version`) thành file JSONL trên Drive, để chạy tiếp được nếu bị ngắt.
- Script là `proposition/weak/llm_verify.py`, hỗ trợ cả hai nhà cung cấp, **Trí tự chạy**.

**4. Prompt** (lưu ở `proposition/weak/prompts/llm_verify_v1.txt`; `prompt_version` là SHA-256 của file này):

```text
You are verifying what a clinical description of a skin lesion explicitly states.
For each sign below, answer based ONLY on the description text.

- PRESENT: the description explicitly states the patient has the sign.
  If the sign lists several sites or alternative forms, one of them is enough.
- ABSENT: the description explicitly states the patient does NOT have the sign.
  Pay attention to the scope of negations such as "without", "but not", "no", "sparing".
- NOT_MENTIONED: the description does not say either way.

Do NOT use medical knowledge to infer signs that are not written.
Do NOT guess the diagnosis.
For PRESENT or ABSENT, copy the exact supporting phrase from the description (verbatim).

Description:
<<<{masked_caption}>>>

Signs:
{numbered list: id | sub-hypotheses joined by " / "}

Return JSON only:
{"results": [{"id": "<proposition_id>", "verdict": "PRESENT|ABSENT|NOT_MENTIONED", "quote": "<verbatim phrase or empty>"}]}
```

**5. Kiểm tra câu trả lời trong code.**

- `quote` phải là chuỗi con của caption đã che tên, sau khi chuẩn hoá khoảng trắng và không phân biệt hoa thường. Nếu không đúng thì đổi verdict thành `NOT_MENTIONED` và đặt cờ `quote_invalid=1`.
- `id` lạ hoặc thiếu thì coi là `NOT_MENTIONED`.

**6. Đổi nhãn** (LLM **không bao giờ** tạo nhãn 1 từ ∅, trừ nhóm C):

| Nhóm | Verdict | `label_text` |
|---|---|---|
| A | PRESENT | 1 (giữ) |
| A | ABSENT | 0 |
| A | NOT_MENTIONED | −1 |
| B | ABSENT | 0 |
| B | PRESENT / NOT_MENTIONED | −1 (giữ) |
| C (nếu bật) | PRESENT | 1 |
| C (nếu bật) | ABSENT / NOT_MENTIONED | −1 (giữ) |

Áp dụng **cùng quy tắc cho train, val và test**. Không chỉnh tay nhãn của bất kỳ ô nào.

### A3. Nhãn từ ảnh (ConceptCLIP), chỉ cho mệnh đề có `image_applicable=1`

1. Với mỗi câu hỏi con *h* của mệnh đề *j*, dùng hai câu dương và hai câu âm (ghi trong config):
   - dương: `"a clinical photo of skin with {h}"`, `"this skin lesion shows {h}"`
   - âm: `"a clinical photo of skin without {h}"`, `"this skin lesion does not show {h}"`
2. Mã hoá ảnh một lần cho mỗi ca và mã hoá văn bản một lần cho mỗi câu (lưu cache). Gọi $\bar c^{+}_{ih}$, $\bar c^{-}_{ih}$ là cosine trung bình với nhóm câu dương và nhóm câu âm. Tính
   $p^{img}_{ih}=\mathrm{softmax}(\text{logit\_scale}\cdot[\bar c^{+}_{ih},\bar c^{-}_{ih}])_0$.
3. Gộp theo nguyên tắc "có một là đủ":

   | Điều kiện | `label_img` |
   |---|---|
   | $\max_h p^{img}_{ih}\ge\tau^{+}_{img}$ | `1` |
   | **mọi** *h* có $p^{img}_{ih}\le\tau^{-}_{img}$ | `0` |
   | còn lại | `-1` |

4. Ngưỡng ban đầu: $\tau^{+}_{img}=0{,}8$, $\tau^{-}_{img}=0{,}2$, đặt trong config. Phần B có histogram để chỉnh.
5. Mệnh đề có `image_applicable=0` nhận `label_img=-1`.

### A4. Gộp hai nguồn

| `label_text` | `label_img` | `label_final` |
|---|---|---|
| 1 | 1 hoặc −1 | 1 |
| 0 | 0 hoặc −1 | 0 |
| −1 | 1 / 0 | bằng `label_img` |
| −1 | −1 | −1 |
| 1 | 0 | −1 (mâu thuẫn) |
| 0 | 1 | −1 (mâu thuẫn) |

Có cờ `--teacher text|image|both` (mặc định `both`) để chạy ablation bộ gán nhãn ở file 06.

### A5. Lưu kết quả

```text
WeakSup/weak_labels/v1/
├── labels.npz
│   ├── case_id            (N,)     str
│   ├── split              (N,)     str   train/val/test
│   ├── proposition_id     (M,)     str   đúng thứ tự propositions_final.json
│   ├── label_text_nli     (N, M)   int8  {1, 0, -1}   kết quả A2
│   ├── label_text         (N, M)   int8  {1, 0, -1}   sau A2b
│   ├── llm_group          (N, M)   int8  0 = không gửi, 1 = A, 2 = B, 3 = C
│   ├── llm_verdict        (N, M)   int8  -1 = không gửi, 0 = NOT_MENTIONED, 1 = PRESENT, 2 = ABSENT
│   ├── llm_quote_invalid  (N, M)   int8
│   ├── label_img          (N, M)   int8
│   ├── label_final        (N, M)   int8
│   ├── p_entail           (N, M)   float16   max theo câu và câu hỏi con
│   ├── p_contra           (N, M)   float16
│   ├── p_img              (N, M)   float16   NaN nếu không áp dụng
│   └── best_sentence_idx  (N, M)   int16     câu cho p_entail cao nhất, -1 nếu không có
├── sentences_masked.json  # {case_id: [câu đã che tên, …]}
├── llm_cache.jsonl        # mọi câu trả lời thô của LLM (kèm quote)
├── disease_alias_v1.csv   # bản Trí đã duyệt
└── meta.json              # tên và SHA-256 bộ gán nhãn, ngưỡng, mẫu câu, phiên bản propositions,
                           # SHA-256 KB / CSV / alias, cờ --teacher, thời gian chạy;
                           # với A2b: provider, tên và phiên bản model, prompt_version,
                           # danh sách từ phủ định, các cờ nhóm, số lần gọi, số token
```

Hàm đọc file này đặt trong `proposition/weak/labels.py`: `load_weak_labels(path) -> WeakLabels`, có chỉ mục `case_id → hàng` và `proposition_id → cột`.

---

## Phần B · Khảo sát thống kê (notebook `WS_03b_label_survey.ipynb`)

Chỉ đọc `labels.npz` rồi xuất kết quả, không sửa nhãn.

1. **Theo mệnh đề**, ghi ra `stats_per_proposition.csv`:
   - `n_present`, `n_absent`, `n_unknown`, `positive_rate`, `coverage` (tỉ lệ ô có nhãn), tính riêng cho từng split;
   - tách theo nguồn: số nhãn đến từ văn bản và số nhãn đến từ ảnh;
   - `n_present_own_disease` và `n_present_other_disease`: nhãn 1 nằm trên ca thuộc chính bệnh của mệnh đề hay trên ca bệnh khác;
   - cờ `positive_only` (có nhãn 1 nhưng không có nhãn 0) và cờ `no_label` (không có nhãn nào).
2. **Theo nhóm** (morphology / location) và **toàn bộ**: tổng số nhãn 1, nhãn 0, ∅, và tỉ lệ nhãn 1 trên nhãn 0. Tỉ lệ này dùng để quyết định có bật cân bằng lớp ở file 04 hay không.
3. **Tác động của LLM (A2b)**: với từng nhóm A/B/C, số ô đã gửi, số ô giữ nguyên, số ô hạ từ 1 xuống ∅, số ô đổi từ 1 sang 0, số ô thêm nhãn 0, và số ô có `quote_invalid`.
4. **Độ khớp giữa văn bản và ảnh** trên các ô mà cả hai nguồn đều có nhãn: bảng chéo 2×2 và số ô mâu thuẫn.
5. **Histogram** của `p_entail`, `p_contra`, `p_img`, kèm vị trí các ngưỡng.
6. **Tóm tắt** trong `survey_summary.md`: các con số chính và danh sách mệnh đề `no_label` / `positive_only`. Chỉ ghi số liệu, không viết diễn giải.

Đầu ra: `WeakSup/weak_labels/v1/survey/`.

---

## Phần C · Mẫu chấm tay

1. Lấy khoảng **150 cặp từ split test** theo các tầng, seed 42:

   | Tầng | Số cặp |
   |---|---|
   | `label_final=1`, nguồn văn bản | 50 |
   | `label_final=1`, chỉ nguồn ảnh | 25 |
   | `label_final=0` | 35 |
   | `label_final=-1` (để xem có bỏ sót dấu hiệu nào) | 40 |
   | ô bị LLM đổi nhãn (A2b) | 30 (lấy thêm, ngoài 150) |

2. Xuất `audit/audit_v1.csv` với các cột: `case_id`, `image_path`, `caption_original`, `proposition_id`, `canonicalDescription`, `category`, cột ẩn `teacher_label`, và cột **`human_label`** (1 / 0 / ? do Trí điền), `note`.
3. Xuất `audit/audit_v1.html` để xem ảnh, caption và mệnh đề. File HTML **không hiển thị nhãn của bộ gán nhãn**, để việc chấm không bị ảnh hưởng.
4. Sau khi Trí chấm xong, chạy ô tính:
   - precision của nhãn 1, precision của nhãn 0, tỉ lệ "∅ nhưng thực ra là 1";
   - so sánh **trước và sau A2b**: tính trên `label_text_nli` và trên `label_text`;
   - tách theo nguồn (văn bản / ảnh) và theo nhóm mệnh đề.

   Ghi kết quả vào `audit/audit_metrics.json`.

Mẫu chấm tay này dùng lại ở file 06 để đánh giá chính mô hình trên cùng các cặp.

---

## Xong khi

- `weak_labels/v1/` đã tạo và `meta.json` đầy đủ.
- Đã có `survey/` và Trí đã xem.
- Trí đã chấm xong mẫu và đã có `audit_metrics.json`.
- Nếu Trí quyết định đổi ngưỡng hoặc mẫu câu: tạo `weak_labels/v2/` và chạy lại phần B và C. Không ghi đè `v1`.

## Không được làm

- Biến ∅ thành 0 ở bất kỳ bước nào.
- Gán nhãn mệnh đề dựa trên nhãn bệnh của ca.
- Cho bộ gán nhãn đọc caption chưa che tên bệnh. Điều này áp dụng cho cả LLM.
- Gọi API thật trước khi Trí xác nhận kết quả chạy thử (A2b, bước 2).
- Để LLM tạo nhãn 1 từ ô ∅ khi nhóm C đang tắt.
- Dùng nhãn của bộ gán nhãn để sửa ngược lại KB hay danh sách mệnh đề.
