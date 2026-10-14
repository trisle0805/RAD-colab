# 03 · Gán nhãn yếu, khảo sát nhãn và mẫu chấm tay

*Lý do: file câu chuyện, mục 5.3–5.4 (ba nguyên tắc của nhãn yếu, LLM, ConceptCLIP) và mục 6.2 (bước gán nhãn, quy tắc gộp hai nguồn).*

## Trạng thái (14/10/2026)

- **`weak_labels/v1` (NLI) không dùng nữa.** Khảo sát cho thấy bộ gán nhãn NLI gán "có" cho hầu hết ca, kể cả với mệnh đề của bệnh khác. Thư mục v1 giữ nguyên để đối chiếu, không xoá.
- **Phiên bản đang làm: `weak_labels/v2`.** Bộ gán nhãn văn bản là **LLM** (bước A2 mới). Che tên bệnh (A1), ConceptCLIP (A3), gộp hai nguồn (A4) giữ như cũ.
- **Mệnh đề dùng bản `propositions/v2`** (file 02, mục Trạng thái).
- Bộ gán nhãn của file 01 không còn được dùng.

## Mục tiêu

Với mỗi cặp (ca, mệnh đề quan sát được), tạo nhãn yếu $r_{ij}\in\{1, 0, -1\}$, trong đó `-1` nghĩa là không đủ thông tin (∅). Nhãn lấy từ hai nguồn:

- **văn bản:** LLM đọc caption đã che tên bệnh;
- **ảnh:** ConceptCLIP đọc ảnh.

Sau đó khảo sát thống kê của nhãn và cho Trí chấm tay một mẫu nhỏ.

## Đầu vào

| Thứ | Nguồn |
|---|---|
| CSV train / val / test và ảnh SkinCAP | `configs/ws.yaml` |
| Mệnh đề và câu hỏi con | `WeakSup/propositions/v2/propositions_final.json` (file 02) |
| Danh sách tên bệnh cần che | `WeakSup/weak_labels/v1/disease_alias_v1.csv` (Trí đã duyệt; dùng lại, không tạo mới) |
| ConceptCLIP | `JerrryNie/ConceptCLIP`: model gated, Trí tự chấp nhận điều khoản và đặt HF token trong Colab Secrets; nạp bằng `AutoModel` / `AutoProcessor` với `trust_remote_code=True` |
| LLM | API của Anthropic hoặc OpenAI, chọn bằng `--llm_provider`; tên model đặt trong config (`llm.model`). API key do Trí đặt trong biến môi trường hoặc Colab Secrets, **không** ghi vào code hay repo |

Gán nhãn cho **cả ba split**: train dùng làm tín hiệu huấn luyện, val dùng chọn checkpoint, test dùng báo cáo.

---

## Phần A · Gán nhãn (notebook `WS_03_weak_labels.ipynb`)

### A1. Che tên bệnh trong caption (chỉ cho bộ gán nhãn)

1. Dùng lại `disease_alias_v1.csv` đã duyệt.
2. Thay mọi alias (không phân biệt hoa thường, khớp nguyên từ) bằng cụm `this skin condition`.
3. **Mô hình chính (file 04) vẫn đọc caption gốc.** Bước che tên chỉ dùng cho bộ gán nhãn.

Lưu caption đã che vào `captions_masked.json`.

### A2. Nhãn từ văn bản (LLM)

**1. Gom mệnh đề thành danh sách dấu hiệu.**

- Chia các mệnh đề quan sát được theo 4 nhóm: `morphology`, `location`, `symptom`, `history`.
- Trong mỗi nhóm, các mệnh đề có cùng câu hỏi con (sau khi chuẩn hoá chữ thường và khoảng trắng) được gộp thành **một dấu hiệu**, có mã `S001`, `S002`… Ví dụ "painful lesions" xuất hiện ở hai bệnh thì chỉ hỏi một lần, và cả hai mệnh đề nhận cùng nhãn.
- Mỗi dấu hiệu được hiển thị bằng các câu hỏi con nối bởi ` / ` (cụm danh từ, **không** bọc mẫu "The patient has…").
- Lưu ánh xạ dấu hiệu → danh sách `proposition_id` vào `sign_map.json`.

**2. Cách gọi.**

- **Mỗi ca gọi 4 lần, mỗi lần một nhóm.** Danh sách dấu hiệu của một nhóm giống hệt nhau ở mọi ca.
- **Thứ tự trong prompt:** hướng dẫn + danh sách dấu hiệu đặt **trước**, caption đặt **cuối**. Như vậy phần đầu prompt giống nhau giữa các ca, và bật được prompt caching:
  - Anthropic: đặt `cache_control` trên khối hướng dẫn + danh sách dấu hiệu;
  - OpenAI: caching tự động khi phần đầu prompt giống nhau.
- `temperature = 0`. Yêu cầu trả JSON theo schema bên dưới. Nếu JSON lỗi thì thử lại tối đa 3 lần; vẫn lỗi thì cả nhóm của ca đó là ∅ và ghi vào log.
- Lưu cache theo khoá (`case_id`, nhóm, `prompt_version`, `model`) thành file JSONL trên Drive, để chạy tiếp được nếu bị ngắt.
- Script là `proposition/weak/llm_label.py`, hỗ trợ cả hai nhà cung cấp, **Trí tự chạy**.

**3. Prompt** (lưu ở `proposition/weak/prompts/llm_label_v1.txt`; `prompt_version` là SHA-256 của file này):

```text
You are extracting what a clinical description of a skin case explicitly states about the patient.

You will receive a list of signs and then one description. For each sign, decide:
- PRESENT: the description explicitly states that this patient has the sign.
  If a sign lists several sites or alternative forms (separated by " / "), one of them is enough.
- ABSENT: the description explicitly states that this patient does NOT have the sign,
  or explicitly describes the same feature in a directly opposite way
  (e.g. the description says "well-demarcated" and the sign is "ill-defined borders").
  Pay attention to the scope of negations such as "without", "but not", "no", "sparing".
- NOT_MENTIONED: anything else. This is the default.

Rules:
1. Use ONLY the words of the description. Do NOT use medical knowledge to infer signs that are not written.
2. Do NOT guess the diagnosis. Do not mark a sign because it is typical of a disease you suspect.
3. Only statements about THIS patient count. Sentences describing the condition in general
   ("this skin condition is characterized by ...", "it commonly causes ...", "it is often triggered by ...")
   are NOT evidence: signs supported only by such sentences are NOT_MENTIONED.
4. Sharing words is not enough. "red papules" does not mean "red hair";
   "lesions on the arm" does not mean "lesions on the trunk".
5. When unsure, answer NOT_MENTIONED.
6. For PRESENT or ABSENT, copy the exact supporting phrase from the description (verbatim, short).

Signs ({category}):
{one line per sign: "S001 | sub-hypothesis 1 / sub-hypothesis 2 / ..."}

Return JSON only. List ONLY the signs that are PRESENT or ABSENT; omit NOT_MENTIONED signs:
{"results": [{"id": "S001", "verdict": "PRESENT|ABSENT", "quote": "<verbatim phrase>"}]}

Description:
<<<{masked_caption}>>>
```

**4. Kiểm tra câu trả lời trong code.**

- `quote` phải là chuỗi con của caption đã che tên, sau khi chuẩn hoá khoảng trắng và không phân biệt hoa thường. Sai thì đổi thành ∅ và đặt `llm_quote_invalid=1`.
- `id` lạ thì bỏ qua. Một `id` xuất hiện hai lần với hai verdict khác nhau thì cho ∅.
- Dấu hiệu không có trong câu trả lời là ∅.

**5. Ra nhãn.** `PRESENT` → `1`, `ABSENT` → `0`, còn lại → `-1`. Nhãn của dấu hiệu được chép sang mọi `proposition_id` trong `sign_map.json`. Lưu vào `label_text`.

Áp dụng **cùng quy tắc cho train, val và test**. Không chỉnh tay nhãn của bất kỳ ô nào.

**6. Chạy thử trước khi gọi API thật (bắt buộc, hai bước).**

- **(a) Dry-run, không gọi API.** In ra: số dấu hiệu theo nhóm, số lần gọi, ước lượng token đầu vào (tách phần được cache và phần caption), ước lượng token đầu ra. Nếu Trí điền giá vào config (`llm.price_per_mtok`) thì in thêm ước lượng chi phí. Dừng lại chờ Trí xác nhận.
- **(b) Pilot 47 ca.** Lấy từ split val, mỗi bệnh 1 ca, seed 42. Gọi API thật cho 47 ca này, rồi xuất `llm_pilot/pilot.csv` gồm: `case_id`, `disease_id` của ca, caption đã che, `proposition_id`, `canonicalDescription`, `category`, `verdict`, `quote`, `quote_valid`, `same_disease` (mệnh đề có thuộc bệnh của ca không). In thêm một bảng tóm tắt theo nhóm: tỉ lệ nhãn 1 trên mệnh đề cùng bệnh và trên mệnh đề bệnh khác. **Dừng lại để Trí xem rồi mới chạy toàn bộ.**

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

4. Ngưỡng: $\tau^{+}_{img}=0{,}8$, $\tau^{-}_{img}=0{,}2$, đặt trong config.
5. Mệnh đề có `image_applicable=0` nhận `label_img=-1`.

Nếu đã có kết quả ConceptCLIP của v1 thì được dùng lại `p_img`, chỉ cần tính lại `label_img` theo `image_applicable` của `propositions/v2`.

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
WeakSup/weak_labels/v2/
├── labels.npz
│   ├── case_id            (N,)     str
│   ├── split              (N,)     str   train/val/test
│   ├── proposition_id     (M,)     str   đúng thứ tự propositions/v2/propositions_final.json
│   ├── label_text         (N, M)   int8  {1, 0, -1}   kết quả A2
│   ├── llm_quote_invalid  (N, M)   int8
│   ├── label_img          (N, M)   int8  {1, 0, -1}
│   ├── label_final        (N, M)   int8  {1, 0, -1}
│   └── p_img              (N, M)   float16   NaN nếu không áp dụng
├── captions_masked.json   # {case_id: caption đã che tên}
├── sign_map.json          # {nhóm: {sign_id: {"text": ..., "proposition_ids": [...]}}}
├── llm_cache.jsonl        # mọi câu trả lời thô của LLM (kèm quote)
├── llm_pilot/pilot.csv    # kết quả pilot 47 ca
└── meta.json              # provider, tên và phiên bản model, prompt_version, temperature,
                           # số lần gọi, số token (đầu vào / cache / đầu ra), số JSON lỗi;
                           # ngưỡng và mẫu câu của ConceptCLIP; phiên bản propositions;
                           # SHA-256 KB / CSV / alias; cờ --teacher; thời gian chạy
```

Hàm đọc file này đặt trong `proposition/weak/labels.py`: `load_weak_labels(path) -> WeakLabels`, có chỉ mục `case_id → hàng` và `proposition_id → cột`.

---

## Phần B · Khảo sát thống kê (notebook `WS_03b_label_survey.ipynb`)

Chỉ đọc `weak_labels/v2/` rồi xuất kết quả. **Không** gọi lại LLM hay ConceptCLIP; **không** sửa `labels.npz`.

1. **Số nhãn:** số và tỉ lệ nhãn 1 / 0 / ∅. Tách theo split, theo nhóm (morphology / location / symptom / history) và theo nguồn (văn bản / ảnh). Kèm tỉ lệ nhãn 1 / nhãn 0 trên train, dùng để quyết định `pos_weight` ở file 04. Theo từng mệnh đề, ghi ra `stats_per_proposition.csv` gồm `n_present`, `n_absent`, `n_unknown`, `n_train_labels`.
2. **Ca cùng bệnh và ca bệnh khác** (kiểm tra quan trọng nhất). Với mỗi mệnh đề:
   - `rate_own`: tỉ lệ nhãn 1 trên các ca thuộc chính bệnh của mệnh đề;
   - `rate_other`: tỉ lệ nhãn 1 trên các ca thuộc bệnh khác;
   - `own_share`: trong các nhãn 1, tỉ lệ nằm trên ca cùng bệnh.

   Báo cáo trung vị theo nhóm, tách theo nguồn. Liệt kê các mệnh đề có `rate_other ≥ 0,5` (gán "có" tràn lan) và các mệnh đề có `own_share ≥ 0,9` (nhãn gần như chép theo bệnh).
3. **Histogram** của `p_img` (kẻ ngưỡng 0,2 và 0,8), vẽ riêng cho ca cùng bệnh và ca bệnh khác.
4. **Mô phỏng `min_labels`:** với `min_labels` ∈ {1, 3, 5, 10}, đếm số mệnh đề còn lại trong tập tổng hợp, số mệnh đề nhỏ nhất theo bệnh, và số bệnh phải dùng fallback.
5. **LLM:** số lần gọi, số JSON lỗi, số ô có `quote_invalid`, theo nhóm.

Đầu ra: `WeakSup/weak_labels/v2/survey/`, gồm các file CSV, các hình PNG và `survey_summary.md`. File tóm tắt chỉ ghi số liệu, không viết diễn giải.

---

## Phần C · Mẫu chấm tay

1. Lấy khoảng **150 cặp từ split test** theo các tầng, seed 42:

   | Tầng | Số cặp |
   |---|---|
   | `label_final=1`, nguồn văn bản, mệnh đề **cùng bệnh** với ca | 25 |
   | `label_final=1`, nguồn văn bản, mệnh đề **bệnh khác** | 25 |
   | `label_final=1`, chỉ nguồn ảnh | 25 |
   | `label_final=0` | 35 |
   | `label_final=-1` (để xem có bỏ sót dấu hiệu nào) | 40 |

2. Xuất `audit/audit_v2.csv` với các cột: `case_id`, `image_path`, `caption_original`, `proposition_id`, `canonicalDescription`, `category`, cột ẩn `teacher_label`, và cột **`human_label`** (1 / 0 / ? do Trí điền), `note`.
3. Xuất `audit/audit_v2.html` để xem ảnh, caption và mệnh đề. File HTML **không hiển thị nhãn của bộ gán nhãn**, để việc chấm không bị ảnh hưởng.
4. Sau khi Trí chấm xong, chạy ô tính:
   - precision của nhãn 1, precision của nhãn 0, tỉ lệ "∅ nhưng thực ra là 1";
   - tách theo nguồn (văn bản / ảnh), theo nhóm mệnh đề, và theo cùng bệnh / bệnh khác.

   Ghi kết quả vào `audit/audit_metrics.json`.

Mẫu chấm tay này dùng lại ở file 06 để đánh giá chính mô hình trên cùng các cặp.

---

## Xong khi

- Trí đã xác nhận dry-run và đã xem pilot 47 ca.
- `weak_labels/v2/` đã tạo và `meta.json` đầy đủ.
- Đã có `survey/` và Trí đã xem.
- Trí đã chấm xong mẫu và đã có `audit_metrics.json`.
- Nếu Trí quyết định đổi prompt, model hay ngưỡng ảnh: tạo `weak_labels/v3/` và chạy lại phần B và C. Không ghi đè phiên bản cũ.

## Không được làm

- Biến ∅ thành 0 ở bất kỳ bước nào.
- Gán nhãn mệnh đề dựa trên nhãn bệnh của ca, hay đưa nhãn bệnh hoặc tên bệnh vào prompt.
- Cho bộ gán nhãn đọc caption chưa che tên bệnh.
- Gọi API cho toàn bộ dữ liệu trước khi Trí xác nhận dry-run và pilot.
- Dùng nhãn của bộ gán nhãn để sửa ngược lại KB hay danh sách mệnh đề.
