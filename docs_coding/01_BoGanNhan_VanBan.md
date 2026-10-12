# 01 · Bộ gán nhãn văn bản: fine-tune Bio_ClinicalBERT trên MedNLI

*Lý do: file câu chuyện, mục 5.4 (NLI là bài toán; bộ gán nhãn là bộ mã hoá BERT được fine-tune trên MedNLI).*

## Mục tiêu

Tạo một mô hình nhận vào **(tiền đề, giả thuyết)** và trả về xác suất của ba nhãn *entailment / neutral / contradiction*. Mô hình này chỉ dùng **offline** ở file 03, không nằm trong mô hình chính.

## Đầu vào

| Thứ | Nguồn |
|---|---|
| MedNLI v1.0.0: `mli_train_v1.jsonl`, `mli_dev_v1.jsonl`, `mli_test_v1.jsonl` | PhysioNet (Trí tải bằng tài khoản credentialed), để trong Drive riêng |
| Bộ mã hoá gốc | `emilyalsentzer/Bio_ClinicalBERT` (HuggingFace) |
| *(tuỳ chọn, để so sánh)* | `ygivenx/modernbert-base-mednli`, model gated nên Trí phải tự chấp nhận điều khoản |

Mỗi dòng MedNLI có các trường `sentence1` (tiền đề), `sentence2` (giả thuyết), `gold_label`.

## Việc cần làm (notebook `WS_01_teacher_text.ipynb`)

1. **Bảng nhãn cố định**, ghi vào `label_map.json`, và **không đổi về sau**:
   `{"entailment": 0, "neutral": 1, "contradiction": 2}`.
2. **Nạp mô hình:** `AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=3, id2label=..., label2id=...)`. Việc thư viện cảnh báo lớp phân loại "newly initialized" là bình thường.
3. **Token hoá theo cặp:** `tokenizer(premise, hypothesis, truncation=True, max_length=128)`.
4. **Fine-tune** bằng `transformers.Trainer`:

   | Tham số | Giá trị khởi điểm |
   |---|---|
   | learning rate | 2e-5 |
   | batch size | 32 |
   | epochs | 4 |
   | warmup ratio | 0.1 |
   | weight decay | 0.01 |
   | fp16 | bật nếu GPU hỗ trợ |
   | seed | 42 |

   Đánh giá trên **dev** sau mỗi epoch và giữ checkpoint có **accuracy dev** cao nhất (`load_best_model_at_end=True`).
5. **Đánh giá trên test một lần duy nhất:** accuracy, macro-F1, F1 từng lớp, ma trận nhầm lẫn.
6. **Thử nhanh trên câu kiểu da liễu.** Giả thuyết bọc theo mẫu `"The patient has {desc}."` (cùng mẫu với file 02). In ra xác suất ba nhãn của từng cặp:

   | # | Tiền đề | Giả thuyết | Kỳ vọng |
   |---|---|---|---|
   | 1 | "Multiple sharply demarcated red plaques with silvery scales." | "The patient has well-demarcated erythematous plaques." | entailment |
   | 2 | "Erythematous plaques without any scale." | "The patient has scaly plaques." | contradiction |
   | 3 | "Plaques with silvery scales on the elbows." | "The patient has nail pitting." | neutral |
   | 4 | "Ill-defined red patches on the cheeks." | "The patient has well-demarcated patches." | contradiction |
   | 5 | "Depigmented white patches with clear borders on the hands." | "The patient has chalk-white macules or patches." | entailment |
   | 6 | "Lesions on the face." | "The patient has lesions on the face, chest and back." | ghi nhận kết quả, không bắt buộc |

   Cặp số 6 dùng để quan sát cách mô hình xử lý câu liệt kê. File 02 sẽ tách các câu liệt kê, nên không cần sửa ở bước này.
7. *(Tuỳ chọn)* Chạy bước 5 và 6 với `ygivenx/modernbert-base-mednli` để có bảng so sánh. Trên thẻ của model này, entailment là lớp 0; phải kiểm tra lại `config.id2label` trước khi so sánh.

## Đầu ra

```text
WeakSup/teacher_text/bioclinicalbert_mednli_v1/
├── model/                 # save_pretrained (mô hình + tokenizer)
├── label_map.json
├── metrics.json           # dev theo từng epoch, test cuối, ma trận nhầm lẫn
├── sanity_derm.csv        # 6 cặp ở bước 6 với xác suất 3 lớp
└── meta.json              # tham số train, seed, phiên bản thư viện, SHA-256 ba file MedNLI
```

Nếu chạy bước 7: thêm `WeakSup/teacher_text/modernbert_mednli_eval/` gồm `metrics.json` và `sanity_derm.csv`.

## Xong khi

- Accuracy trên test MedNLI nằm khoảng 0,80–0,85, gần mức paper Bio_ClinicalBERT báo cáo **[kiểm lại con số trong Alsentzer et al., 2019]**. Nếu thấp hơn hẳn, kiểm tra lại bảng nhãn và cách ghép cặp.
- Cặp 1–3 trong bảng thử nhanh ra đúng nhãn kỳ vọng. Cặp 4–5 nên đúng; nếu sai thì ghi chú lại, không chặn việc chuyển sang file tiếp theo.
- Trí đã xem `metrics.json` và `sanity_derm.csv`.

## Trạng thái: đã xong (13/10/2026)

Trí chạy thêm một bộ thử mở rộng gồm 30 cặp tự soạn (`sanity_derm_extended.csv`, đặt cùng thư mục kết quả). Kết quả:

- Lấy nhãn có xác suất cao nhất: đúng 21/30 (dễ 9/10, trung bình 7/10, khó 5/10).
- Theo quy tắc ngưỡng: ở ngưỡng 0,9 / 0,8, nhãn 1 đúng 12/13 và nhãn 0 đúng 6/6; các ca còn lại rơi vào ∅.
- Kiểu lỗi chính là **phủ định nằm giữa câu** ("but not", "without", "sparing", "unaffected"), dẫn tới entailment sai. Đây là lý do có bước A2b (LLM kiểm tra lại) ở file 03.
- Ca 5, 7, 9, 25 trong bộ thử bị bọc mẫu hai lần ("The patient has pustules are present."). Nên sửa lại thành cụm danh từ rồi chạy lại 4 ca này. Không ảnh hưởng tới pipeline thật, vì ở đó giả thuyết sinh từ `canonicalDescription`.

Các con số này dùng để đặt ngưỡng ban đầu ở file 03 ($\tau^{+}_{txt}=0{,}9$, $\tau^{-}_{txt}=0{,}8$).

## Không được làm

- Dùng tập test để chọn checkpoint hay chỉnh tham số.
- Đổi `label_map.json` sau khi đã train.
- Đưa mô hình lên HuggingFace công khai.
- Dùng chung trọng số này với `text_encoder` của mô hình chính. Hai bộ này hoàn toàn tách biệt.
