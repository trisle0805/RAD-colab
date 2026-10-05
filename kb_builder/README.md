# kb_builder v2: dựng tập mệnh đề (KB) từ guideline

Đặt thư mục này ở gốc repo, cạnh `guideline/` và `data/`:
```
repo/
  guideline/qwen_maxtoken2k_skincap47_4sources.jsonl
  data/skincap_47_train.csv
  kb_builder/
```
Đặt chỗ khác thì sửa đường dẫn trong `config.yaml`.

## Cài đặt
```
pip install pyyaml anthropic openai
export ANTHROPIC_API_KEY=...     # nếu provider: anthropic
export OPENAI_API_KEY=...        # nếu provider: openai
```
Trong `config.yaml`: điền `llm.provider` (`anthropic` hoặc `openai`) và `llm.model` (bắt buộc). Model không hỗ trợ `temperature` thì đặt `temperature: null`.

## Chạy nhanh một nhóm bệnh

Dùng `00_run.py` để chạy từ segment đến tạo CSV review (script **không** tự finalize):

```bash
python 00_run.py --only psoriasis acne
python 00_run.py --random 3 --seed 42
```

- `--only`: chạy chính xác các bệnh đã nêu. Nếu bệnh đã có kết quả, kết quả trích được chạy lại và ghi đè; trạng thái duyệt cũ của bệnh đó bị xóa để duyệt lại.
- `--random N`: chọn ngẫu nhiên `N` bệnh chưa có kết quả trích. Dùng `--seed` nếu cần tái lập tập bệnh đã chọn.
- Mỗi lần cập nhật review, script sao lưu hai CSV hiện có thành `*.bak_YYYYMMDD_HHMMSS.csv`. Các bệnh không chạy lại giữ nguyên các cột `action`, `new_*`, `restore_*` và `note`.
- Thêm `--no-llm-dedup` để chỉ gộp trùng chữ ở bước dedup.

Sau khi duyệt CSV, tự chốt tập thử nghiệm bằng:

```bash
python 05_export.py finalize --partial
```

## 5 bước (chạy trong thư mục kb_builder/)

| Bước | Lệnh | Việc làm | Ai làm |
|---|---|---|---|
| 1 | `python 01_segment.py` | Chia 47 guideline thành từng câu (mỗi bullet / mỗi câu một mã) | Code |
| 2 | `python 02_extract.py --only psoriasis acne` để thử, rồi `python 02_extract.py` | Giữ/bỏ từng câu, tách câu giữ thành mệnh đề (quy tắc 1–6). Mỗi bệnh 1 lần gọi | LLM |
| 3 | `python 03_check.py` | Kiểm tra quy tắc 3, 4, 5, 6 → `outputs/check/flags.csv` | Code |
| 4 | `python 04_dedup.py` | Gộp mệnh đề trùng / gần trùng trong cùng bệnh (quy tắc 8). LLM chỉ trả mã | LLM |
| 5 | `python 05_export.py review` → duyệt CSV → `python 05_export.py finalize` | Người duyệt, rồi xuất file KB cuối, kiểm tra lại quy tắc 3–8 | Người + code |

- Bệnh nào bước 2 báo lỗi: `python 02_extract.py --force --only "<tên bệnh>"`.
- Bước 2 và 4 lưu lại prompt, phản hồi thô và thông tin model/token trong `outputs/extract/`, `outputs/dedup/`.

## Duyệt (bước 5)
Mở `outputs/review/review_propositions.csv` và `review_excluded_units.csv` bằng Excel/Google Sheets, lưu lại dạng CSV UTF-8, giữ nguyên tên file.
- Mệnh đề: cột `action` để trống = giữ; `edit` = sửa theo `new_canonical` / `new_polarity` / `new_category`; `delete` = bỏ.
- Câu bị bỏ: `action = restore` để khôi phục thành mệnh đề (điền `restore_canonical`, `restore_polarity`, `restore_category`).
- Nên xem kỹ: dòng có `ERROR` trong cột `flags`, mọi mệnh đề `polarity = -1`, cờ `OTHER_LABEL_NAME`, và danh sách câu bị bỏ.

## Đầu ra cuối: `outputs/final/`
- `kb_propositions.json`: file KB dùng cho mô hình. Định dạng giống `examples/kb_propositions_example_psoriasis.json`:
  danh sách 47 bệnh; mỗi bệnh có `disease_id`, `sourceReference`, `propositions`; mỗi mệnh đề có
  `proposition_id`, `category`, `canonicalDescription`, `polarity`, `sourceExcerpt`. Mệnh đề xếp theo nhóm
  morphology → location → symptom → history → lab.
- `kb_stats.md`, `kb_stats.json`: số liệu cho Chương 5.
- `review_log.csv`: các chỉnh sửa của người.

## 8 quy tắc (chi tiết trong `prompts/extract_v2.system.md`)
1. Mỗi mệnh đề là một ý chẩn đoán độc lập; tách khi là dấu hiệu bác sĩ kiểm tra riêng, giữ chung khi cùng mô tả một dấu hiệu.
2. Chỉ dấu hiệu của chính bệnh đó; bỏ điều trị, tiên lượng, xử trí, số liệu dịch tễ, bệnh khác.
3. Mô tả ngắn, tiếng Anh, khẳng định, không có từ phủ định hay từ chỉ khả năng.
4. Không bao giờ chứa tên bệnh (danh sách tên trong `synonyms.json`).
5. `polarity = -1` chỉ khi câu gốc nói rõ vắng mặt / bình thường / âm tính.
6. Đúng một trong 5 nhóm: morphology, location, symptom, history, lab.
7. `sourceExcerpt` là câu nguyên văn trong guideline.
8. Không có mệnh đề trùng hoặc gần trùng trong cùng bệnh; bản sơ sài gộp vào bản chi tiết.

## Thư mục examples/
- `psoriasis_extract_example.json`: đầu ra mẫu của bước 2 cho psoriasis (cũng là ví dụ nằm trong prompt).
- `kb_propositions_example_psoriasis.json`: đầu ra cuối mẫu (24 mệnh đề).

## Ghi lại cho luận văn
Tên model (trong `outputs/extract/*.meta.json`), phiên bản prompt (`extract_v2`, `dedup_v2`), nhiệt độ, ngày chạy, `kb_stats.md`.
