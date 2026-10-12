# 02 · Chuẩn bị mệnh đề: tập quan sát được và câu hỏi cho bộ gán nhãn

*Lý do: file câu chuyện, mục 5.5 (tập mệnh đề quan sát được) và mục 6.2 (đọc mệnh đề theo nghĩa lâm sàng: liệt kê vị trí hoặc dạng thay thế thì chỉ cần có một).*

## Mục tiêu

Từ KB, tạo **một bảng duy nhất** để Trí duyệt. Ở đây, một mệnh đề được xem là **quan sát được** nếu có thể xác định đáng tin cậy từ ảnh da **hoặc caption lâm sàng** của SkinCAP; không giới hạn ở dấu hiệu nhìn trực tiếp trên ảnh. Bảng ghi, cho từng mệnh đề:

1. Mệnh đề có **quan sát được** trên SkinCAP không.
2. Bộ gán nhãn **ảnh** có áp dụng cho mệnh đề này không.
3. Danh sách **câu hỏi con** (sub-hypotheses) dùng để hỏi bộ gán nhãn.

KB gốc giữ nguyên, không sửa.

## Đầu vào

- `kb_propositions.json` (47 bệnh, 1.436 mệnh đề; các trường `proposition_id`, `category`, `canonicalDescription`, `polarity`, `sourceExcerpt`).

## Việc cần làm (notebook `WS_02_propositions.ipynb`, hàm trong `proposition/weak/propositions.py`)

### Bước 1. Đề xuất tự động "quan sát được"

| Điều kiện | `auto_observable` |
|---|---|
| Bệnh sử hoặc triệu chứng có thể được nêu trực tiếp trong caption (ví dụ: family history, onset, itching, pain, fatigue) | `1`, gắn cờ `review_needed=1` |
| Xét nghiệm, sinh thiết, chẩn đoán hình ảnh, hoặc thông tin chỉ có trong hồ sơ mà không thể hiện đáng tin cậy trong caption | `0` |
| `category` ∈ {morphology, location} **và** mô tả nói về cơ quan bên trong hoặc toàn thân | `0` |
| `category` ∈ {morphology, location} còn lại | `1` |
| Trường hợp lưng chừng (mắt, kết mạc, niêm mạc miệng/sinh dục, toàn thân nhưng có biểu hiện da) | `1`, gắn cờ `review_needed=1` |

Danh sách từ khoá gợi ý cho "cơ quan bên trong hoặc toàn thân" (Trí có thể chỉnh): lung, pulmonary, kidney, renal, liver, heart, cardiac, pericardium, pleura, brain, central nervous system, muscle, weakness, lymph node, metastases, gastrointestinal, pancreas, omentum, testis, cervix, oropharynx, hypertension, internal organs, multisystem, larval migration, parasitic infection.

Kiểm tra thử trên KB hiện tại: khoảng 50 trong 502 mệnh đề morphology/location bị loại, còn khoảng 452. Mỗi bệnh còn ít nhất 2 mệnh đề (ít nhất là nematode infection). Con số cuối cùng tuỳ Trí duyệt.

### Bước 2. Bộ gán nhãn ảnh có áp dụng không

`image_applicable = 1` khi và chỉ khi `observable = 1` **và** `category = morphology`. Vì vậy, bệnh sử và triệu chứng được giữ để gán nhãn từ văn bản nhưng có `image_applicable = 0`. Mệnh đề vị trí cũng không dùng nhãn từ ảnh, vì ảnh không chụp tới một vị trí không có nghĩa là không có tổn thương ở đó (file câu chuyện, mục 6.2).

### Bước 3. Phân loại cách diễn đạt và tách câu hỏi con

Với mỗi mệnh đề quan sát được, gán `expression_type`:

| `expression_type` | Nhận biết | Ví dụ | Tách thành câu hỏi con |
|---|---|---|---|
| `single` | không có "and / or / ," | "nail pitting" | giữ nguyên 1 câu |
| `site_list` | dạng "<tổn thương> on/in A, B and/or C" | "lesions on the face, chest and back" | "lesions on the face" ‖ "lesions on the chest" ‖ "lesions on the back" |
| `alternatives` | các dạng nối bằng "or" có chung đầu ngữ hoặc đuôi ngữ | "papules or nodules", "gray pink red or skin-colored lesions" | "papules" ‖ "nodules"; "gray lesions" ‖ "pink lesions" ‖ … |
| `pattern` | một dấu hiệu kinh điển gồm nhiều đặc điểm ("with … and …") | "bull's-eye rash with central red spot, central clearing and outer red ring" | **không tách**, 1 câu |
| `and_pair` | "A and B" mà không phải danh sách vị trí | "papules and plaques", "vesicles and crusts" | mặc định không tách; Trí quyết khi duyệt |

Quy tắc tách tự động chỉ áp dụng cho hai mẫu rõ ràng `site_list` và `alternatives`. Mọi dòng được tách đều gắn `review_needed=1`.

Câu hỏi đưa vào bộ gán nhãn văn bản được bọc theo mẫu cố định:

```text
hypothesis = "The patient has {sub_hypothesis}."
```

Mẫu này ghi trong `meta.json`. Đổi mẫu thì phải tạo phiên bản mới.

### Bước 4. Xuất bảng để Trí duyệt

`propositions_review.csv`, một dòng cho mỗi mệnh đề trong KB (đủ 1.436 dòng):

| Cột | Ý nghĩa |
|---|---|
| `proposition_id`, `disease_id`, `category`, `polarity`, `canonicalDescription` | sao từ KB |
| `auto_observable`, `auto_reason` | đề xuất của bước 1 và lý do (từ khoá khớp, hoặc nhóm) |
| `image_applicable` | bước 2 |
| `expression_type`, `sub_hypotheses` | bước 3; các câu hỏi con cách nhau bằng ` ‖ ` |
| `review_needed` | 0/1 |
| `final_observable` | **Trí điền** (0/1); mặc định copy từ `auto_observable` |
| `final_sub_hypotheses` | **Trí sửa nếu cần**; mặc định copy từ `sub_hypotheses` |
| `note` | ghi chú của Trí |

Sắp xếp sao cho các dòng `review_needed=1` nằm lên đầu.

### Bước 5. Chốt phiên bản (chạy sau khi Trí duyệt xong)

Đọc lại CSV đã duyệt, kiểm tra hợp lệ, rồi ghi:

```text
WeakSup/propositions/v1/
├── propositions_review.csv     # bản Trí đã duyệt
├── propositions_final.json     # chỉ các mệnh đề final_observable=1:
│                               #   proposition_id, disease_id, category, polarity,
│                               #   image_applicable, sub_hypotheses[list], hypotheses[list đã bọc mẫu]
├── summary_per_disease.csv     # số mệnh đề quan sát được theo bệnh và theo nhóm
└── meta.json                   # SHA-256 KB, mẫu câu hỏi, từ khoá, ngày duyệt
```

Kiểm tra hợp lệ (assert):

- Mỗi bệnh có ít nhất 1 mệnh đề `final_observable=1`.
- Mọi `proposition_id` tồn tại trong KB.
- Không có câu hỏi con nào rỗng.
- Thứ tự mệnh đề trong `propositions_final.json` theo đúng thứ tự trong KB. Thứ tự này cố định cột của ma trận nhãn và ma trận $S$ về sau.

## Xong khi

- Trí đã duyệt `propositions_review.csv`, ưu tiên các dòng `review_needed=1`.
- `propositions/v1/` đã tạo và qua mọi assert.

## Không được làm

- Sửa `kb_propositions.json`.
- Tự quyết các dòng `review_needed=1` mà không qua Trí duyệt.
- Dùng LLM để viết lại `canonicalDescription`. Tách câu hỏi chỉ bằng quy tắc, phần còn lại do Trí sửa tay.
