# 02 · Chuẩn bị mệnh đề: tập quan sát được và câu hỏi cho bộ gán nhãn

*Lý do: file câu chuyện, mục 5.5 (tập mệnh đề quan sát được) và mục 6.2 (đọc mệnh đề theo nghĩa lâm sàng: liệt kê vị trí hoặc dạng thay thế thì chỉ cần có một).*

## Trạng thái: dùng bảng v2 (14/10/2026)

**Bảng đang dùng là `propositions_v2/propositions_review_final.csv`.** Agent **không tạo lại bảng**; agent chỉ chạy **bước 5** với file này và ghi ra `WeakSup/propositions/v2/`. Thư mục `propositions/v1/` giữ nguyên, không ghi đè.

v2 khác v1 đúng 27 dòng, mỗi dòng có ghi chú `[Claude-v2]` ở cột `note`. Trí duyệt các dòng này trước khi chạy:

- **10 mệnh đề chuyển thành không quan sát được**, vì đúng với mọi ca da liễu hoặc không mang thông tin (quy tắc 4 bên dưới): "skin involvement" (2), "skin symptoms", "cutaneous symptoms" (2), "cutaneous manifestations", "underlying skin disorders", "variable lesion size and number", "lesion size", "diameter ranging from a few millimeters to several centimeters".
- **12 mệnh đề kích thước tắt nhãn ảnh** (`image_applicable=0`), vì ảnh không có thước đo. Văn bản vẫn gán nhãn được nếu caption ghi kích thước.
- **5 mệnh đề về bên trong mắt tắt nhãn ảnh**: hyphema, glaucoma, uveitis, ocular manifestations, corneal ulcers.

Kết quả v2: **978 mệnh đề quan sát được** (morphology 315, location 122, symptom 154, history 387); 298 mệnh đề có `image_applicable=1`; mỗi bệnh có từ 11 đến 39 mệnh đề, trung vị 19. Thống kê theo bệnh: `propositions_v2/summary_per_disease.csv`.

Quy tắc bổ sung từ v2: **mệnh đề chỉ nói về kích thước → `image_applicable=0`.**

### Bảng v1 (13/10/2026, giữ để đối chiếu)

Bảng v1 là `propositions_v1/propositions_review_final.csv`. Trí đã duyệt bảng này; Claude sửa thêm 6 mệnh đề luôn đúng (xem quy tắc 4 bên dưới). Lịch sử và lý do sửa của từng dòng nằm ở cột `note`, có tiền tố `[GPT-final]` hoặc `[Claude-final]`.

Kết quả của bảng v1:

| Nhóm | Quan sát được | Không |
|---|---:|---:|
| morphology | 320 | 1 |
| location | 123 | 58 |
| symptom | 157 | 0 |
| history | 388 | 4 |
| lab | 0 | 385 |
| **Tổng** | **988** | 448 |

- 320 mệnh đề có `image_applicable=1`.
- Mỗi bệnh có từ 11 đến 39 mệnh đề quan sát được, trung vị 19.
- Thống kê theo bệnh: `propositions_v1/summary_per_disease.csv`.

Các quy tắc đã áp dụng khi duyệt, dùng lại cho những lần tạo phiên bản sau:

1. **"Quan sát được" nghĩa là có thể xác định đáng tin cậy từ ảnh *hoặc* caption.** Vì vậy bệnh sử và triệu chứng được giữ để gán nhãn từ văn bản. Xét nghiệm, sinh thiết, chẩn đoán hình ảnh, cùng tổn thương ở cơ quan bên trong hoặc toàn thân (hạch, khớp, màng phổi, màng ngoài tim, ruột, nội tạng, di căn…) thì loại.
2. **Tách các dạng thay thế:** khi tách thì **lặp lại danh từ chính** cho mỗi vế, ví dụ "large lesions ‖ thick lesions", "gray lesions ‖ pink lesions ‖ red lesions ‖ skin-colored lesions".
3. **Không tách** các trường hợp sau: dấu hiệu có tên riêng ("shawl sign", "mechanics' hands", "butterfly rash"); kích thước; diễn tiến theo thời gian; vị trí có trạng từ chen giữa ("…, especially the legs"). Các câu này giữ nguyên.
4. **Mệnh đề luôn đúng với mọi ca → 0.** Đó là các mệnh đề mà các vế cộng lại bao hết mọi khả năng, nên không mang thông tin: "isolated or numerous lesions", "solitary or grouped lesions", "male or female sex", "any age", "onset at any age", "onset in children or adults".
5. **Vị trí:** mỗi câu hỏi con phải đúng ngữ pháp và đủ nghĩa.

**Lưu ý dành cho file 03.** Bệnh sử và triệu chứng trong caption SkinCAP thường nằm trong đoạn kiến thức chung về bệnh, không phải mô tả của riêng bệnh nhân. Prompt của LLM (file 03, A2) yêu cầu bỏ qua các câu kiểu này.

---

## Mục tiêu

Từ KB, tạo **một bảng duy nhất** để Trí duyệt. Ở đây, một mệnh đề được xem là **quan sát được** nếu có thể xác định đáng tin cậy từ ảnh da **hoặc caption lâm sàng** của SkinCAP; không giới hạn ở dấu hiệu nhìn trực tiếp trên ảnh. Bảng ghi, cho từng mệnh đề:

1. Mệnh đề có **quan sát được** trên SkinCAP không.
2. Bộ gán nhãn **ảnh** có áp dụng cho mệnh đề này không.
3. Danh sách **câu hỏi con** (sub-hypotheses) dùng để hỏi bộ gán nhãn.

KB gốc giữ nguyên, không sửa.

## Đầu vào

- `kb_propositions.json` (47 bệnh, 1.436 mệnh đề; các trường `proposition_id`, `category`, `canonicalDescription`, `polarity`, `sourceExcerpt`).

## Việc cần làm (notebook `WS_02_propositions.ipynb`, hàm trong `proposition/weak/propositions.py`)

> Với phiên bản v1, bước 1–4 **đã xong**: bảng đã được duyệt (xem mục Trạng thái). Phần dưới đây giữ lại để dùng khi tạo phiên bản sau.

### Bước 1. Đề xuất tự động "quan sát được"

| Điều kiện | `auto_observable` |
|---|---|
| Bệnh sử hoặc triệu chứng có thể được nêu trực tiếp trong caption (ví dụ: family history, onset, itching, pain, fatigue) | `1`, gắn cờ `review_needed=1` |
| Xét nghiệm, sinh thiết, chẩn đoán hình ảnh, hoặc thông tin chỉ có trong hồ sơ mà không thể hiện đáng tin cậy trong caption | `0` |
| `category` ∈ {morphology, location} **và** mô tả nói về cơ quan bên trong hoặc toàn thân | `0` |
| `category` ∈ {morphology, location} còn lại | `1` |
| Trường hợp lưng chừng (mắt, kết mạc, niêm mạc miệng/sinh dục, toàn thân nhưng có biểu hiện da) | `1`, gắn cờ `review_needed=1` |
| Mệnh đề luôn đúng với mọi ca (các vế bao hết mọi khả năng) | `0` |

Danh sách từ khoá gợi ý cho "cơ quan bên trong hoặc toàn thân" (Trí có thể chỉnh): lung(s), pulmonary, kidney, renal, liver, heart, cardiac, pericardium/pericardial, pleura/pleural, brain, central nervous system, muscle, weakness, lymph node(s), lymphadenopathy, arthritis, joint(s), spine, metastases, gastrointestinal, bowel, rectal, pancreas, omentum, testis, cervix/cervical, oropharynx, hypertension, internal organs, visceral, multiorgan/multisystem, dissemination, larval migration, parasitic infection, deep tissue. **Không** loại vì từ "muscle" khi nó chỉ một vùng da (vd. "triangular muscle").

### Bước 2. Bộ gán nhãn ảnh có áp dụng không

`image_applicable = 1` khi và chỉ khi `observable = 1` **và** `category = morphology`. Vì vậy, bệnh sử và triệu chứng được giữ để gán nhãn từ văn bản nhưng có `image_applicable = 0`. Mệnh đề vị trí cũng không dùng nhãn từ ảnh, vì ảnh không chụp tới một vị trí không có nghĩa là không có tổn thương ở đó (file câu chuyện, mục 6.2).

### Bước 3. Phân loại cách diễn đạt và tách câu hỏi con

Với mỗi mệnh đề quan sát được, gán `expression_type`:

| `expression_type` | Nhận biết | Ví dụ | Tách thành câu hỏi con |
|---|---|---|---|
| `single` | không có "and / or / ," | "nail pitting" | giữ nguyên 1 câu |
| `site_list` | dạng "<tổn thương> on/in A, B and/or C" | "lesions on the face, chest and back" | "lesions on the face" ‖ "lesions on the chest" ‖ "lesions on the back" |
| `alternatives` | các dạng nối bằng "or" có chung đầu ngữ hoặc đuôi ngữ | "papules or nodules", "gray pink red or skin-colored lesions" | "papules" ‖ "nodules"; "gray lesions" ‖ "pink lesions" ‖ … (luôn lặp lại danh từ chính) |
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
| `note` | ghi chú |

Sắp xếp sao cho các dòng `review_needed=1` nằm lên đầu.

### Bước 5. Chốt phiên bản (chạy với `propositions_review_final.csv`)

Đọc CSV đã duyệt, kiểm tra hợp lệ, rồi ghi:

```text
WeakSup/propositions/v<k>/          # v2 với bảng hiện tại
├── propositions_review.csv     # = propositions_review_final.csv
├── propositions_final.json     # chỉ các mệnh đề final_observable=1:
│                               #   proposition_id, disease_id, category, polarity,
│                               #   image_applicable, sub_hypotheses[list], hypotheses[list đã bọc mẫu]
├── summary_per_disease.csv     # số mệnh đề quan sát được theo bệnh và theo nhóm
└── meta.json                   # SHA-256 KB và CSV đã duyệt, mẫu câu hỏi, từ khoá, ngày duyệt
```

Kiểm tra hợp lệ (assert):

- Mỗi bệnh có ít nhất 1 mệnh đề `final_observable=1`.
- Mọi `proposition_id` tồn tại trong KB.
- Không có câu hỏi con nào rỗng.
- Mệnh đề `final_observable=0` phải có `image_applicable=0`.
- Thứ tự mệnh đề trong `propositions_final.json` theo đúng thứ tự trong KB. Thứ tự này cố định cột của ma trận nhãn và ma trận $S$ về sau.

## Xong khi

- `propositions/v2/` đã được tạo từ `propositions_v2/propositions_review_final.csv` và qua mọi assert.

## Không được làm

- Sửa `kb_propositions.json`, hay sửa `propositions_review_final.csv`.
- Tự quyết các dòng `review_needed=1` mà không qua Trí duyệt.
- Dùng LLM để viết lại `canonicalDescription`. Tách câu hỏi chỉ bằng quy tắc, phần còn lại do Trí sửa tay.
