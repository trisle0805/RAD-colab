# Chuyển code RAD sang kiến trúc dựa trên mệnh đề

Tài liệu cho agent coding. Mỗi hạng mục trình bày theo khuôn: **RAD gốc làm gì → Mô hình đề xuất đổi gì → Tại sao**.

- Nội dung chỉ mô tả *cần làm gì* và *vì sao*. Agent tự quyết cách viết code, miễn đúng các ràng buộc ở đây.
- Cơ sở lý thuyết là Chương 4 luận văn: Mục 4.2 (tri thức, cấu hình mức bệnh), 4.3 (chuỗi dữ liệu, mã hóa mệnh đề), 4.4 (PECL), 4.5 (dự đoán dựa trên mệnh đề), 4.6 (nhánh nhãn bệnh), 4.7 (loss tổng), 4.8 (bảng kiểm).
- **Chương 4 (bản hiện tại) là nguồn chuẩn.** Nếu tài liệu này và các file `.tex` mâu thuẫn nhau thì làm theo `.tex`, ghi lại chỗ mâu thuẫn và báo lại.
- Các quyết định thiết kế chưa có trong Chương 4 được đánh mã **[QĐ-x]** và tổng hợp ở mục 12. Người dùng có thể đổi các quyết định này sau.

> **Phiên bản 3.** Thay đổi so với bản 2:
> - bổ sung 7 điểm theo phản hồi đợt 2 của agent (xem mục 14b);
> - thêm **mục 15: đầu ra phục vụ viết Chương 5**, gồm danh sách lần chạy, thông tin phải lưu và script phân tích;
> - thêm QĐ-15 đến QĐ-18.
>
> **Phiên bản 2.** Đồng bộ với Chương 4 mới. Thay đổi so với bản 1:
> - cross-attention đổi sang cosine/τ_att;
> - PECL đổi sang target 1/0, lấy trung bình riêng trên tập dương và tập âm;
> - bỏ ClipLoss khỏi cấu hình chính;
> - bổ sung 10 điểm nhỏ theo phản hồi của agent (xem mục 14).

Repo: `trisle0805/RAD-colab`, nhánh `RAD-colab-enhanced`.

---

## 0. Ràng buộc chung (bắt buộc)

1. **Không sửa code baseline RAD.** Giữ nguyên `main_rad.py`, `engine/train_rad.py`, `models/clip_tqn.py`, `factory/loss.py`, `optim/`. RAD phải chạy lại được y như cũ trên cùng nhánh. Code mô hình dựa trên mệnh đề nằm trong file mới. Được phép import và tái sử dụng hàm hoặc lớp của RAD.
2. **Không sửa dữ liệu và tri thức.** Không đụng `guideline/`, các CSV `skincap_47_*` và `kb_builder/outputs/final/kb_propositions.json`.
3. **Giữ nguyên mọi thứ ngoài những điểm Chương 4 thay đổi** để so sánh công bằng với RAD:
   - encoder ảnh ResNet50 và ClinicalBERT;
   - chuỗi dữ liệu bệnh nhân;
   - augmentation, optimizer, scheduler, batch size, số epoch, seed;
   - giao thức train/val/test;
   - toàn bộ hàm metrics.
4. **Không hard-code các con số của SkinCAP trong lớp mô hình.** Số bệnh C, số mệnh đề J, số patch ảnh và độ dài caption lấy từ dữ liệu, KB và tensor thực tế. Các con số 47, 1436, 31, 256 và 512 chỉ xuất hiện trong **kiểm tra tích hợp cho SkinCAP** (mục 11). Nhờ vậy sau này dùng lại cho dữ liệu bệnh lý xương mà không phải sửa model.
5. **Không tự thêm thành phần ngoài tài liệu này:** không thêm lớp, loss hay trick huấn luyện. Thấy cần thì ghi vào phần "Câu hỏi mở" khi bàn giao, không tự làm.

---

## 1. Tổ chức file

| File mới | Nội dung |
|---|---|
| `dataset/kb.py` | Đọc và kiểm tra `kb_propositions.json`, tạo các tensor chỉ mục (mục 2) |
| `models/proposition_model.py` | Module cross-attention dùng chung, đường dự đoán mệnh đề, nhánh nhãn bệnh (mục 3, 4) |
| `factory/pecl_loss.py` | PECL (mục 5) |
| `engine/train_proposition.py` | Vòng train / val / test cho mô hình dựa trên mệnh đề (mục 6–9), viết dựa trên `engine/train_rad.py` |
| `main_proposition.py` | Điểm vào, viết dựa trên `main_rad.py`; checkpoint và resume giống RAD |
| `K34_Proposition_Model.ipynb` (hoặc thêm phần mới trong notebook hiện có) | Chạy trên Colab, cùng đường dẫn dữ liệu với notebook RAD |
| `tests/test_proposition_model.py` | Các kiểm tra ở mục 11 |
| `analysis/` | Các script phân tích cho Chương 5 (mục 15.3, làm ở đợt 2) |

Ghi chú: chạy `git rm --cached` cho các file `__pycache__` đang bị track. Việc này nên làm trong một commit riêng.

---

## 2. Đọc tri thức: guideline → tập mệnh đề

**RAD gốc làm gì.** `train_rad.py` (khoảng dòng 252–287):

- đọc 47 dòng `guideline_1_content` từ file jsonl;
- kiểm tra `query` khớp với tên nhãn;
- cắt thêm đoạn "Radiological / Imaging" của mỗi guideline làm prototype ảnh cho GECL.

Việc này lặp lại ở mỗi batch.

**Mô hình đề xuất đổi gì.**

- Thêm tham số `--kb_path`, mặc định `kb_builder/outputs/final/kb_propositions.json`.
- Đọc một lần khi khởi động:
  - 47 bệnh, 1436 mệnh đề;
  - mỗi mệnh đề lấy `canonicalDescription` (gọi là t_j), `polarity` (ε_j ∈ {+1, −1}) và `disease_index` (vị trí bệnh theo **thứ tự header CSV**).
- Kiểm tra bắt buộc, sai thì dừng chương trình:
  - thứ tự `disease_id` trong KB phải trùng đúng thứ tự `skin_label_list` lấy từ header CSV;
  - không có bệnh nào rỗng.
- Tạo sẵn:
  - `prop_disease` (LongTensor, 1436), `prop_polarity` (tensor ±1, 1436);
  - danh sách **văn bản duy nhất** cần mã hóa, gồm t_j và văn bản căn chỉnh t_j^align (mục 5);
  - chỉ mục từ mệnh đề về văn bản duy nhất.

  KB có 1283 `canonicalDescription` khác nhau, cộng thêm khoảng 31 văn bản "absence of …", nên mỗi bước chỉ cần mã hóa khoảng 1314 câu thay vì 1436 + 1436.

**Tại sao.** Thay đổi trung tâm của luận văn (Mục 4.1) là đưa đơn vị tri thức từ mức bệnh xuống mức mệnh đề. Mã hóa văn bản trùng một lần vừa tiết kiệm vừa bảo đảm cùng một dấu hiệu luôn có cùng một embedding ở mọi bệnh.

---

## 3. Đường dự đoán: nhánh guideline → mệnh đề làm query (Mục 4.5)

**RAD gốc làm gì.**

1. Guideline được ClinicalBERT mã hóa 512 token. Query của mỗi bệnh chỉ là **16 token đầu** (`guideline_last_hidden_state[:, :16, :]`). Đoạn đầu guideline luôn là tiêu đề ("### Summary of Key Diagnostic Features for Acne Vulgaris…"), nên query gần như chỉ chứa tên bệnh.
2. `TQN_Model_fusion` là Transformer decoder 4 lớp, 4 head, có FFN 1024. Key/value là `fusion_features` = [patch ảnh; token caption].
3. Mỗi bệnh cho 1 logit qua `Linear(768→1)`, rồi tính BCE.
4. Lỗi đã biết: do cách `repeat`, mẫu thứ b trong batch dùng token thứ (b mod 16) làm query. **Không sửa lỗi này trong RAD.**

**Mô hình đề xuất đổi gì.** Với batch có B ca bệnh, J mệnh đề, C bệnh, d = 768 (SkinCAP: J = 1436, C = 47):

1. **Chuỗi bệnh nhân H** giữ nguyên `fusion_features` của RAD. Với SkinCAP đó là 256 patch ảnh (ảnh 512 px, lưới 16×16) nối với 512 token caption, kích thước (B, 768, d). Không thêm phép chiếu mới **[QĐ-1]**.
2. **Query mệnh đề q_j.** Mã hóa t_j bằng **cùng `text_encoder` của RAD**, lấy đầu ra pooled của `encode_text` (CLS → `mlp_embed`).
   - `max_length` **đo bằng đúng tokenizer ClinicalBERT** trên toàn bộ văn bản query và văn bản căn chỉnh, rồi đặt bằng độ dài lớn nhất đo được, làm tròn lên bội số của 8. **Không được cắt mệnh đề nào**; thêm assert kiểm tra điều này **[QĐ-2]**.
   - Mã hóa lại ở mỗi bước huấn luyện, có gradient, giống cách RAD mã hóa guideline mỗi batch **[QĐ-3]**.
3. **Cross-attention theo đúng công thức 4.5: một head, cosine.**
   - Có đúng 1 lớp, 1 head, không FFN, không residual, không LayerNorm.
   - W_Q, W_K, W_V có học, kích thước d×d.
   - Công thức:
     - e_ijn = cos(q_j W_Q, h_in W_K) / τ_att;
     - A_ijn = softmax theo n;
     - c_ij = Σ_n A_ijn · (h_in W_V).
   - **τ_att = 0.1, cố định, đặt qua tham số `--tau_att`** **[QĐ-4]**.
   - **Dùng key padding mask cho các token PAD của caption** **[QĐ-5]**.
   - Trả về c (B, J, d) và A (B, J, L).
4. **Scorer dùng chung f_θ.** MLP 2 lớp: [c_ij; q_j] (2d) → d → GELU → 1, cho ra r_ij **[QĐ-6]**. Sau đó s_ij = σ(r_ij).
5. **Polarity.** s̃_ij = s_ij nếu ε_j = +1; s̃_ij = 1 − s_ij nếu ε_j = −1.
6. **Tổng hợp.** z_ic = trung bình s̃_ij trên các mệnh đề của bệnh c (dùng `prop_disease`, kiểu scatter-mean). Mỗi bệnh chỉ dùng mệnh đề của chính nó.
   - Cấu hình ablation `--agg weighted`: z_ic = Σ_j w_cj · s̃_ij, với w_cj = softmax của a_cj **chỉ trong tập mệnh đề của bệnh c**; a_cj khởi tạo 0, tức bắt đầu đúng bằng trung bình đều. Khi dùng cấu hình này, xuất w_cj vào bảng kiểm (mục 9).
7. **Hiệu chỉnh.**
   - o_ic = γ_c · z_ic + b_c, với γ_c = softplus(ρ_c), bảo đảm γ_c > 0.
   - Khởi tạo γ_c ≈ 10, b_c = −5 **[QĐ-7]**.
   - ŷ_ic = σ(o_ic).
8. **Loss.** L_prop = BCE-with-logits(o, y), lấy trung bình trên B×C, giống cách RAD tính BCE.

Đầu ra có dạng (B, C), giống RAD.

**Tại sao.**

- Mỗi dấu hiệu có điểm hỗ trợ s_ij riêng **nằm ngay trên đường tạo dự đoán**. Đây là nền của bảng kiểm truy ngược (Mục 4.8).
- Query đọc toàn bộ 1436 dấu hiệu thay vì vài token tiêu đề, và không chứa tên bệnh.
- Mệnh đề j luôn là cùng một query cho mọi mẫu, nên không có lỗi batch-position.
- H, encoder và dạng đầu ra giống RAD, nên khác biệt kết quả đến từ cách dùng tri thức. Code metrics dùng lại nguyên vẹn.
- Một head, một lớp, cosine: làm đúng Mục 4.5. Mỗi mệnh đề có một phân bố A_ij duy nhất gắn trực tiếp với c_ij, nên đọc thẳng được cho bảng kiểm.

---

## 4. Nhánh nhãn bệnh: decoder nhãn của RAD → nhánh bổ trợ (Mục 4.6)

**RAD gốc làm gì.**

- `model = TQN_Model_fusion` dùng 16 token đầu của tên bệnh làm query, cùng H, ra logit, tính BCE. Nhánh này cũng mắc lỗi batch-position; vì tên bệnh rất ngắn, nhiều mẫu thực chất dùng token PAD làm query.
- Khi đánh giá, RAD trộn điểm hai nhánh (nhãn và guideline). Trên SkinCAP, trọng số trộn theo từng lớp được học trên val (`_fit_skin_fusion_weights`).

**Mô hình đề xuất đổi gì.**

- **Query nhãn.** q_c^label = đầu ra pooled của `encode_text` trên tên bệnh c (một vector cho mỗi bệnh).
- **Đọc dữ liệu.** Dùng cùng loại module cross-attention ở mục 3 nhưng **tham số riêng**, cùng H, cùng padding mask, cho ra h_ic.
- **Scorer riêng.** MLP [h_ic; q_c^label] → logit o_ic^label. L_label = BCE.
- **Vai trò của nhánh, nói chính xác:**
  - chỉ tham gia **hàm mục tiêu** khi train;
  - **được chạy** ở val/test để lưu điểm phục vụ phân tích;
  - **tuyệt đối không** tham gia dự đoán chính, việc chọn epoch, hay bất kỳ phép trộn nào.
- **Khi `--no_label_branch`** (hoặc `--lambda_label 0`): không tạo nhánh. **Bỏ hẳn** `pred_label` trong `.npz`, `label_aux` trong `final_metrics.json` và các cột `label_*` trong `metrics_history.csv`. Không ghi `null`, không tạo dữ liệu giả.
- Không dùng lại `TQN_Model_fusion` cho nhánh này **[QĐ-8]**.

**Tại sao.**

- Mục 4.6 giữ nhánh nhãn như tín hiệu giám sát ngắn hơn và kiểm tra giá trị của nó bằng ablation `--no_label_branch`.
- Viết lại bằng cùng module với đường mệnh đề để đúng công thức 4.6 (một vector query cho mỗi bệnh) và tránh lỗi batch-position. Bản thân nhánh này không phải đóng góp của luận văn.

---

## 5. GECL → PECL (Mục 4.4)

**RAD gốc làm gì.** `SupConLoss` (`factory/loss.py`):

- Mỗi bệnh có 1 prototype.
  - Prototype văn bản: guideline pooled, có gradient, so với caption pooled, τ = 0.5, hệ số 0.1.
  - Prototype ảnh: đoạn "Radiological" của guideline, tính trong `no_grad`, so với ảnh pooled, τ = 2.0, hệ số 0.001.
- Với mỗi mẫu:
  - positive = các bệnh có nhãn 1;
  - lấy ngẫu nhiên tối đa 5×|positive| bệnh âm;
  - loss = BCE-with-logits trên similarity / τ, với target 1/|P| cho positive và 0 cho negative.

**Mô hình đề xuất đổi gì.**

1. **Văn bản căn chỉnh.** t_j^align = t_j nếu ε_j = +1; `"absence of " + t_j` nếu ε_j = −1. a_j = đầu ra pooled của `encode_text`, dùng chung lượt mã hóa với mục 3.
   - Đã kiểm tra tay cả 31 câu "absence of …" của KB hiện tại: tất cả đều hợp nghĩa (ví dụ "absence of dermal invasion", "absence of lesions on the face"). Agent vẫn in danh sách này trong kiểm tra 1.
2. **Với mỗi mẫu i:**
   - **P_i** = a_j của mọi mệnh đề thuộc các bệnh dương, **bỏ trùng theo văn bản** t_j^align.
   - **N_i** = a_j của mệnh đề thuộc các bệnh âm, **loại những văn bản trùng chính xác với một văn bản trong P_i**, rồi bỏ trùng.
   - **Q_i** = lấy ngẫu nhiên min(r·|P_i|, |N_i|) phần tử từ N_i, với **r = 5** như RAD.
   - S_i = P_i ∪ Q_i.
3. **Loss: làm đúng công thức 4.4 (bản mới). Khác `SupConLoss` của RAD.**
   - φ = cos(z_i, a_j) / τ, với z_i và a_j đều được chuẩn hóa L2.
   - Loss của mẫu i = −[ (1/|P_i|) Σ_{P_i} log σ(φ) + 𝟙[|Q_i|>0] · (1/|Q_i|) Σ_{Q_i} log(1 − σ(φ)) ].
     - Target là **1** cho positive và **0** cho negative.
     - Hai tập được lấy trung bình **riêng**.
     - Nếu Q_i rỗng, bỏ hạng negative.
   - Loss của modality = **tổng loss các mẫu có |P_i| > 0, chia cho N là kích thước batch**, không chia cho số mẫu hợp lệ **[QĐ-13]**.
4. **Hai modality, giữ τ và hệ số của RAD:**

   | Modality | Đặc trưng ca bệnh | τ | Hệ số α | Prototype |
   |---|---|---|---|---|
   | Văn bản | caption pooled | 0.5 | 0.1 | a_j có gradient |
   | Ảnh | ảnh pooled | 2.0 | 0.001 | a_j.detach() **[QĐ-9]** |

   Cả hai modality dùng **toàn bộ** mệnh đề, không lọc theo nhóm.
5. Mệnh đề polarity −1 thuộc bệnh dương **vẫn là positive**; chỉ văn bản căn chỉnh của nó mang nghĩa vắng mặt.

**Tại sao.**

- PECL giữ cách xây tập đối sánh, cách lấy mẫu negative và similarity theo prototype của GECL.
- **Không** giữ target 1/|P_i|. Ở mức mệnh đề, |P_i| là số mệnh đề của bệnh dương (khoảng 15–51), nên target của mỗi cặp dương sẽ bị thu nhỏ rất mạnh. Mục 4.4 giải thích lý do này.
- Giữ nguyên τ, r và hệ số theo modality của RAD.
- Lọc trùng chính xác (thay vì so khớp ngữ nghĩa) theo đúng Mục 4.4 để tránh cùng một văn bản vừa là positive vừa là negative.
- Detach prototype ở modality ảnh vì RAD cũng không truyền gradient qua prototype ảnh.

---

## 6. ClipLoss: bỏ khỏi cấu hình chính

**RAD gốc làm gì.** `ClipLoss` giữa ảnh pooled và caption pooled, hệ số `--loss_ratio 0.1`. Lỗi đã biết: `logit_scale` được tạo mới mỗi lần forward nên không bao giờ được học.

**Mô hình đề xuất đổi gì.**

- Cấu hình chính **không có ClipLoss**, đúng công thức 4.7 (chỉ có L_prop, L_PECL, L_label) **[QĐ-10]**.
- Vẫn giữ tham số `--clip_ratio` (mặc định 0). Nếu cần một lần chạy kiểm tra với 0.1, gọi đúng lớp `ClipLoss` của RAD.

**Tại sao.**

- Chương 4 là nguồn chuẩn, và phương pháp không có ClipLoss.
- Tham số `--clip_ratio` dùng để trả lời câu hỏi có thể gặp: "Mô hình dựa trên mệnh đề tốt hơn có phải chỉ vì bỏ ClipLoss không?". Chỉ cần một lần chạy đối chứng, không cần đưa ClipLoss vào phương pháp.

---

## 7. Loss tổng (Mục 4.7)

**RAD:** L = CE_label + CE_guideline + 0.1·Clip + 0.1·GECL_text + 0.001·GECL_img

**Mô hình đề xuất:** L = L_prop + β·L_PECL + λ·L_label

- β·L_PECL = 0.1·PECL_text + 0.001·PECL_img. Tức là α_text = 0.1 và α_img = 0.001 như RAD, còn β = 1.
- **λ = 1** **[QĐ-11]**.

Thêm tham số dòng lệnh:

- `--lambda_label` (mặc định 1.0; 0 tương đương `--no_label_branch`);
- `--no_pecl`;
- `--agg {mean,weighted}` (mặc định `mean`);
- `--tau_att` (mặc định 0.1);
- `--clip_ratio` (mặc định 0);
- `--query_level {proposition,disease}` (mục 10).

**Tại sao.** Đúng công thức 4.7, với các hệ số lấy theo RAD. Các tham số này đủ cho mọi ablation của Chương 5 mà không phải sửa code.

---

## 8. Validation và chọn epoch

**RAD gốc làm gì.** Mỗi epoch tính metrics cho nhánh nhãn và nhánh guideline. Điểm chọn epoch = max(macro-F1 nhãn, macro-F1 guideline) trên val. Lưu `predictions/val_epoch_XXX.npz` và `best.pt`.

**Mô hình đề xuất đổi gì.**

- Điểm chọn epoch = **macro-F1 của đường mệnh đề (ŷ) trên val**, không lấy max với nhánh nhãn.
- File npz lưu `gt`, `pred_proposition` (ŷ) và `pred_label` (nhánh bổ trợ, để tham khảo; bỏ hẳn khi không có nhánh, xem mục 4).
- `metrics_history.csv` ghi đủ metrics cho cả hai, đặt tên rõ ràng: `proposition_*` và `label_*`.
- Checkpoint và resume giống RAD. Checkpoint lưu thêm trạng thái RNG: Python, NumPy, torch CPU, CUDA, và một `torch.Generator` riêng dùng cho việc lấy mẫu negative của PECL **[QĐ-14]**.
  - Mục tiêu là chạy tiếp sau resume giống chạy liên tục ở mức tốt nhất có thể.
  - Không cam kết giống từng bit trên GPU, vì cuDNN có phép tính không tất định. Ghi rõ điều này trong README.

**Tại sao.** Mục 4.7: đầu ra chính chỉ đi qua mệnh đề. Chọn epoch theo đúng đầu ra sẽ báo cáo thì giao thức mới nhất quán.

---

## 9. Đánh giá test và xuất bảng kiểm (Mục 4.8)

**RAD gốc làm gì.** `evaluate_skin_test`:

- nạp `best.pt`;
- học trọng số trộn trên val của best epoch;
- áp dụng lên test;
- ghi `final_metrics.json` gồm `label`, `guideline`, `fused`.

**Mô hình đề xuất đổi gì.**

1. Nạp `best.pt` và chạy test **một lần**. **Không** có bước trộn.
2. `final_metrics.json` gồm:
   - `proposition`: kết quả chính;
   - `label_aux`: tham khảo (bỏ hẳn khi không có nhánh nhãn);
   - `best_epoch`, `best_validation_score`.

   Mỗi phần dùng đúng hàm `_skin_final_metrics` của RAD (top-1/2/3, MRR@k, macro-F1, balanced accuracy, AUC, mAP, metrics cũ của RAD).
3. **Bảng kiểm** (chỉ khi `--query_level proposition`), file `checklist/test_checklist.npz`:
   - `s` (N, J) và `s_tilde` (N, J), float16;
   - `z` (N, C), `y_hat` (N, C), `gt` (N, C);
   - các thông tin attention cho **mọi** cặp (ca bệnh, mệnh đề) **[QĐ-12]**:
     - `attn_top_idx` và `attn_top_w`, kích thước (N, J, 10): 10 vị trí có attention cao nhất;
     - `attn_entropy` (N, J): entropy của toàn bộ phân bố;
   - `attn_full` (K, J, L), float16: **toàn bộ** A cho K = 20 ca test chọn cố định bằng seed;
   - quy ước vị trí: với SkinCAP, 0–255 là patch ảnh trên lưới 16×16; 256 trở đi là token caption (lưu kèm `n_img_tokens` và kích thước lưới, không ngầm định);
   - `caption_input_ids` (N, L_caption), để ánh xạ vị trí token về chữ;
   - `image_paths`;
   - `agg_weights` (J): w_cj, chỉ khi `--agg weighted`.
4. File kèm `checklist/proposition_index.json` cho từng mệnh đề, theo đúng thứ tự cột của `s`: `proposition_id`, `disease_id`, `category`, `canonicalDescription`, `polarity`, `sourceExcerpt`, **`sourceReference`**.
5. **Khi `--query_level disease`:** không có bảng kiểm mức mệnh đề. Chỉ xuất `z`, `y_hat`, `gt` vào `checklist/test_disease_level.npz`.
6. Các file `.npz` và `.json` ở đây là **dữ liệu** để dựng bảng kiểm. Bảng kiểm dạng đọc được cho người dùng do script `analysis/checklist_report.py` tạo ra (mục 15.3).

**Tại sao.**

- Bảng kiểm lấy trực tiếp các đại lượng trên đường dự đoán. Mục 4.8 yêu cầu truy nguyên cả `sourceExcerpt` lẫn `sourceReference`.
- Không lưu toàn bộ A cho mọi ca vì quá lớn: 546 × 1436 × 768 ≈ 600 triệu số. Top-10 và entropy có cho mọi ca, toàn bộ A có cho 20 ca để làm case study.
- Các phép can thiệp ở Chương 5 sẽ chạy lại mô hình từ `best.pt` trong một script riêng, nên không cần A đã lưu.

---

## 10. Cấu hình đối chứng ở mức bệnh

**Mục đích.** Tách hai yếu tố: "đổi đơn vị tri thức sang mệnh đề" và "đổi nguồn tri thức sang KB". Câu hỏi là kết quả tốt lên nhờ mệnh đề, hay chỉ nhờ dùng văn bản KB (Mục 4.2).

**Cách làm (`--query_level disease`).**

- Mỗi bệnh có **1 query** = nối các t_j^align của bệnh đó bằng "; ". Polarity đã nằm trong văn bản qua "absence of …".
- Mã hóa với `max_length = 512`.
- Query đi qua **cùng** cross-attention và scorer, nhưng z_ic = s_ic trực tiếp: không có bước polarity, không tổng hợp.
- Hiệu chỉnh, loss, PECL và nhánh nhãn giữ nguyên.
- **Lưu ý cách diễn giải:** cấu hình này **vẫn dùng PECL ở mức mệnh đề**. Đối chứng chỉ đổi độ phân giải của **đường dự đoán**, không đổi toàn bộ mô hình sang mức bệnh. Ghi rõ điều này trong `run_config.json` và README.
- **Đo và ghi** ra `disease_level_lengths.json` (theo Mục 4.2), dùng tokenizer ClinicalBERT:
  - độ dài token của từng bệnh;
  - số bệnh vượt 512;
  - tỉ lệ token bị cắt của từng bệnh.

  **Không** tự áp dụng chiến lược cắt hay rút gọn nào khác. Nếu có bệnh bị cắt, báo lại người dùng để quyết định.

**Tại sao.** Cấu hình này có cùng tri thức và cùng module, chỉ khác độ phân giải tri thức trên đường dự đoán (một vector cho mỗi bệnh thay vì một vector cho mỗi dấu hiệu). Đây là phép so sánh trực tiếp cho câu hỏi về đơn vị mệnh đề.

---

## 11. Kiểm tra bắt buộc trước khi bàn giao

Các con số dưới đây là **kiểm tra tích hợp cho SkinCAP**, không phải hằng số trong model (mục 0, ràng buộc 4).

1. **KB:** 47 bệnh, 1436 mệnh đề, 31 polarity −1; thứ tự bệnh trùng header CSV. In danh sách 31 văn bản "absence of …".
2. **Độ dài token:** in ra độ dài lớn nhất (theo tokenizer ClinicalBERT) của văn bản query và văn bản căn chỉnh, cùng `max_length` đã chọn; assert không mệnh đề nào bị cắt.
3. **Shape và attention:**
   - batch giả B = 2 cho o (2, 47), s (2, 1436), A (2, 1436, 768);
   - mỗi hàng của A cộng lại bằng 1 và bằng 0 tại vị trí PAD;
   - logit attention nằm trong [−1/τ_att, 1/τ_att] (vì dùng cosine).
4. **Polarity:** với mệnh đề −1, s_tilde = 1 − s.
5. **Tổng hợp:** z_c tính bằng vòng lặp thường phải bằng z_c từ scatter-mean (sai số nhỏ hơn 1e-6).
6. **Đơn điệu, hai chiều:**
   - tăng s của một mệnh đề +1 thuộc bệnh c thì ŷ_c tăng;
   - **giảm** s của một mệnh đề −1 thuộc bệnh c thì ŷ_c tăng;
   - γ_c > 0 ở mọi thời điểm.
7. **PECL:**
   - không văn bản nào nằm đồng thời trong P_i và Q_i; |Q_i| ≤ 5|P_i|;
   - mẫu có |P_i| = 0 không đóng góp và **mẫu số vẫn là N**;
   - so loss tính bằng vòng lặp thường với công thức 4.4 trên một ví dụ nhỏ (sai số nhỏ hơn 1e-6).
8. **Resume:** chạy 2 epoch liên tục, rồi chạy 1 epoch + resume 1 epoch; so loss và metrics (báo chênh lệch, không bắt buộc bằng 0).
9. **Baseline nguyên vẹn:** `git diff` không có thay đổi trong các file RAD liệt kê ở mục 0.
10. **Chạy thử:**
    - 1 epoch trên khoảng 64 ảnh train và 32 ảnh val/test, ra được `final_metrics.json`, `run_config.json` và `checklist/test_checklist.npz`;
    - chạy cả `--query_level disease` để ra `disease_level_lengths.json`;
    - chạy cả `--no_label_branch` và xác nhận không còn key `pred_label` / `label_aux`.
11. **Báo bộ nhớ GPU** đo được khi chạy thử (xem QĐ-3).

---

## 12. Các quyết định đề xuất (chưa có trong Chương 4)

Đây là các lựa chọn thực nghiệm. Tất cả được ghi vào `run_config.json` và trình bày lại trong phần thiết lập thực nghiệm của Chương 5.

| Mã | Quyết định | Lý do ngắn | Nếu đổi |
|---|---|---|---|
| QĐ-1 | H = `fusion_features` của RAD, không thêm phép chiếu | Dữ liệu đầu vào giống RAD; ảnh đã có phép chiếu `res_l2`, caption đã ở 768 | Thêm Linear theo modality (công thức 4.3) |
| QĐ-2 | `max_length` mệnh đề = độ dài đo được, không cắt | Không mất nội dung mệnh đề | — |
| QĐ-3 | Mã hóa mệnh đề có gradient ở mỗi bước. Nếu hết bộ nhớ: bật gradient checkpointing cho ClinicalBERT (cùng phép tính, chỉ chậm hơn). Nếu vẫn hết bộ nhớ: **dừng và báo người dùng** | RAD cũng huấn luyện encoder trên guideline. **Không tự chuyển sang `no_grad`** vì như vậy là đổi mô hình | Người dùng quyết định trước khi chạy thí nghiệm |
| QĐ-4 | τ_att = 0.1, cố định | Cosine/0.1 cho logit trong [−10, 10], đủ sắc. Cố định để ít tham số | Cho τ_att học được |
| QĐ-5 | Mask token PAD của caption | Tránh query đọc vào PAD. RAD không mask; ghi nhận đây là khác biệt nhỏ | Bỏ mask |
| QĐ-6 | Scorer MLP [c; q] → d → GELU → 1 | Mạng nhỏ dùng chung (4.5) | — |
| QĐ-7 | γ_c = softplus, khởi tạo ≈ 10; b_c = −5 | z nằm trong [0, 1]; tránh logit quá dẹt lúc đầu | γ khởi tạo 1, b 0 |
| QĐ-8 | Nhánh nhãn viết lại bằng module cross-attention (cosine, một head), không dùng `TQN_Model_fusion` | Đúng 4.6, tránh lỗi batch-position | Dùng nguyên decoder nhãn của RAD |
| QĐ-9 | Prototype ảnh trong PECL bị detach | Giống RAD (prototype ảnh tính trong `no_grad`) | Bỏ detach |
| QĐ-10 | Cấu hình chính không có ClipLoss; `--clip_ratio 0.1` chỉ để chạy đối chứng | Đúng công thức 4.7 | Giữ ClipLoss thì phải thêm vào công thức 4.7 |
| QĐ-11 | λ = 1, β = 1 (α_text = 0.1, α_img = 0.001) | Theo trọng số của RAD | Ablation với λ = 0 hoặc `--no_pecl` |
| QĐ-12 | Mọi ca: top-10 và entropy; 20 ca: toàn bộ A | Đủ cho phân tích, dung lượng nhỏ; can thiệp chạy lại từ `best.pt` | Lưu toàn bộ A cho nhiều ca hơn |
| QĐ-13 | Mẫu số của PECL là N (cả batch) | Đúng 4.4. SkinCAP mỗi ca có đúng 1 nhãn dương nên không khác biệt thực tế | — |
| QĐ-14 | Lưu trạng thái RNG trong checkpoint; dùng generator riêng cho lấy mẫu negative | Resume gần giống chạy liên tục | — |
| QĐ-15 | 3 seed (42, 43, 44) cho RAD, mô hình đề xuất và cấu hình mức bệnh; 1 seed (42) cho các ablation | Đủ để báo trung bình ± độ lệch cho các so sánh chính, chi phí Colab vừa phải | Chạy 3 seed cho cả ablation |
| QĐ-16 | So với RAD bằng đầu ra chính thức của RAD (`fused`); báo thêm nhánh `guideline` | `fused` là cách RAD báo kết quả | Chỉ dùng `guideline` |
| QĐ-17 | Bootstrap ghép cặp trên tập test, 1000 lần lặp, khoảng tin cậy 95% | Cho biết chênh lệch có ổn định hay không mà không cần chạy thêm | — |
| QĐ-18 | Can thiệp mức attention: che top-10 vị trí, so với che 10 vị trí ngẫu nhiên (lặp 5 lần) | Rẻ (không mã hóa lại), kiểm tra trực tiếp dấu vết bằng chứng | Thêm can thiệp ở mức ảnh gốc cho 20 ca case study |

---

## 13. Bàn giao

Commit lên nhánh `RAD-colab-enhanced` và ghi trong mô tả commit:

1. danh sách file mới;
2. kết quả 11 kiểm tra ở mục 11;
3. nội dung `disease_level_lengths.json`, tóm tắt;
4. các chỗ đã làm khác tài liệu này (nếu có) và lý do;
5. "Câu hỏi mở", nếu có.

---

## 14. Trả lời phản hồi của agent (bản 1 → bản 2)

| Phản hồi | Cách xử lý |
|---|---|
| Cross-attention: dot-product hay cosine | Theo Chương 4: cosine/τ_att, một head (mục 3, QĐ-4) |
| PECL: target 1/\|P\| hay 1/0 | Theo Chương 4: target 1/0, trung bình riêng trên P và Q (mục 5) |
| ClipLoss | Bỏ khỏi cấu hình chính; `--clip_ratio` chỉ để đối chứng (mục 6, QĐ-10) |
| Thiếu `sourceReference` | Đã thêm (mục 9) |
| Cấu hình mức bệnh có thể bị cắt ở 512 token | Đo và báo, không tự xử lý (mục 10) |
| Bảng kiểm khi `query_level=disease` | Chỉ áp dụng cho cấu hình mệnh đề; cấu hình bệnh xuất file riêng (mục 9) |
| Top-10 attention chưa đủ | Thêm entropy, toàn bộ A cho 20 ca; can thiệp chạy lại từ checkpoint (mục 9, QĐ-12) |
| Mã hóa có gradient và nguy cơ hết bộ nhớ | Chốt chính sách: gradient checkpointing, nếu vẫn hết thì dừng và báo; không tự chuyển `no_grad` (QĐ-3) |
| `max_length = 32` | Đo bằng tokenizer thật, không cắt (QĐ-2) |
| Mẫu số PECL | N, đúng 4.4 (QĐ-13) |
| RNG khi resume | Lưu RNG và generator riêng (QĐ-14) |
| Test polarity âm | Đã thêm (kiểm tra 6) |
| Xuất w_cj khi weighted | Đã thêm (mục 3 và 9) |

---

## 14b. Trả lời phản hồi đợt 2 của agent (bản 2 → bản 3)

| Phản hồi | Cách xử lý |
|---|---|
| τ_att, detach prototype ảnh, λ, hệ số PECL, khởi tạo calibration là quyết định thực nghiệm | Đồng ý. Mọi siêu tham số ghi vào `run_config.json`; script phân tích xuất bảng siêu tham số cho Chương 5 (mục 12, 15) |
| Vai trò của nhánh nhãn | Đã viết lại chính xác ở mục 4: chỉ vào hàm mục tiêu khi train; được chạy ở val/test để phân tích; không vào dự đoán chính, chọn epoch hay trộn |
| `--query_level disease` vẫn dùng PECL mức mệnh đề | Đồng ý. Đã ghi rõ ở mục 10: đối chứng chỉ đổi độ phân giải của đường dự đoán |
| Kiểm tra `"absence of " + t_j` | Đã kiểm tra tay cả 31 câu, đều hợp nghĩa. Agent vẫn in danh sách trong kiểm tra 1 (mục 5, 11) |
| Không hard-code 47/1436/31/256/512 | Đồng ý. Thêm ràng buộc 4 ở mục 0; mục 3 và 9 dùng ký hiệu J, C |
| `.npz` + JSON chỉ là dữ liệu; cần script dựng bảng kiểm | Đồng ý. Thêm `analysis/checklist_report.py` (mục 9 điểm 6, mục 15.3) |
| Schema khi `--no_label_branch` | Bỏ hẳn các key, không ghi `null` (mục 4, kiểm tra 10) |

---

## 15. Đầu ra phục vụ viết Chương 5

Mục tiêu: khi chạy xong, có đủ số liệu, bảng và hình để viết Chương 5 cho cả ba đóng góp ở Chương 1:

1. biểu diễn tri thức chẩn đoán ở mức mệnh đề;
2. tích hợp tri thức mức mệnh đề vào mô hình đa phương thức;
3. thông tin hỗ trợ theo từng tiêu chí (bảng kiểm).

Làm theo **hai đợt**:

- **Đợt 1** (cùng lần code này): code huấn luyện (mục 1–11) và **thông tin phải lưu ở mục 15.2**. Thiếu phần lưu này thì sau này phải chạy lại.
- **Đợt 2** (sau khi có kết quả chạy đầu tiên): các script phân tích ở mục 15.3. Các script này đọc file đã lưu; chỉ `intervention.py` cần GPU.

### 15.1. Danh sách lần chạy [QĐ-15]

| Mã | Mô hình | Cấu hình | Seed | Dùng để |
|---|---|---|---|---|
| R0 | RAD (`main_rad.py`, không sửa) | gốc | 42, 43, 44 | Baseline |
| P1 | Mô hình đề xuất | đầy đủ | 42, 43, 44 | Kết quả chính |
| P2 | Mô hình đề xuất | `--query_level disease` | 42, 43, 44 | Mệnh đề so với mức bệnh, cùng tri thức |
| A1 | Mô hình đề xuất | `--no_pecl` | 42 | Đóng góp của PECL |
| A2 | Mô hình đề xuất | `--no_label_branch` | 42 | Đóng góp của nhánh nhãn |
| A3 | Mô hình đề xuất | `--agg weighted` | 42 | Giả định trọng số bằng nhau |
| A4 | Mô hình đề xuất | `--clip_ratio 0.1` | 42 | Đối chứng ClipLoss |

Cách tổ chức:

- Seed chỉ đổi phần ngẫu nhiên khi huấn luyện; tập train/val/test giữ cố định.
- Lần chạy RAD đã có (47 lớp, cùng giao thức val/test) được tính là R0 seed 42 nếu đúng commit hiện tại.
- Thư mục mỗi lần chạy đặt tên theo mẫu `<Mã>_s<seed>`, ví dụ `P1_s42`.

### 15.2. Mỗi lần chạy phải lưu (đợt 1, cho cả RAD và mô hình đề xuất)

1. Các file đã có: `final_metrics.json`, `metrics_history.csv`, `predictions/val_epoch_XXX.npz`, `predictions/test_best.npz`, `checkpoints/best.pt`.
2. **`run_config.json`**, gồm:
   - toàn bộ tham số dòng lệnh và config yaml;
   - mã lần chạy và seed;
   - git commit hash;
   - sha256 của KB và của 3 file CSV; số mẫu train/val/test; số bệnh và số mệnh đề;
   - tên GPU; phiên bản torch và transformers;
   - **số tham số huấn luyện được của từng module** (encoder ảnh, encoder văn bản, cross-attention, scorer, calibration, nhánh nhãn);
   - **thời gian trung bình mỗi epoch**, **bộ nhớ GPU đỉnh**, **thời gian suy luận trung bình mỗi ca** trên test;
   - `best_epoch`.

   Với RAD, các thông tin này được ghi bằng một wrapper hoặc script riêng; **không sửa code RAD**.
3. `test_best.npz` của mô hình đề xuất lưu thêm `image_paths` theo đúng thứ tự, để ghép cặp với RAD. Thứ tự test của hai mô hình phải trùng nhau (cùng `SequentialSampler` và cùng CSV); script phân tích phải assert điều này.
4. P1 lưu bảng kiểm như mục 9 (A3 cũng lưu, để có `agg_weights`); P2 lưu file mức bệnh như mục 9 điểm 5. Các ablation còn lại chỉ cần metrics.

### 15.3. Script phân tích (đợt 2, thư mục `analysis/`)

Mỗi script xuất **CSV** (số liệu) và **LaTeX** (bảng `booktabs`, dán thẳng vào luận văn). Hình xuất PNG.

| Script | Kết quả cho Chương 5 | Cần GPU |
|---|---|---|
| `make_tables.py` | **Bảng chính:** R0 / P1 / P2, trung bình ± độ lệch qua 3 seed, đủ hai tầng metrics. Tầng 1 là metrics gốc của RAD: F1, P, R, AUC, mAP, Acc. Tầng 2 là top-1/2/3, MRR@2/3, macro-MRR, macro-F1, balanced accuracy. **Bảng ablation** A1–A4 so với P1 (seed 42). **Bảng siêu tham số** lấy từ `run_config.json` | Không |
| `bootstrap.py` | Chênh lệch P1 − R0 và P1 − P2 trên test, ghép cặp theo seed, 1000 lần lặp, khoảng tin cậy 95% cho Δmacro-F1, Δtop-1, Δtop-3, ΔMRR@3 **[QĐ-16, QĐ-17]** | Không |
| `per_class.py` | Bảng F1 và top-3 theo từng bệnh cho R0 và P1. Tương quan Spearman giữa ΔF1 theo bệnh với số mệnh đề M_c và với số mẫu train. 10 cặp bệnh bị nhầm nhiều nhất | Không |
| `checklist_report.py` | Bảng kiểm dễ đọc (HTML), ghép từ `test_checklist.npz` và `proposition_index.json`, cho 10 ca chọn cố định bằng seed: 5 ca đúng top-1, 3 ca đúng trong top-3 nhưng sai top-1, 2 ca sai. Mỗi ca có: ảnh kèm heatmap attention của mệnh đề đóng góp cao nhất; caption tô đậm các token được chú ý; top-3 bệnh với ŷ và z; với mỗi bệnh là bảng mệnh đề gồm nhóm, polarity, s, s̃, `sourceExcerpt`, `sourceReference` | Không |
| `intervention.py` | **Faithfulness, Mục 4.8.** Với mỗi ca test và mỗi mệnh đề của bệnh được dự đoán: che top-10 vị trí có attention cao nhất của mệnh đề đó **ngay trong attention** (H giữ nguyên, chỉ tính lại query đó). So với che 10 vị trí ngẫu nhiên, lặp 5 lần. Báo Δs và Δŷ trung bình, và tỉ lệ ca mà che top-10 làm s giảm nhiều hơn che ngẫu nhiên **[QĐ-18]**. Thêm: bỏ 5 mệnh đề có s̃ cao nhất khỏi phép tổng hợp so với bỏ 5 mệnh đề ngẫu nhiên, báo Δŷ | Có |
| `cost_table.py` | Bảng chi phí R0 và P1: số tham số, thời gian mỗi epoch, bộ nhớ đỉnh, thời gian suy luận mỗi ca | Không |

Thống kê KB (số mệnh đề theo bệnh và theo nhóm, polarity −1, số câu nguồn bị loại) đã có sẵn trong `kb_builder/outputs/final/kb_stats.md`, không cần script mới.

### 15.4. Các kết quả này phục vụ phần nào của Chương 5

| Nội dung Chương 5 | Lấy từ |
|---|---|
| Dữ liệu và tri thức | Số mẫu train/val/test (`run_config.json`); `kb_stats.md` |
| Thiết lập thực nghiệm | Bảng siêu tham số và các QĐ ở mục 12, phần cứng, số seed (`make_tables.py`, `run_config.json`) |
| Kết quả chính so với RAD | Bảng chính và khoảng tin cậy bootstrap |
| Mệnh đề so với mức bệnh | P1 so với P2 (bảng chính, bootstrap); `disease_level_lengths.json` (yếu tố cắt token) |
| Ablation | Bảng ablation A1–A4 |
| Phân tích theo bệnh | `per_class.py` |
| Bảng kiểm và tính diễn giải | `checklist_report.py` (case study), `intervention.py` |
| Ý nghĩa thực tiễn | Top-3 và MRR@3 (hỗ trợ ra quyết định); bảng kiểm truy nguyên; bảng chi phí |
| Giới hạn | PECL là giám sát yếu; trung bình đều; τ_att và các khởi tạo là lựa chọn thực nghiệm |