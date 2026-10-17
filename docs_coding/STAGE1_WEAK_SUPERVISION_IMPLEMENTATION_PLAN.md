# KẾ HOẠCH TRIỂN KHAI STAGE 1 — WEAKLY SUPERVISED PROPOSITION EVIDENCE LEARNING

**Repository:** `trisle0805/RAD-colab`  
**Branch:** `WS`  
**Phạm vi:** viết lại phần proposition theo kiến trúc weak supervision hiện tại; trước mắt chỉ triển khai và khảo sát **Stage 1**.  
**Mục tiêu:** đưa file này cho coding agent làm theo từng commit nhỏ; sau mỗi commit sẽ review trên GitHub trước khi đi tiếp.

---

## 1. Mục tiêu Stage 1

Stage 1 không học chẩn đoán bệnh. Nó chỉ học:

> Với bệnh nhân `i` và proposition `j`, ảnh hoặc caption có cung cấp bằng chứng cho **sự hiện diện của concept** mà proposition mô tả hay không?

Kiến trúc:

```text
Image
  ↓
Frozen ResNet50
  ↓
image patch features V_i
  ↑
proposition query q_j
  ↓
image cross-attention
  ↓
h_img(i,j)
  ↓
shared image head
  ↓
s_img(i,j) ∈ [0,1]

Masked caption
  ↓
Frozen Bio_ClinicalBERT
  ↓
text token features T_i
  ↑
proposition query q_j
  ↓
text cross-attention
  ↓
h_txt(i,j)
  ↓
shared text head
  ↓
s_txt(i,j) ∈ [0,1]
```

Stage 1 **không được có**:

- disease BCE / cross-entropy;
- disease aggregation;
- disease calibration;
- polarity transform;
- PECL cũ;
- auxiliary disease-label branch;
- learned proposition-ID embedding;
- learned disease-ID embedding;
- disease name / disease ID / polarity trong proposition query.

Teacher LLM và ConceptCLIP không nằm trong Stage 1. Chúng đã chạy offline và chỉ còn tồn tại dưới dạng weak-label artifacts.

---

## 2. Source of truth

Một số file cũ trong repo / tài liệu cũ mô tả teacher disease-blind, NLI, PECL hoặc P1. **Không được phục hồi các thiết kế đó.**

Weak-label generation đã đóng và không phải nhiệm vụ của coding agent.

Artifacts v3 hiện tại là input chính thức:

```text
WeakSup/weak_labels/v3/
├── labels_raw_v3.npz
├── active_propositions_v3.csv
├── active_masks_v3.npz
├── metadata_final_v3.json
└── checkpoints/
    └── llm_text_labels_raw_v3.npz
```

Sanity expectation, **không hard-code vào model**:

```text
Full KB propositions       = 403
Active scoring             ≈ 307
Inactive / N/A             ≈ 96
Text-trainable             ≈ 240
Image-trainable            ≈ 180
Both text + image          ≈ 113
Text-only                  ≈ 127
Image-only                 ≈ 67
Diseases                   = 47
Patients                   = 2526
min_labels                 = 10
ConceptCLIP positive thr   = 0.80
ConceptCLIP negative thr   = 0.20
```

Nếu artifact thật không khớp các con số trên, code phải fail rõ ràng hoặc báo mismatch; không tự sửa dữ liệu.

---

## 3. Phạm vi refactor repository

Branch `P1` đã lưu thiết kế cũ riêng. Vì vậy trên branch `WS`:

> Có thể viết lại toàn bộ `proposition/` theo final weak-supervision design.

Không cần backward compatibility với P1 bên trong package `proposition`.

Tuy nhiên:

- không sửa `main_rad.py`;
- không làm thay đổi baseline RAD;
- tái sử dụng `models.clip_tqn.ModelRes` và `Text_Encoder_Bert` nếu phù hợp;
- viết lại `main_proposition.py` thành Stage-1 entrypoint;
- `configs/ws.yaml` phải chuyển sang artifact v3.

Các module P1 không còn dùng nên xóa hoặc thay thế:

```text
proposition/losses/pecl.py
auxiliary disease-label logic
disease aggregation/calibration trong Stage 1
old P1 disease metrics
old v1 weak-label runtime assumptions
```

Git history đã giữ code cũ, nên không cần giữ file chết chỉ để tham khảo.

---

## 4. Cấu trúc package đề nghị

```text
proposition/
├── README.md
├── data/
│   ├── knowledge_base.py
│   ├── weak_supervision.py
│   ├── stage1_dataset.py
│   └── caption_masking.py
│
├── models/
│   ├── cross_attention.py
│   └── stage1_evidence_model.py
│
├── losses/
│   └── masked_bce.py
│
├── engine/
│   ├── trainer_stage1.py
│   ├── metrics_stage1.py
│   ├── artifacts.py
│   └── diagnostics.py
│
└── tests/
    ├── test_weak_supervision.py
    ├── test_stage1_dataset.py
    ├── test_stage1_model.py
    ├── test_masked_bce.py
    ├── test_stage1_metrics.py
    └── test_stage1_smoke.py
```

Tên file có thể khác, nhưng trách nhiệm module phải rõ như trên.

---

## 5. Data contracts phải làm trước

### 5.1 `labels_raw_v3.npz`

Đọc tối thiểu:

```text
case_id
split
proposition_id
label_text
label_img
p_img
```

Semantics:

```text
 1 = PRESENT
 0 = ABSENT
-1 = UNKNOWN
```

Ràng buộc cứng:

```text
UNKNOWN != ABSENT
```

UNKNOWN không bao giờ được đổi thành 0.

### 5.2 `active_masks_v3.npz`

Đọc:

```text
proposition_id
text_trainable
image_trainable
active_scoring
min_labels
```

Mọi alignment phải dựa vào `proposition_id`, **không giả định vị trí cột trùng nhau**.

### 5.3 `active_propositions_v3.csv`

Dùng để audit / metadata:

```text
proposition_id
disease_id
category
canonicalDescription
polarity
text_trainable
image_trainable
active_scoring
...
```

### 5.4 `metadata_final_v3.json`

Dùng làm manifest/audit. Nếu có SHA256, loader nên có option kiểm tra hash.

### 5.5 Knowledge base

Full KB vẫn giữ đủ 403 propositions.

Stage 1 chỉ query propositions có:

```text
active_scoring == 1
```

Inactive proposition:

```text
N/A / unsupported
```

không phải score 0.

---

## 6. Dataset Stage 1

Dataset mới phải trả ít nhất:

```python
{
    "case_id": ...,
    "image": ...,
    "caption": ...,
    "split": ...,
    "disease_label": ...  # chỉ diagnostic/evaluation, không vào loss/model
}
```

Weak labels lookup bằng `case_id`.

Không map bằng row index nếu chưa xác nhận alignment.

### 6.1 Caption cho student

Teacher v3 đã dùng raw caption và known disease khi tạo weak label. Việc đó đã đóng.

**Student Stage 1 không được đọc trực tiếp disease name.**

Caption student:

```text
raw caption
→ mask disease name / alias / abbreviation tiết lộ diagnosis
→ tokenize
```

Không cần loại toàn bộ “general disease knowledge”.

Masking phải disease-agnostic:

- dùng danh sách alias toàn cục của 47 diseases;
- không dùng ground-truth disease của patient để quyết định mask;
- cùng preprocessing cho train/val/test;
- sau này baseline RAD dùng cùng masked caption để so sánh công bằng.

Phải có unit test chứng minh tên bệnh bị mask nhưng phần còn lại không bị xóa.

---

## 7. Encoder policy Stage 1

Backbones:

```text
Image: ResNet50
Text: Bio_ClinicalBERT
```

Trong Stage 1:

```text
ResNet50          FROZEN
Bio_ClinicalBERT  FROZEN
```

Chỉ train:

```text
image cross-attention
text cross-attention
image scalar head
text scalar head
```

Rất quan trọng:

- `requires_grad=False` cho backbone;
- backbone ở `eval()` ngay cả khi Stage-1 wrapper ở `train()`;
- ResNet BatchNorm không cập nhật running stats;
- BERT dropout không ngẫu nhiên trong frozen encoder;
- encoder forward có thể chạy trong `torch.no_grad()`.

Phải có test xác nhận backbone weights và BN stats không đổi sau optimizer step.

---

## 8. Proposition query

Mỗi proposition query chỉ được tạo từ:

```text
canonicalDescription
```

Không đưa vào query:

- proposition ID;
- disease ID;
- disease name;
- polarity;
- category token đặc biệt.

Query encoder:

```text
Bio_ClinicalBERT frozen
```

Dùng pooled representation hiện có từ `Text_Encoder_Bert.encode_text` để tương thích codebase.

Queries là static knowledge:

> Encode một lần khi khởi tạo run, cache trong RAM/GPU hoặc artifact cache.

Không encode lại toàn bộ 307 proposition texts ở mỗi mini-batch.

Expected shape:

```text
Q = [J_active, 768]
```

---

## 9. Cross-attention mới

Không dùng lại `CosineCrossAttention` single-head P1 làm main implementation.

Final Stage 1:

```text
1 cross-attention layer
8 attention heads
model dimension = 768
```

Có **hai module riêng**:

```text
image_cross_attention
text_cross_attention
```

Không share attention parameters giữa image và text.

Implementation mặc định:

```python
nn.MultiheadAttention(
    embed_dim=768,
    num_heads=8,
    batch_first=True,
)
```

Block tối giản:

```text
query
  ↓
MHA(query, memory, memory)
  ↓
residual with query
  ↓
LayerNorm
  ↓
h_ij
```

Không thêm Transformer stack sâu / FFN lớn nếu chưa có lý do thực nghiệm.

Text attention phải nhận padding mask.

Phải support `return_attention=True`.

---

## 10. Hai proposition heads

Hai head riêng theo modality:

```text
image_head: Linear(768, 1)
text_head : Linear(768, 1)
```

Head được **share giữa tất cả propositions trong cùng modality**.

Không có head riêng theo proposition ID hoặc disease.

Outputs:

```text
logits_img: [B, J]
logits_txt: [B, J]

s_img = sigmoid(logits_img)
s_txt = sigmoid(logits_txt)
```

Semantics:

> `s_*[i,j]` = mức evidence cho **sự hiện diện của concept j**.

Không phải disease probability.

---

## 11. Modality masks và fusion

Training mask:

```text
text branch:
    text_trainable[j] == 1
    AND label_text[i,j] ∈ {0,1}

image branch:
    image_trainable[j] == 1
    AND label_img[i,j] ∈ {0,1}
```

Không train branch từ UNKNOWN.

### 11.1 Final support score

Không dùng score của modality chưa được train cho proposition đó.

```text
both text + image trainable:
    s = (s_img + s_txt) / 2

text-only:
    s = s_txt

image-only:
    s = s_img

inactive:
    N/A
```

Với dữ liệu hiện tại, image branch chỉ áp cho morphology.

Location / symptom / history về thực tế đi text-only.

Fusion weight là fixed, không học.

Stage 1 chưa áp polarity.

---

## 12. Masked BCE loss

Tạo loss mới hoàn toàn, không disease BCE và không PECL.

Mỗi modality:

```text
mask = trainable_prop_mask
       AND weak_label != UNKNOWN
```

BCE tính trên logits tại đúng các ô mask.

Normalize từng branch theo số valid labels của branch trong batch:

```text
L_txt = masked BCE mean over valid text cells
L_img = masked BCE mean over valid image cells

L_stage1 = L_txt + L_img
```

Nếu một branch không có valid cell trong batch:

```text
branch loss = 0
```

Không NaN, không divide-by-zero.

---

## 13. `pos_weight` với active set hiện tại

Text weak labels có nhiều propositions positive-only.

Không được viết mù:

```python
pos_weight = N0 / N1
```

vì khi `N0 = 0`, weight = 0 và positive loss biến mất.

Tính `pos_weight` per `(modality, proposition)` trên **training split**.

Rule:

```text
if N0 > 0 and N1 > 0:
    pos_weight = min(N0 / N1, 10)

if N0 == 0 or N1 == 0:
    pos_weight = 1.0
    mark single_class = true
```

Không loại proposition chỉ vì single-class; đây là quyết định active-set hiện tại.

Lưu bảng audit:

```text
proposition_id
modality
N_present
N_absent
N_unknown
pos_weight
single_class
```

---

## 14. Optimizer Stage 1

Main config:

```text
Optimizer          AdamW
Learning rate      1e-4
Early stopping     patience 10
Trainable params   attention + heads only
```

Weight decay để config, default nhỏ.

Không optimizer group cho frozen encoders.

Không disease label trong loss.

Không PECL.

Không auxiliary branch.

---

## 15. Checkpoint selection Stage 1

Không chọn best model bằng disease macro-F1 như P1.

Validation metric chính:

> Macro ROC-AUC theo `(modality, proposition)` có cả 0 và 1 trên validation.

Cách tính:

1. Text: từng text-trainable proposition có cả hai lớp trên val → ROC-AUC.
2. Image: tương tự.
3. `selection_macro_auc` = mean không trọng số của tất cả valid `(modality, proposition)` AUC.

Báo thêm:

```text
text_macro_roc_auc
image_macro_roc_auc
selection_macro_auc

text_macro_pr_auc
image_macro_pr_auc

precision / recall / F1 @ 0.5
number of evaluable propositions
number of single-class propositions
masked validation loss
```

Best checkpoint:

```text
max selection_macro_auc
```

Tie-breaker nếu cần:

```text
lower validation masked loss
```

Phải ghi rõ bao nhiêu propositions thực sự tính được AUC.

---

## 16. Diagnostics bắt buộc

### 16.1 Score differentiation

Lưu:

```text
within-patient std across propositions
pairwise proposition score correlation
fraction of near-identical score pairs
```

Mục tiêu: phát hiện collapse kiểu P1.

### 16.2 Query swap

Validation/test:

```text
đổi permutation proposition queries
```

Nếu model dùng semantics query, metrics / score assignment phải thay đổi rõ.

Không retrain.

### 16.3 Query ablation

Support:

```text
zero query
mean/constant query
random query
```

So với normal query.

### 16.4 Weak-label shuffle

Cho phép một control run:

```text
shuffle weak labels trong training
```

Không cần chạy ở commit đầu nhưng code path nên thêm được dễ dàng.

### 16.5 Attention

Lưu attention cho subset deterministic.

Chỉ gọi:

```text
evidence trace / attention trace
```

không gọi causal explanation.

---

## 17. Pilot Stage 1 trước full training

Config phải cho phép:

```text
--pilot_diseases acne ...
```

Pilot disease filter nên:

- chọn propositions thuộc 3–5 diseases;
- vẫn có thể dùng toàn bộ patient cases để học các propositions đó vì weak labels có cho mọi patient-proposition pair;
- metrics report thêm within-disease subset.

Đề nghị pilot đầu:

```text
acne
actinic keratosis
allergic contact dermatitis
basal cell carcinoma
dermatofibroma
```

Pilot gate:

```text
1. loss giảm bình thường
2. không NaN / mask bug
3. frozen encoders thật sự không đổi
4. proposition AUC tốt hơn trivial/shuffle control
5. query swap/ablation làm metrics hoặc score mapping giảm
6. score giữa propositions không collapse gần-identical
7. text/image branch có behavior hợp lý
```

Nếu pilot fail, không viết Stage 2.

---

## 18. Output artifacts Stage 1

Một run tạo:

```text
<output_dir>/
├── checkpoints/
│   ├── latest.pt
│   └── best.pt
│
├── metrics/
│   ├── history.csv
│   ├── best_validation.json
│   ├── proposition_metrics.csv
│   └── modality_summary.json
│
├── predictions/
│   ├── val_best.npz
│   └── test_best.npz
│
├── diagnostics/
│   ├── score_differentiation.json
│   ├── query_swap.json
│   ├── query_ablation.json
│   └── attention_sample.npz
│
├── supervision/
│   └── label_distribution.csv
│
└── run_config.json
```

`val_best.npz` / `test_best.npz` nên có:

```text
case_id
proposition_id
score_text
score_image
score_fused
label_text
label_img
text_valid_mask
image_valid_mask
text_trainable
image_trainable
active_scoring
```

Không tạo disease probability ở Stage 1.

---

## 19. Checkpoint phải lưu gì

```text
image_cross_attention state
text_cross_attention state
image_head state
text_head state

optimizer
epoch
best metric
config
RNG state

proposition_id order
artifact hashes
git commit
```

Frozen ResNet/BERT có thể lưu state hoặc model identifier + version/hash.

Ưu tiên reproducible: lưu đủ state nếu dung lượng chấp nhận được.

---

## 20. `main_proposition.py`

Viết lại thành Stage 1 entrypoint.

Không còn:

```text
DiseaseAggregator
PositiveCalibration
PECLLoss
AuxiliaryLabelBranch
disease multiclass validation score
checklist disease prediction
```

CLI tối thiểu:

```text
--config
--output_dir
--kb_path
--labels_npz
--active_masks_npz
--active_csv
--metadata_json

--bert_model_name
--image_encoder_name

--seed
--device
--query_chunk_size

--pilot_diseases ...
--return_attention_samples ...

--resume
```

Stage 2 có entrypoint riêng sau khi Stage 1 pass.

---

## 21. `configs/ws.yaml`

Rewrite sang v3:

```yaml
weak_supervision:
  root: /content/drive/MyDrive/Cao học/WeakSup/weak_labels/v3
  labels: .../labels_raw_v3.npz
  active_masks: .../active_masks_v3.npz
  active_csv: .../active_propositions_v3.csv
  metadata: .../metadata_final_v3.json
  min_labels: 10

stage1:
  embed_dim: 768
  attention_heads: 8
  attention_layers: 1

  freeze_image_encoder: true
  freeze_text_encoder: true

  optimizer: adamw
  lr: 1.0e-4
  early_stopping_patience: 10

  pos_weight_cap: 10.0
  query_chunk_size: 64
```

Không hard-code 403/307 trong YAML.

---

## 22. Tests tối thiểu

### Artifact/alignment

- proposition IDs align đúng;
- case IDs align CSV;
- duplicate ID → fail;
- missing ID → fail;
- UNKNOWN giữ -1;
- inactive proposition không vào active query set.

### Dataset

- trả đúng case_id;
- masked caption không còn exact disease label;
- disease label không truyền vào model forward.

### Frozen encoders

- `requires_grad=False`;
- optimizer không chứa encoder params;
- weights không đổi sau train step;
- ResNet BN running stats không đổi;
- BERT ở eval mode.

### Model

- output shapes đúng;
- image/text attention tách riêng;
- padding mask hoạt động;
- score ∈ [0,1];
- fusion đúng cho both/text-only/image-only;
- inactive = N/A/masked, không phải 0.

### Loss

- UNKNOWN không ảnh hưởng loss;
- untrainable modality không ảnh hưởng loss;
- positive-only vẫn có finite/non-zero loss;
- no-valid-label batch → zero branch loss, không NaN.

### Metrics

- AUC bỏ proposition single-class đúng;
- macro AUC tính trên valid pairs;
- checkpoint metric không dùng disease label.

### Smoke test

```text
forward
→ loss
→ backward
→ optimizer step
→ checkpoint save/load
→ identical prediction after reload
```

---

## 23. Commit plan

Coding agent **không làm tất cả trong một commit**.

### Commit 1 — Stage 1 contracts and v3 artifact loaders

- rewrite `proposition/data`;
- load KB v3;
- load `labels_raw_v3.npz`;
- load active masks/CSV;
- ID alignment validation;
- basic tests;
- update README contracts.

Không model/training.

**Review gate:** schema, masking semantics, alignment, invariant.

### Commit 2 — Stage 1 dataset and caption masking

- Stage1 SkinCAP dataset;
- case_id mapping;
- global disease-name masking;
- train/val/test preprocessing;
- tests.

Không model/loss.

**Review gate:** kiểm không có label leakage qua dataset API.

### Commit 3 — Dual-modality Stage 1 evidence model

- query cache;
- frozen encoders integration;
- 8-head image/text cross-attention riêng;
- scalar heads;
- fixed fusion;
- modality masks;
- shape/freeze tests.

Không training loop lớn.

**Review gate:** đúng proposition semantics, không còn P1 disease path.

### Commit 4 — Masked weak-supervision loss

- masked BCE;
- per-proposition per-modality pos_weight;
- safe positive-only / negative-only;
- label distribution report;
- tests.

**Review gate:** UNKNOWN và positive-only behavior.

### Commit 5 — Stage 1 trainer + checkpointing

- optimizer;
- epoch loop;
- early stopping;
- resume;
- best checkpoint by macro AUC valid proposition/modality pairs;
- metrics artifacts.

**Review gate:** disease label không vào loss; checkpoint criterion đúng.

### Commit 6 — Stage 1 diagnostics

- within-patient differentiation;
- pairwise correlation;
- query swap;
- query ablation;
- attention artifacts;
- optional weak-label shuffle hook.

**Review gate:** diagnostics đủ để kiểm model thật sự dùng proposition query.

### Commit 7 — Colab / entrypoint / pilot config

- rewrite `main_proposition.py`;
- rewrite `configs/ws.yaml`;
- pilot disease selection;
- end-to-end smoke/pilot command;
- README run instructions.

Sau commit này mới chạy pilot thật.

---

## 24. Quy tắc review commit với ChatGPT

Sau mỗi commit, người dùng chỉ cần gửi:

```text
Review commit <SHA> trên branch WS theo STAGE1_WEAK_SUPERVISION_IMPLEMENTATION_PLAN.md
```

Quy trình review:

1. fetch đúng commit SHA;
2. compare với commit trước;
3. đọc toàn bộ file thay đổi;
4. đối chiếu plan;
5. kiểm regression vào RAD baseline;
6. kết luận:
   - PASS;
   - PASS WITH NOTES;
   - REQUEST CHANGES;
7. chỉ khi đạt gate mới đi commit tiếp.

Không review chỉ dựa vào commit message.

---

## 25. Các lỗi coding agent không được phép mắc

1. `UNKNOWN=-1` bị cast thành target 0.
2. Dùng disease label làm Stage-1 target.
3. Giữ PECL trong main Stage 1.
4. Giữ auxiliary disease branch.
5. Aggregate disease score trong Stage 1.
6. Áp polarity trước proposition support.
7. Average image/text khi một branch chưa được train.
8. Query chứa disease name, disease ID hoặc polarity.
9. Learned proposition ID embedding.
10. Encoder “freeze” bằng `requires_grad=False` nhưng vẫn update BN/dropout behavior.
11. Re-encode proposition text mỗi batch.
12. Dùng row index thay vì case/proposition IDs để align artifacts.
13. `pos_weight=0` cho positive-only proposition.
14. Chọn best checkpoint bằng disease macro-F1.
15. Report AUC trên single-class proposition.
16. Inactive proposition được gán score 0.
17. Stage 1 output gọi là disease probability.
18. Attention được mô tả như causal explanation.
19. Re-run / regenerate weak labels v3.
20. Thay đổi `main_rad.py` trong Stage 1.

---

## 26. Definition of Done — Stage 1 implementation

```text
[ ] v3 artifacts load + validate đúng
[ ] active set load từ artifact, không hard-code
[ ] student captions disease-masked
[ ] ResNet50 frozen thật sự
[ ] Bio_ClinicalBERT frozen thật sự
[ ] proposition queries cached
[ ] image/text cross-attention tách riêng
[ ] 8 heads, d=768
[ ] text/image scalar heads
[ ] masked BCE dùng đúng P/A/U
[ ] positive-only không zero-loss do pos_weight
[ ] fixed modality fusion
[ ] không disease loss
[ ] không PECL
[ ] không auxiliary disease branch
[ ] best checkpoint theo proposition macro AUC
[ ] resume checkpoint hoạt động
[ ] query swap chạy được
[ ] query ablation chạy được
[ ] score differentiation metrics được lưu
[ ] attention artifacts được lưu
[ ] synthetic smoke test pass
[ ] 3–5 disease pilot pass sanity gate
```

Chỉ sau đó mới thiết kế **Stage 2**.

---

## 27. Stage 2 — chưa implement ở vòng này

Không code Stage 2 trong các commit Stage 1 trừ interface tối thiểu export evidence scores.

Sau Stage 1 pass mới làm:

```text
frozen s_img / s_txt
→ fixed modality fusion
→ support s_ij
→ polarity
→ mean over active propositions per disease
→ monotonic calibration gamma_d > 0, b_d
→ softmax 47 diseases
```

Stage 2 chỉ train calibration/decision parameters.

Disease gradient không đi ngược vào evidence model.

---

## 28. Câu chuyện khoa học cần bảo toàn trong code

```text
KNOWLEDGE
Guideline propositions define WHAT TO ASK.

EVIDENCE
Weak supervision teaches HOW TO RECOGNIZE each proposition.

DECISION
Disease labels later teach HOW TO COMBINE frozen proposition evidence.
```

Stage 1 hiện tại chỉ triển khai tầng **EVIDENCE**.

Đây là ranh giới quan trọng nhất của toàn bộ refactor.
