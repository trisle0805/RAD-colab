# Running RAD on Google Colab

This repository was originally launched through Slurm scripts. Google Colab does not provide Slurm, so do **not** run `skin.sh`, `fair.sh`, `nacc.sh`, or `icd.sh`. The updated Python entry points support a single CUDA GPU process, which is the recommended Colab setup for an A100 or H100 runtime.

> Do not upload restricted clinical data, credentials, or Hugging Face access tokens to a public notebook or repository. Follow the access agreements for MIMIC, NACC, FairVLMed, and SkinCAP.

## 1. Runtime and repository

In Colab, choose **Runtime → Change runtime type → GPU**. Clone the repository and enter it:

```bash
%cd /content
!git clone https://github.com/tdlhl/RAD.git
%cd /content/RAD
!nvidia-smi
```

The provided `RAD.yml` and `requirements.txt` describe the authors' broad Linux research environment. They are not a minimal Colab installation recipe. In particular, do **not** install the entire `requirements.txt` before a baseline run: it includes packages not imported by RAD training and packages with CUDA compilation/platform requirements, such as FlashAttention, DeepSpeed, bitsandbytes, nmslib, LLaVA, and ms-swift.

## 2. Install the focused training dependencies

First inspect the PyTorch stack already included in the current Colab GPU image:

```python
import torch, torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

For SkinCAP, install the focused dependency file. It intentionally does not pin `torch` or `torchvision`; retain Colab's compatible GPU pair unless an observed error requires changing it.

```bash
!pip install -q -r requirements-colab-skin.txt
```

For NACC, additionally install NIfTI and scientific-image dependencies:

```bash
!pip install -q "nibabel==5.3.2" "scipy==1.14.1"
```

For ICD53, the entry point imports `pytorch-pretrained-vit` even when using the default ResNet encoder:

```bash
!pip install -q "pytorch-pretrained-vit==0.0.7"
```

If you intentionally replace Colab's PyTorch installation, install a **matched** `torch`/`torchvision` pair from the official PyTorch wheel index. The author-pinned pair is `torch==2.4.1+cu121` and `torchvision==0.19.1+cu121`; do not mix it with an unrelated torchvision release.

## 3. Persistent data and outputs

Mount Drive if you need to preserve checkpoints. Training directly from Drive can be slow; copy data to `/content` for training when local disk capacity permits, then save outputs back to Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

The YAML files may contain default dataset paths. Use the command-line overrides below for private local or Colab paths, so they do not need to be committed into a YAML file:

| Field | Meaning |
|---|---|
| `--train_csv` | Training CSV path; overrides `ICD_train_file` |
| `--test_csv` | Validation/test CSV path; overrides `ICD_test_file` |
| `--image_root` | Root prepended to relative image paths in column 0 of the CSV; overrides `image_root` |

Absolute image paths in the CSV are also accepted and do not need `--image_root`.

Expected CSV formats are positional:

| Task | CSV columns |
|---|---|
| FairVLMed | image `.npz` relative path, clinical note, ignored columns, labels from column 3 onward. Each `.npz` needs `slo_fundus`. |
| SkinCAP | relative image path, text, labels from column 2 onward. |
| NACC | relative NIfTI path, EHR text, labels from column 2 onward. |
| ICD53 | relative MIMIC JPEG path, text, 53 labels from column 2 onward. For `main_rad_icd.py`, text must follow `<Report>: ... <EHR>: ...`. |

The label ordering must match both the supplied task guideline JSONL and the hard-coded label lists in the training engines. The provided SkinCAP CSVs use `pilar cyst` at label index 30, and the SkinCAP engine label lists and supplied guideline JSONL have been aligned with that order. Do not reorder CSV label columns.

### SkinCAP data layout

The supplied SkinCAP CSVs are already preprocessed; do **not** run anything in `preprocess/skin/` again. Their first column contains image filenames such as `3413.png`, so the loader combines each filename with the configured image root.

In a Colab cell, set your private dataset root once. Replace the placeholder with your own Drive location:

```python
SKINCAP_ROOT = "/content/drive/MyDrive/path/to/SkinCAP"
```

The expected layout beneath that root is:

```text
SkinCAP/
├── skincap_50_train_set.csv
├── skincap_50_test_set.csv
└── skincap/
  └── 3413.png
```

The CSV files are intentionally stored outside the cloned repository, so `csv_files/` is not required. Supply all three private paths with the SkinCAP command below instead of editing and committing `configs/skin.yaml`.

## 4. ClinicalBERT and pretrained vision weights

`--bert_model_name` is required. It must identify a compatible Hugging Face BERT model with hidden size 768. A local directory is recommended for reproducibility; the tokenizer and model loader then both operate offline.

Example directory layout:

```text
/content/models/ClinicalBERT/
  config.json
  model.safetensors  # or pytorch_model.bin
  tokenizer_config.json
  vocab.txt
  ...
```

For a Hugging Face model ID, the Colab runtime needs network access and any necessary authentication. The default ResNet-50 vision encoder also downloads ImageNet weights on first use if they are not in the Torch cache. Let those downloads complete before starting an expensive run, or pre-cache them.

## 5. Run one process on one GPU

Do not use `srun`, `scontrol`, `torchrun`, or Slurm environment variables on Colab. The source has a distributed initialization helper, but it does **not** wrap models with DistributedDataParallel; multi-process training would therefore be incorrect.

Set the optional CUDA allocator configuration before Python begins:

```bash
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### SkinCAP example

Run:

```bash
%cd /content/RAD
!python main_rad.py \
  --config ./configs/skin.yaml \
  --dataset skin \
  --train_csv "{SKINCAP_ROOT}/skincap_50_train_set.csv" \
  --test_csv "{SKINCAP_ROOT}/skincap_50_test_set.csv" \
  --image_root "{SKINCAP_ROOT}/skincap" \
  --output_dir /content/drive/MyDrive/RAD-outputs/skin-run-01 \
  --bert_model_name /content/models/ClinicalBERT \
  --guideline_path ./guideline/qwen_maxtoken2k_skincap50_4sources.jsonl \
  --max_length 512 --embed_dim 768 \
  --loss_ratio 0.1 --contrast_ratio_text 0.1 --contrast_ratio_vision 0.001
```

### FairVLMed example

```bash
!python main_rad.py \
  --config ./configs/fair.yaml \
  --dataset fair_ori \
  --image_root /content/Harvard-FairVLMed \
  --output_dir /content/drive/MyDrive/RAD-outputs/fair-run-01 \
  --bert_model_name /content/models/ClinicalBERT \
  --guideline_path ./guideline/qwen_maxtoken2k_fair_4sources.jsonl \
  --max_length 512 --embed_dim 768 \
  --loss_ratio 0.1 --contrast_ratio_text 2 --contrast_ratio_vision 0.001
```

### NACC example

```bash
!python main_rad.py \
  --config ./configs/nacc.yaml \
  --dataset nacc \
  --image_root /content/NACC_images \
  --output_dir /content/drive/MyDrive/RAD-outputs/nacc-run-01 \
  --bert_model_name /content/models/ClinicalBERT \
  --guideline_path ./guideline/qwen_maxtoken2k_nacc11_4sources.jsonl \
  --max_length 512 --embed_dim 768 \
  --loss_ratio 0.1 --contrast_ratio_text 0.1 --contrast_ratio_vision 0.001
```

### ICD53 example

Use the specialized dual report/EHR entry point:

```bash
!python main_rad_icd.py \
  --config ./configs/ICD53.yaml \
  --dataset icd53 \
  --image_root /content/MIMIC-CXR-JPG/files \
  --output_dir /content/drive/MyDrive/RAD-outputs/icd53-run-01 \
  --bert_model_name /content/models/ClinicalBERT \
  --guideline_path ./guideline/qwen_maxtoken2k_icd53_4sources.jsonl \
  --max_length 512 --embed_dim 768 \
  --loss_ratio 0.1 --contrast_ratio_text 0.1 --contrast_ratio_vision 0.001
```

`--guideline_path` must not be omitted: the active loss and final evaluation require the guideline features/predictions.

## 6. Start conservatively, then tune

The configuration now uses two DataLoader workers. On a Drive-backed dataset, begin at `0–2` workers; increase only after measuring. The original batch sizes were designed for the authors' infrastructure. Gradient accumulation does not reduce the activation memory of an individual batch.

Recommended initial smoke-test batch sizes:

| Task | Initial `batch_size` / `test_batch_size` |
|---|---:|
| FairVLMed | 8 / 8 |
| SkinCAP | 4 / 4 |
| ICD53 ($512\times512$) | 2 / 2 |
| NACC ($96^3$ 3D volumes) | 1 / 1 |

After a successful full epoch, increase batch size gradually. An A100/H100 has large memory, but ICD53 tokenizes report and EHR inputs up to 512 tokens and NACC trains a 3D ResNet, so out-of-memory failures remain possible. The current training loop uses FP32; automatic mixed precision is not yet implemented.

## 7. Checkpoints and diagnostics

- The program resumes automatically from the numerically latest `checkpoint_*.pt` in an existing output directory. Use a new directory for a fresh run.
- Checkpoints are currently saved once every 10 epochs. Colab runtimes are ephemeral, so use a Drive output directory and consider reducing the checkpoint interval before a long experiment.
- Each epoch writes `gt_epoch_*.npy`, `pred_epoch_*.npy`, and guideline predictions to the output directory. Budget sufficient Drive space.
- Before a full run, verify one CSV row's resolved image path and execute a short smoke test. Small or one-class validation subsets may fail in the current F1/confusion-matrix evaluation code.

## What was changed for Colab portability

1. Dataset roots are now supplied through `image_root` in YAML or `--image_root`; the former `/your_path/...` literals were removed from active dataset loaders.
2. `num_workers` and `test_num_workers` in YAML now control the DataLoaders; they default to 2 in the supplied task configs.
3. Entry points default to a single process and reject CPU-only runs clearly.
4. The ICD entry point defaults to the existing `configs/ICD53.yaml` and reports a clear error if used for FairVLMed.
5. Local ClinicalBERT directories now load tokenizer and encoder consistently with offline mode.
6. Distributed initialization no longer overwrites an external rendezvous address/port. This is not a multi-GPU implementation; keep one process on Colab.