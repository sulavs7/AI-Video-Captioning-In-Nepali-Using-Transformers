# AI Video Captioning in Nepali Using Transformers

An end-to-end deep learning system that generates video captions directly in Nepali (Devanagari script). Built as a bachelor thesis at Khwopa College of Engineering and presented at the **KHWOPA CEEL 2026 National Conference**.

---

## Overview

This project addresses the challenge of automatic video description in a low-resource language — Nepali. The system takes a video as input and outputs a descriptive caption in Devanagari script, without any intermediate English translation step.

The architecture bridges a video understanding model (TimeSformer) with a multilingual language model (NLLB-200) using a custom Q-Former cross-modal adapter — inspired by BLIP-2. LoRA fine-tuning is applied to the decoder to reduce trainable parameters while preserving generalization.

---

## Architecture

```
Video (.mp4)
    │
    ▼
TimeSformer (facebook/timesformer-base-finetuned-k600)
    │   Extracts spatiotemporal features from 16 uniformly sampled frames
    │   Output: (B, N, 768)
    │
    ▼
Linear Projection (768 → 1024)
    │
    ▼
Q-Former (4 layers, 64 query tokens)
    │   Cross-attention: queries attend to video features
    │   Self-attention:  queries refine among themselves
    │   Output: (B, 64, 1024) — compressed visual representation
    │
    ▼
NLLB-200-distilled-600M Decoder (facebook/nllb-200-distilled-600M)
    │   LoRA applied to all attention projections (q, k, v, out)
    │   Generates Nepali tokens autoregressively
    │
    ▼
Nepali Caption (Devanagari script)
```

### Key Design Choices

- **TimeSformer** was chosen for its efficient divided space-time attention over video frames, pretrained on Kinetics-600.
- **Q-Former** acts as a bottleneck adapter — instead of feeding all video tokens (which can be 1000+) directly to the decoder, 64 learned query tokens compress the visual information. This is both memory-efficient and more stable for training.
- **NLLB-200** was selected over mBART because it has native Nepali (`npi_Deva`) support with a dedicated tokenizer, making it far better suited for low-resource Nepali generation.
- **LoRA** (r=16, alpha=32) targets all attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) in the NLLB decoder, drastically reducing trainable parameters without sacrificing performance.
- **Random caption sampling** — during training, one of the multiple available captions per video is randomly selected per epoch, acting as data augmentation and improving generalization.

---

## Results

Evaluated on the **MSVD-Nepali** benchmark:

| Metric   | Score |
|----------|-------|
| BLEU-4   | 34.07 |
| METEOR   | 58.24 |
| ROUGE-L  | 64.53 |
| CIDEr    | 79.74 |

Evaluation used **SacreBLEU** with `tokenize="intl"` for correct Devanagari tokenization, and NFC Unicode normalization to ensure consistent character representation.

---

## Dataset

Training was done on **VATEX-Nepali** — a translated version of the VATEX dataset with Nepali captions. The dataset contains multiple captions per video, stored in JSON format.

```
VATEX_ne/
├── videos_train_val/       # .mp4 video files
└── nepali_captions/
    ├── vatex_train_ne.json
    ├── vatex_val_ne.json
    └── vatex_test_ne.json
```

Each JSON entry:
```json
{
  "video_id": "video1234",
  "caption": ["नेपाली क्याप्सन १", "नेपाली क्याप्सन २", "..."]
}
```

---

## Training Setup

| Config                    | Value                              |
|---------------------------|------------------------------------|
| Batch size (per device)   | 4                                  |
| Gradient accumulation     | 4 (effective batch size = 16)      |
| Precision                 | FP16                               |
| Epochs                    | 12–30 (resumed from checkpoints)   |
| Optimizer                 | AdamW (differential LR)           |
| TimeSformer LR            | 1e-5                               |
| Q-Former / Decoder LR     | 1e-4                               |
| LR Scheduler              | Cosine with warmup (10%)           |
| Frames per video          | 16 (uniformly sampled)             |
| Max caption length        | 64 tokens                          |
| LoRA rank (r)             | 16, alpha=32                       |
| Platform                  | Kaggle (GPU), RunPod               |
| Checkpoints               | HuggingFace Hub                    |
| Experiment tracking       | Weights & Biases (WandB)           |

**Freezing strategy:**
- TimeSformer: frozen except last 3 transformer blocks (partially unfrozen for fine-tuning)
- NLLB encoder: fully frozen (not used — Q-Former output feeds directly to the decoder)
- NLLB decoder: trainable via LoRA

---

## Project Structure

```
├── nllb-model-caption-sampling-lora-q64-vatex.ipynb   # Main training notebook
├── README.md
```



## Pre-trained Model

The trained checkpoint is available on HuggingFace Hub:

🤗 [`sulavs7/vatex-nllb-sampling-q64-lora`](https://huggingface.co/sulavs7/vatex-nllb-sampling-q64-lora)

---

## Dependencies

```bash
pip install torch transformers peft av wandb python-dotenv
```

Core libraries:
- `transformers` — TimeSformer, NLLB-200, tokenizers
- `peft` — LoRA via `get_peft_model`
- `av` (PyAV) — video decoding
- `wandb` — experiment tracking

---

## Citation / Acknowledgement

This work was conducted as a Bachelor's thesis at **Khwopa College of Engineering, Bhaktapur, Nepal** (2022–2026) and presented at the **National Conference on Computer, Electrical, Electronics & Communication (KHWOPA CEEL 2026)**.

---

## Author

**Sulav Sapkota**
- GitHub: [github.com/sulavs7](https://github.com/sulavs7)
- Email: sulavspkt7@gmail.com
