# ZeroShot-350M — 337M Parameter LLM Trained from Scratch

[![Parameters](https://img.shields.io/badge/Parameters-337M-blue)](https://github.com/TobiasLogic/LocalAITraining)
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-orange)](https://github.com/TobiasLogic/LocalAITraining)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_5090_32GB-green)](https://github.com/TobiasLogic/LocalAITraining)
[![Dataset](https://img.shields.io/badge/Dataset-FineWeb--Edu_(Streamed)-yellow)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2)
[![Pipeline](https://img.shields.io/badge/Pipeline-Base_+_Mid_+_SFT-purple)](https://github.com/TobiasLogic/LocalAITraining)
[![Base Loss](https://img.shields.io/badge/Base_Loss-3.20-brightgreen)](https://github.com/TobiasLogic/LocalAITraining)
[![Cost](https://img.shields.io/badge/Total_Cost-~$7.88-red)](https://github.com/TobiasLogic/LocalAITraining)

ZeroShot-350M is a fully custom 337 million parameter language model trained entirely from scratch — base pre-training, mid-training, and supervised fine-tuning — on a single rented consumer GPU. The successor to [ZeroShot-124M](https://github.com/TobiasLogic/ZeroShot-124M), this model nearly triples the parameter count, doubles the context window, and pushes the base loss down to 3.20.

The entire pipeline goes from random weights to a functional chatbot,  on rented Vast.ai GPUs.

Total compute cost: **~$7.88**.

## Model Specifications

The architecture is a scaled-up GPT-2 style Decoder-only Transformer, designed to push what's possible on a single 32GB GPU.

* **Total Parameters:** ~337 Million
* **Vocabulary Size:** 50,304 (GPT-2 BPE via `tiktoken`, padded to 64 for tensor core alignment)
* **Embedding Dimensions (`n_embd`):** 1,200
* **Transformer Layers (`n_layer`):** 16
* **Attention Heads (`n_head`):** 12
* **Context Window (`block_size`):** 2,048 tokens
* **Activation Function:** GELU
* **Precision:** `bfloat16` (Mixed Precision)
* **Attention:** Flash Attention via `F.scaled_dot_product_attention`
* **Weight Tying:** Embedding and LM head share weights
* **Normalization:** LayerNorm (bias=False)

## Hardware & Training Setup

This model was trained on a rented Vast.ai instance located in Vietnam. Lesson learned from ZeroShot-124M: never rent behind the Great Firewall if your pipeline depends on HuggingFace.

* **GPU:** NVIDIA GeForce RTX 5090 (32GB GDDR7)
* **GPU Cost:** $0.309/hr
* **CUDA:** 12.8, Blackwell architecture (sm_120)
* **PyTorch:** Nightly build with cu128 (required for sm_120 support)
* **Throughput:** ~45,000 tokens/sec
* **Data:** All streamed from HuggingFace — zero disk usage for datasets

**Note on Blackwell GPUs:** Same as with the 5060 Ti — `torch.compile` is broken on sm_120 (Blackwell). Training runs with `--no-compile`. This costs ~20-30% throughput but works reliably. The 5090's raw speed more than compensates.

## The Three Training Stages

### Stage 1: Base Pre-training

The model learns English by predicting the next token on billions of unique web text tokens.

* **Dataset:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) (streamed directly from HuggingFace — zero disk usage)
* **Optimizer:** AdamW (fused)
* **Learning Rate:** 6e-4 with cosine decay to 6e-5
* **Warmup:** 2,000 steps
* **Effective Batch Size:** 64 (via gradient accumulation)
* **Total Steps:** 18,000
* **Tokens Seen:** ~4.7 Billion (all unique — no data repetition)
* **Final Loss:** 3.20

### Stage 2: Mid-training

The model learns conversation format — user/assistant turns, Q&A structure, instruction following.

* **Datasets:** [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) + [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) + [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst2)
* **Learning Rate:** 2e-4
* **Total Steps:** 5,000
* **Final Loss:** 1.30

### Stage 3: Supervised Fine-Tuning

Final polish — lower learning rate, same chat data, teaches the model to give cleaner, less repetitive answers.

* **Learning Rate:** 5e-5
* **Total Steps:** 2,000
* **Status:** ⚠️ *Pending — SFT not yet completed*

## Training Results

| Stage | Steps | Loss (start → end) | Status |
| --- | --- | --- | --- |
| Base Pre-training | 18,000 | 11.0 → 3.20 | ✅ Complete |
| Mid-training | 5,000 | 3.20 → 1.30 | ✅ Complete |
| SFT | 2,000 | TBD | ⚠️ Pending |
| **Total** | **~25,000** | | |

## Scaling Comparison: ZeroShot Family

|  | MicroGPT | ZeroShot-124M | **ZeroShot-350M** |
| --- | --- | --- | --- |
| Parameters | 30.5M | 124M | **337M** |
| GPU | RTX 3050 (4GB) | RTX 5060 Ti (16GB) | **RTX 5090 (32GB)** |
| Context Window | 128 tokens | 1,024 tokens | **2,048 tokens** |
| Embedding Dim | 384 | 768 | **1,200** |
| Layers | 6 | 12 | **16** |
| Heads | 6 | 12 | **12** |
| Dataset | FineWeb-Edu | FineWeb-Edu | **FineWeb-Edu** |
| Tokens Seen | ~1.6B | ~7.2B | **~4.7B** |
| Training Stages | Base only | Base + Mid + SFT | **Base + Mid + SFT** |
| Can Chat? | ❌ | ✅ | ✅ |
| Final Base Loss | 3.85 | 3.45 | **3.20** |
| Training Time | ~15 hours | ~29 hours | **~25 hours** |
| Total Cost | Free (local) | $0.54 | **~$7.88** |

The loss improvement from 3.45 → 3.20 is significant — it represents meaningfully better next-token prediction across the board. The 2x context window (2,048 vs 1,024) also means the model can attend to much longer passages.

## How to Use

### 1. Install Dependencies

```bash
pip install torch tiktoken datasets numpy
```

> **RTX 50-series (Blackwell) users:** You need the PyTorch nightly:
> ```bash
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

### 2. Download Checkpoints

Grab the checkpoint files from [GitHub Releases](https://github.com/TobiasLogic/LocalAITraining/releases):
- `ckpt_base_final.pt` (1.3GB) — base pre-trained model
- `ckpt_mid_final.pt` — after mid-training (ready for SFT)

### 3. Train (Full Pipeline)

```bash
python train.py train --no-compile
```

This runs all three stages automatically. Data is streamed — no disk space needed for datasets.

**Resume from checkpoint:**
```bash
python train.py train --resume checkpoints/ckpt_base_step10000.pt --no-compile
```

### 4. Fine-tune Only (Complete the SFT)

If you have the mid-training checkpoint and want to finish SFT:

```bash
python train.py finetune --checkpoint ckpt_mid_final.pt --no-compile --skip_mid
```

This skips base and mid-training, running only the SFT stage. Takes ~1.5 hours on an RTX 5090 (~2,000 steps).

### 5. Chat with Your Model

```bash
python train.py chat --checkpoint checkpoints/ckpt_sft_final.pt
```

Best prompts are concrete questions: "What is gravity?", "Explain how computers work", "What are the benefits of exercise?"

### 6. Text Completion (Base Model)

```bash
python train.py generate \
    --checkpoint ckpt_base_final.pt \
    --prompt "The development of artificial intelligence" \
    --max_tokens 200 \
    --temperature 0.7
```

## Key Lessons Learned (Building on 124M)

All the 124M lessons still apply, plus:

- **RTX 5090 (Blackwell):** Same `torch.compile` issue as the 5060 Ti — `--no-compile` is mandatory. But raw throughput of ~45k tok/s makes up for it.
- **Scaling is not free:** 337M on 4.7B tokens is somewhat undertrained by Chinchilla standards (ideal would be ~6.7B tokens). More tokens would push loss lower, but credits ran out.
- **Mid-training loss 1.3 is a great sign:** The model absorbed conversation structure much better than the 124M, likely because the larger embedding dimension captures more nuanced patterns.
- **Streaming is still king:** Zero disk usage for datasets. Critical when you're renting VPS instances with 32-50GB SSDs.
- **O(N²) concatenation bug:** Still relevant — always accumulate tokens to a list, then `np.concatenate` once. This was a 124M lesson but bit us again during debugging.

## Project Background

This is part of an ongoing series of from-scratch LLM training experiments:

1. **[MicroGPT](https://github.com/TobiasLogic/microgpt)** — 30.5M params, base-only, trained on a laptop RTX 3050
2. **[ZeroShot-124M](https://github.com/TobiasLogic/ZeroShot-124M)** — 124M params, full pipeline, RTX 5060 Ti, $0.54
3. **[ZeroShot-350M](https://github.com/TobiasLogic/LocalAITraining)** (this repo) — 337M params, full pipeline, RTX 5090, ~$7.88
4. **ZeroShot-1B** — planned next

All built by [Tobias](https://github.com/TobiasLogic)

## Cost Breakdown

| Resource | Cost |
| --- | --- |
| Vast.ai RTX 5090 (Vietnam) | $0.309/hr |
| Base training (~18hrs) | ~$5.56 |
| Mid-training (~5hrs) | ~$1.55 |
| SFT (estimated ~2hrs) | ~$0.62 |
| **Total (~25hrs)** | **~$7.88** |

## Acknowledgments

* [nanoGPT](https://github.com/karpathy/nanoGPT) & [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
* [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) for the tokens-to-parameters ratio
* [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) by HuggingFace


## License

[MIT](LICENSE)
