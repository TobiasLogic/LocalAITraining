# ZeroShot-350M — 337M Parameter LLM Trained from Scratch

[![Parameters](https://img.shields.io/badge/Parameters-337M-blue)](https://github.com/TobiasLogic/ZeroShot-350M)
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-orange)](https://github.com/TobiasLogic/ZeroShot-350M)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_5090_32GB-green)](https://github.com/TobiasLogic/ZeroShot-350M)
[![Dataset](https://img.shields.io/badge/Dataset-FineWeb--Edu_(Streamed)-yellow)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2)
[![Pipeline](https://img.shields.io/badge/Pipeline-Base_+_Mid_+_SFT-purple)](https://github.com/TobiasLogic/ZeroShot-350M)
[![SFT Loss](https://img.shields.io/badge/SFT_Loss-~1.3-brightgreen)](https://github.com/TobiasLogic/ZeroShot-350M)
[![Cost](https://img.shields.io/badge/Total_Cost-~$7.88-red)](https://github.com/TobiasLogic/ZeroShot-350M)

ZeroShot-350M is a fully custom 337 million parameter language model trained entirely from scratch — base pre-training, mid-training, and supervised fine-tuning — on a single rented consumer GPU. The successor to [ZeroShot-124M](https://github.com/TobiasLogic/ZeroShot-124M), this model nearly triples the parameter count, doubles the context window, and pushes the base loss down to 3.20.

The entire pipeline goes from random weights to a functional chatbot, on rented Vast.ai GPUs.

Total compute cost: **~$7.88**.

## Sample Output

```
You: What is the meaning of life?

AI: The meaning of life is a profound and universal one that is deeply rooted in our
human experience of the universe. It is a state of being that has been shaped by our
experiences, emotions, and the choices we make. It is a state of being that allows us
to experience the world through our senses, body, and soul, which are intertwined with
our sense of identity and purpose.

The meaning of life is often described in terms of how it can be expressed and
experienced. It is about how we can define ourselves, what we value, and what we are
capable of achieving. It is about finding meaning and purpose in our lives, and how we
can make choices that are meaningful and fulfilling.
```

## Model Specifications

| | |
|---|---|
| **Total Parameters** | ~337 Million |
| **Vocabulary Size** | 50,304 (GPT-2 BPE via `tiktoken`, padded for tensor cores) |
| **Embedding Dimensions** | 1,200 |
| **Transformer Layers** | 16 |
| **Attention Heads** | 12 |
| **Context Window** | 2,048 tokens |
| **Activation** | GELU |
| **Precision** | bfloat16 |
| **Attention** | Flash Attention via `F.scaled_dot_product_attention` |
| **Weight Tying** | Embedding and LM head share weights |
| **Normalization** | LayerNorm (bias=False) |

## Training Results

All three stages are complete. The model goes from random weights to a chatbot in ~25 hours.

| Stage | Steps | Loss (start → end) | Status |
|---|---|---|---|
| Base Pre-training | 18,000 | 11.0 → 3.20 | ✅ Complete |
| Mid-training | 5,000 | 3.20 → 1.30 | ✅ Complete |
| SFT | 2,000 | 1.32 → ~1.3 | ✅ Complete |

## Hardware & Training Setup

Trained on a rented Vast.ai instance located in Vietnam. Lesson learned from ZeroShot-124M: never rent behind the Great Firewall if your pipeline depends on HuggingFace.

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5090 (32GB GDDR7) |
| **GPU Cost** | $0.309/hr |
| **CUDA** | 12.8, Blackwell architecture (sm_120) |
| **PyTorch** | Nightly build with cu128 (required for sm_120) |
| **Throughput** | ~45,000 tokens/sec (base), ~22,500 tok/s (SFT) |
| **Data** | All streamed from HuggingFace — zero disk usage for datasets |

> **Note on Blackwell GPUs:** `torch.compile` is broken on sm_120 (Blackwell). Training runs with `--no-compile`. This costs ~20-30% throughput but works reliably. The 5090's raw speed more than compensates.

## The Three Training Stages

### Stage 1: Base Pre-training

The model learns English by predicting the next token on billions of unique web text tokens.

| | |
|---|---|
| **Dataset** | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) (streamed) |
| **Optimizer** | AdamW (fused) |
| **Learning Rate** | 6e-4 → 6e-5 (cosine decay) |
| **Warmup** | 2,000 steps |
| **Effective Batch** | 64 sequences (via gradient accumulation) |
| **Steps** | 18,000 |
| **Tokens Seen** | ~4.7B (all unique, no repetition) |
| **Final Loss** | 3.20 |

### Stage 2: Mid-training

The model learns conversation format — user/assistant turns, Q&A structure, instruction following.

| | |
|---|---|
| **Datasets** | [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (80k) + [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (50k) + [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst2) (50k) |
| **Learning Rate** | 1e-4 |
| **Steps** | 5,000 |
| **Final Loss** | 1.30 |

### Stage 3: Supervised Fine-Tuning

Final polish — lower learning rate, same chat data, teaches the model to produce cleaner responses.

| | |
|---|---|
| **Learning Rate** | 3e-5 → 3e-6 (cosine decay) |
| **Steps** | 2,000 |
| **Final Loss** | ~1.3 |

## Scaling Comparison: ZeroShot Family

| | MicroGPT | ZeroShot-124M | **ZeroShot-350M** |
|---|---|---|---|
| Parameters | 30.5M | 124M | **337M** |
| GPU | RTX 3050 (4GB) | RTX 5060 Ti (16GB) | **RTX 5090 (32GB)** |
| Context Window | 128 tokens | 1,024 tokens | **2,048 tokens** |
| Embedding Dim | 384 | 768 | **1,200** |
| Layers | 6 | 12 | **16** |
| Heads | 6 | 12 | **12** |
| Dataset | FineWeb-Edu | FineWeb-Edu | **FineWeb-Edu** |
| Tokens Seen | ~1.6B | ~7.2B | **~4.7B** |
| Training Stages | Base only | Base + Mid + SFT | **Base + Mid + SFT** |
| Can Chat? | ❌ | ✅ | **✅** |
| Final Base Loss | 3.85 | 3.45 | **3.20** |
| Training Time | ~15 hours | ~29 hours | **~25 hours** |
| Total Cost | Free (local) | $0.54 | **~$7.88** |

## How to Use

### Install Dependencies

```bash
pip install torch tiktoken datasets numpy
```

> **RTX 50-series (Blackwell) users:** You need PyTorch nightly:
> ```bash
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

### Download Checkpoints

Grab checkpoint files from [GitHub Releases](https://github.com/TobiasLogic/ZeroShot-350M/releases/tag/v1.0-sft):

| Checkpoint | Description |
|---|---|
| `ckpt_sft_final.pt` | **Final model** — ready to chat |
| `ckpt_mid_final.pt` | After mid-training (if you want to re-do SFT) |
| `ckpt_base_final.pt` | Base pre-trained only (text completion) |

### Chat

```bash
python train.py chat --checkpoint ckpt_sft_final.pt
```

### Generate Text (Base Model)

```bash
python train.py generate \
    --checkpoint ckpt_base_final.pt \
    --prompt "The development of artificial intelligence" \
    --max_tokens 200 \
    --temperature 0.7
```

### Train from Scratch

```bash
python train.py train --no-compile
```

Runs all three stages automatically. Data is streamed — no disk space needed for datasets.

### Resume / Re-run SFT

```bash
# Resume from any checkpoint
python train.py train --resume checkpoints/ckpt_base_step10000.pt --no-compile

# Run mid-training + SFT on an existing base checkpoint
python train.py finetune --checkpoint ckpt_base_final.pt --no-compile

# Skip mid-training, SFT only
python train.py train --skip_base --skip_mid --resume ckpt_mid_final.pt --no-compile
```

## Cost Breakdown

| Resource | Cost |
|---|---|
| Vast.ai RTX 5090 (Vietnam) | $0.309/hr |
| Base training (~18hrs) | ~$5.56 |
| Mid-training (~5hrs) | ~$1.55 |
| SFT (~1hr) | ~$0.31 |
| **Total** | **~$7.88** |

## Key Lessons Learned

All the 124M lessons still apply, plus:

- **RTX 5090 (Blackwell):** Same `torch.compile` issue as the 5060 Ti — `--no-compile` is mandatory. Raw throughput of ~45k tok/s makes up for it.
- **Scaling is not free:** 337M on 4.7B tokens is slightly undertrained by Chinchilla standards (ideal ~6.7B). More tokens would push loss lower, but credits ran out.
- **Mid-training loss 1.3 is strong:** The model absorbed conversation structure much better than 124M, likely because the larger embedding dimension captures more nuanced patterns.
- **Streaming is still king:** Zero disk usage for datasets. Critical when renting VPS instances with limited SSDs.
- **16GB GPUs can SFT 350M:** Batch size 1 with gradient accumulation works fine — only uses ~10GB VRAM.

## Project Background

This is part of an ongoing series of from-scratch LLM training experiments:

1. **[MicroGPT](https://github.com/TobiasLogic/microgpt)** — 30.5M params, base-only, trained on a laptop RTX 3050
2. **[ZeroShot-124M](https://github.com/TobiasLogic/ZeroShot-124M)** — 124M params, full pipeline, RTX 5060 Ti, $0.54
3. **[ZeroShot-350M](https://github.com/TobiasLogic/ZeroShot-350M)** (this repo) — 337M params, full pipeline, RTX 5090, ~$7.88
4. **ZeroShot-500M** — next up

All built by [Tobias](https://github.com/TobiasLogic).

## Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) & [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) for the tokens-to-parameters ratio
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) by HuggingFace

## License

[MIT](LICENSE)
