"""
ZeroShot-350M — Full LLM Training Pipeline
Target: 1x RTX 6000 Ada (48GB VRAM) — ~24 hours budget
Data:   FineWeb-Edu (streamed), SmolTalk + Alpaca + OpenAssistant (chat)

Stage 1: Base pretraining  — learn English from web text (~18 hrs)
Stage 2: Midtraining       — learn conversation format (~3 hrs)
Stage 3: SFT               — polish chat quality (~1.5 hrs)

Model: ~350M parameters
  - 16 layers, 12 heads, 1200 embedding dim
  - 2048 context window
  - ~8B tokens from FineWeb-Edu (streamed, no disk)
"""

import os
import math
import time
import json
import logging
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 50304          # padded to 64 for tensor cores
    block_size: int = 2048           # 2k context — longer coherent outputs
    n_layer: int = 16                # balanced depth
    n_head: int = 12                 # 1200 / 12 = 100 head_dim
    n_embd: int = 1200              # 3x wider than 124M
    dropout: float = 0.1
    bias: bool = False

@dataclass
class TrainConfig:
    # base training
    batch_size: int = 8              # bigger micro batch — 350M fits easily in 48GB
    gradient_accumulation_steps: int = 16  # effective batch = 8*16 = 128 seqs = 262k tokens/step
    max_steps: int = 30_000          # ~7.9B tokens — Chinchilla optimal for 350M
    learning_rate: float = 5e-4      # slightly higher for smaller model
    min_lr: float = 5e-5
    warmup_steps: int = 1500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # system
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True       # Ada supports torch.compile

    # logging
    log_interval: int = 25
    eval_interval: int = 500
    save_interval: int = 5000
    out_dir: str = "checkpoints"
    resume: str = ""


# ---------------------------------------------------------------------------
# 1. MODEL — GPT-2 XL scale
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn  = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp   = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()
        print(f"Model parameters: {n_params / 1e6:.1f}M ({n_params / 1e9:.2f}B)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# 2. STREAMING DATA — FineWeb-Edu
# ---------------------------------------------------------------------------

def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


class StreamingTokenBuffer:
    """Streams FineWeb-Edu from HuggingFace, tokenizes on the fly."""

    def __init__(self, block_size: int, batch_size: int,
                 buffer_size: int = 2000,
                 dataset_name: str = "HuggingFaceFW/fineweb-edu-score-2",
                 dataset_subset: str = "default"):
        self.block_size = block_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.enc = get_tokenizer()
        self.token_buffer = np.array([], dtype=np.uint16)
        self._stream = None

    def _get_stream(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, self.dataset_subset,
                          split="train", streaming=True)
        ds = ds.shuffle(seed=random.randint(0, 2**31), buffer_size=self.buffer_size)
        return iter(ds)

    def _refill_buffer(self, min_tokens: int):
        if self._stream is None:
            self._stream = self._get_stream()
        new_tokens = []
        current_len = len(self.token_buffer)
        while current_len + len(new_tokens) < min_tokens:
            try:
                doc = next(self._stream)
            except StopIteration:
                self._stream = self._get_stream()
                doc = next(self._stream)
            text = doc.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            tokens = self.enc.encode_ordinary(text)
            tokens.append(self.enc.eot_token)
            new_tokens.extend(tokens)
        if new_tokens:
            self.token_buffer = np.concatenate([
                self.token_buffer, np.array(new_tokens, dtype=np.uint16)
            ])

    def get_batch(self) -> tuple:
        needed = self.batch_size * (self.block_size + 1)
        self._refill_buffer(needed)
        chunk = self.token_buffer[:needed].astype(np.int64)
        self.token_buffer = self.token_buffer[needed:]
        chunk = chunk.reshape(self.batch_size, self.block_size + 1)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        return x, y


# ---------------------------------------------------------------------------
# 3. CONVERSATION DATA FOR MIDTRAINING + SFT
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {
    "user_start": "<|user|>",
    "user_end": "<|end_user|>",
    "assistant_start": "<|assistant|>",
    "assistant_end": "<|end_assistant|>",
    "system_start": "<|system|>",
    "system_end": "<|end_system|>",
}


def build_chat_datasets():
    """Download and prepare chat datasets for midtraining and SFT."""
    from datasets import load_dataset
    conversations = []

    # SmolTalk
    print("Loading SmolTalk conversations...")
    try:
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        count = 0
        for doc in ds:
            messages = doc.get("messages", [])
            if not messages:
                continue
            formatted = format_conversation(messages)
            if formatted and len(formatted) > 50:
                conversations.append(formatted)
                count += 1
                if count >= 80000:
                    break
        print(f"  loaded {count} SmolTalk conversations")
    except Exception as e:
        print(f"  SmolTalk failed: {e}")

    # OpenAssistant
    print("Loading OpenAssistant conversations...")
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        count = 0
        for doc in ds:
            text = doc.get("text", "")
            role = doc.get("role", "")
            if role == "prompter":
                formatted = f"{SPECIAL_TOKENS['user_start']}{text}{SPECIAL_TOKENS['user_end']}"
                conversations.append(formatted)
                count += 1
            elif role == "assistant" and conversations:
                conversations[-1] += f"{SPECIAL_TOKENS['assistant_start']}{text}{SPECIAL_TOKENS['assistant_end']}"
                count += 1
            if count >= 50000:
                break
        print(f"  loaded {count} OpenAssistant entries")
    except Exception as e:
        print(f"  OpenAssistant failed: {e}")

    # Alpaca
    print("Loading Alpaca instructions...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count = 0
        for doc in ds:
            instruction = doc.get("instruction", "")
            inp = doc.get("input", "")
            output = doc.get("output", "")
            if not instruction or not output:
                continue
            user_msg = instruction
            if inp:
                user_msg += f"\n{inp}"
            formatted = (
                f"{SPECIAL_TOKENS['user_start']}{user_msg}{SPECIAL_TOKENS['user_end']}"
                f"{SPECIAL_TOKENS['assistant_start']}{output}{SPECIAL_TOKENS['assistant_end']}"
            )
            conversations.append(formatted)
            count += 1
            if count >= 50000:
                break
        print(f"  loaded {count} Alpaca instructions")
    except Exception as e:
        print(f"  Alpaca failed: {e}")

    random.shuffle(conversations)
    print(f"Total chat examples: {len(conversations)}")
    return conversations


def format_conversation(messages: list) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role in ("user", "human"):
            parts.append(f"{SPECIAL_TOKENS['user_start']}{content}{SPECIAL_TOKENS['user_end']}")
        elif role in ("assistant", "gpt"):
            parts.append(f"{SPECIAL_TOKENS['assistant_start']}{content}{SPECIAL_TOKENS['assistant_end']}")
        elif role == "system":
            parts.append(f"{SPECIAL_TOKENS['system_start']}{content}{SPECIAL_TOKENS['system_end']}")
    return "".join(parts)


class ChatTokenBuffer:
    """Tokenizes chat conversations and yields training batches."""

    def __init__(self, conversations: list, block_size: int, batch_size: int):
        self.block_size = block_size
        self.batch_size = batch_size
        self.enc = get_tokenizer()
        self.conversations = conversations
        self.idx = 0
        self.token_buffer = np.array([], dtype=np.uint16)

    def _refill_buffer(self, min_tokens: int):
        new_tokens = []
        current_len = len(self.token_buffer)
        while current_len + len(new_tokens) < min_tokens:
            if self.idx >= len(self.conversations):
                self.idx = 0
                random.shuffle(self.conversations)
            text = self.conversations[self.idx]
            self.idx += 1
            tokens = self.enc.encode_ordinary(text)
            tokens.append(self.enc.eot_token)
            new_tokens.extend(tokens)
        if new_tokens:
            self.token_buffer = np.concatenate([
                self.token_buffer, np.array(new_tokens, dtype=np.uint16)
            ])

    def get_batch(self) -> tuple:
        needed = self.batch_size * (self.block_size + 1)
        self._refill_buffer(needed)
        chunk = self.token_buffer[:needed].astype(np.int64)
        self.token_buffer = self.token_buffer[needed:]
        chunk = chunk.reshape(self.batch_size, self.block_size + 1)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        return x, y


# ---------------------------------------------------------------------------
# 4. LR SCHEDULE
# ---------------------------------------------------------------------------

def get_lr(step, max_steps, lr, min_lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------------------------

class LossTracker:
    """Tracks loss, LR, throughput over training and saves to JSON + PNG plot."""

    def __init__(self, out_dir: str, stage_name: str):
        self.out_dir = out_dir
        self.stage_name = stage_name
        self.steps = []
        self.losses = []
        self.lrs = []
        self.tok_per_sec = []
        self.timestamps = []

    def log(self, step, loss, lr, tok_s):
        self.steps.append(step)
        self.losses.append(loss)
        self.lrs.append(lr)
        self.tok_per_sec.append(tok_s)
        self.timestamps.append(time.time())

    def save_json(self):
        path = os.path.join(self.out_dir, f"loss_{self.stage_name}.json")
        data = {
            "stage": self.stage_name,
            "steps": self.steps,
            "losses": self.losses,
            "learning_rates": self.lrs,
            "tokens_per_sec": self.tok_per_sec,
            "timestamps": self.timestamps,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def save_plot(self):
        """Generate a loss curve PNG using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
            fig.suptitle(f"ZeroShot-350M — {self.stage_name.upper()} Training", fontsize=16, fontweight="bold")

            # loss curve
            ax1.plot(self.steps, self.losses, color="#2196F3", linewidth=1.0, alpha=0.6, label="Raw loss")
            # smoothed loss (rolling average)
            if len(self.losses) > 20:
                window = min(50, len(self.losses) // 5)
                smoothed = np.convolve(self.losses, np.ones(window) / window, mode="valid")
                smoothed_steps = self.steps[window - 1:]
                ax1.plot(smoothed_steps, smoothed, color="#F44336", linewidth=2.0, label=f"Smoothed (window={window})")
            ax1.set_ylabel("Loss", fontsize=12)
            ax1.set_xlabel("Step", fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Loss: {self.losses[0]:.2f} → {self.losses[-1]:.2f}", fontsize=12)

            # learning rate curve
            ax2.plot(self.steps, self.lrs, color="#4CAF50", linewidth=1.5)
            ax2.set_ylabel("Learning Rate", fontsize=12)
            ax2.set_xlabel("Step", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

            plt.tight_layout()
            path = os.path.join(self.out_dir, f"loss_curve_{self.stage_name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            return path
        except ImportError:
            return None

    def save_all(self):
        json_path = self.save_json()
        png_path = self.save_plot()
        return json_path, png_path


def train_loop(model, data_source, max_steps, cfg, stage_name="train",
               start_step=0, optimizer=None, learning_rate=None, warmup_steps=None):
    log = logging.getLogger(__name__)
    lr = learning_rate or cfg.learning_rate
    warmup = warmup_steps if warmup_steps is not None else cfg.warmup_steps
    min_lr = cfg.min_lr
    tracker = LossTracker(cfg.out_dir, stage_name)

    if optimizer is None:
        decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": decay_params,   "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ], lr=lr, betas=(cfg.beta1, cfg.beta2), fused=True)

    ctx = autocast(device_type="cuda", dtype=getattr(torch, cfg.dtype))
    model.train()
    t0 = time.time()
    tokens_processed = 0
    toks_per_step = cfg.batch_size * cfg.gradient_accumulation_steps * model.config.block_size

    log.info(f"[{stage_name}] step {start_step} → {max_steps} | "
             f"eff_batch={cfg.batch_size * cfg.gradient_accumulation_steps} | "
             f"tok/step={toks_per_step:,} | lr={lr:.1e}")

    for step in range(start_step, max_steps):
        current_lr = get_lr(step, max_steps, lr, min_lr, warmup)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(cfg.gradient_accumulation_steps):
            x, y = data_source.get_batch()
            x, y = x.to(cfg.device), y.to(cfg.device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            loss_accum += loss.item()
            tokens_processed += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tok_s = tokens_processed / dt if dt > 0 else 0
            gpu_gb = torch.cuda.max_memory_allocated() / 1e9
            log.info(f"[{stage_name}] step {step:>6d} | loss {loss_accum:.4f} | "
                     f"lr {current_lr:.2e} | {tok_s:,.0f} tok/s | gpu {gpu_gb:.1f}GB")
            tracker.log(step, loss_accum, current_lr, tok_s)
            tokens_processed = 0
            t0 = time.time()

        if step > 0 and step % cfg.eval_interval == 0:
            try:
                enc = get_tokenizer()
                if stage_name == "base":
                    prompt = "The meaning of life is"
                else:
                    prompt = f"{SPECIAL_TOKENS['user_start']}What is the meaning of life?{SPECIAL_TOKENS['user_end']}{SPECIAL_TOKENS['assistant_start']}"
                prompt_t = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=cfg.device)
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                raw.eval()
                gen = raw.generate(prompt_t, max_new_tokens=150, temperature=0.8, top_k=200)
                raw.train()
                text = enc.decode(gen[0].tolist())
                log.info(f"  sample: {text[:400]}")
            except Exception as e:
                log.warning(f"  sample failed: {e}")
                model.train()

        if step > 0 and step % cfg.save_interval == 0:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{stage_name}_step{step}.pt")
            torch.save({
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": raw.config.__dict__,
                "step": step,
                "stage": stage_name,
            }, ckpt_path)
            log.info(f"  saved → {ckpt_path}")
            for old in sorted(Path(cfg.out_dir).glob(f"ckpt_{stage_name}_step*.pt")):
                old_step = int(old.stem.split("step")[1])
                if old_step != step and old_step % 25000 != 0:
                    old.unlink()

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = os.path.join(cfg.out_dir, f"ckpt_{stage_name}_final.pt")
    torch.save({
        "model": raw.state_dict(),
        "model_config": raw.config.__dict__,
        "step": max_steps,
        "stage": stage_name,
    }, final_path)
    log.info(f"[{stage_name}] done → {final_path}")

    # save loss curve
    json_path, png_path = tracker.save_all()
    log.info(f"  loss log → {json_path}")
    if png_path:
        log.info(f"  loss curve → {png_path}")

    return model, optimizer


# ---------------------------------------------------------------------------
# 6. PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(cfg, model_cfg, resume="", skip_base=False, skip_mid=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger(__name__)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # build model
    model = GPT(model_cfg).to(cfg.device)

    if cfg.compile_model and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # resume
    start_step = 0
    if resume and os.path.exists(resume):
        log.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=cfg.device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        log.info(f"Loaded step {start_step}, stage: {ckpt.get('stage', 'unknown')}")

    # ---- STAGE 1: BASE ----
    if not skip_base:
        log.info("=" * 60)
        log.info("STAGE 1: BASE PRETRAINING (FineWeb-Edu, streaming)")
        log.info("=" * 60)

        stream = StreamingTokenBuffer(
            block_size=model_cfg.block_size,
            batch_size=cfg.batch_size,
            buffer_size=2000,
        )
        model, optimizer = train_loop(
            model=model, data_source=stream, max_steps=cfg.max_steps,
            cfg=cfg, stage_name="base", start_step=start_step,
        )
    else:
        optimizer = None

    # ---- STAGE 2: MIDTRAINING ----
    if not skip_mid:
        log.info("=" * 60)
        log.info("STAGE 2: MIDTRAINING (conversation format)")
        log.info("=" * 60)

        conversations = build_chat_datasets()
        chat_buffer = ChatTokenBuffer(conversations, model_cfg.block_size, cfg.batch_size)

        mid_cfg = TrainConfig(
            batch_size=cfg.batch_size,
            gradient_accumulation_steps=16,
            max_steps=5000,              # more steps for larger model
            learning_rate=1e-4,
            min_lr=1e-5,
            warmup_steps=300,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
            device=cfg.device,
            dtype=cfg.dtype,
            compile_model=False,
            log_interval=25,
            eval_interval=500,
            save_interval=2500,
            out_dir=cfg.out_dir,
        )
        model, optimizer = train_loop(
            model=model, data_source=chat_buffer, max_steps=5000,
            cfg=mid_cfg, stage_name="mid", learning_rate=1e-4, warmup_steps=300,
        )

    # ---- STAGE 3: SFT ----
    log.info("=" * 60)
    log.info("STAGE 3: SFT (supervised fine-tuning)")
    log.info("=" * 60)

    if skip_mid:
        conversations = build_chat_datasets()

    sft_buffer = ChatTokenBuffer(conversations, model_cfg.block_size, cfg.batch_size)

    sft_cfg = TrainConfig(
        batch_size=cfg.batch_size,
        gradient_accumulation_steps=16,
        max_steps=2000,
        learning_rate=3e-5,
        min_lr=3e-6,
        warmup_steps=200,
        weight_decay=0.01,
        grad_clip=cfg.grad_clip,
        device=cfg.device,
        dtype=cfg.dtype,
        compile_model=False,
        log_interval=25,
        eval_interval=500,
        save_interval=1000,
        out_dir=cfg.out_dir,
    )
    model, _ = train_loop(
        model=model, data_source=sft_buffer, max_steps=2000,
        cfg=sft_cfg, stage_name="sft", learning_rate=3e-5, warmup_steps=200,
    )

    log.info("=" * 60)
    log.info("ALL STAGES COMPLETE!")
    log.info(f"Final model: {cfg.out_dir}/ckpt_sft_final.pt")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# 7. INTERACTIVE CHAT
# ---------------------------------------------------------------------------

def chat_interactive(checkpoint_path):
    enc = get_tokenizer()
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"])).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("\n" + "=" * 50)
    print("Chat with ZeroShot-350M! Type 'quit' to exit.")
    print("=" * 50 + "\n")

    history = ""
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not user_input:
            continue

        history += f"{SPECIAL_TOKENS['user_start']}{user_input}{SPECIAL_TOKENS['user_end']}"
        prompt = history + SPECIAL_TOKENS['assistant_start']
        tokens = enc.encode(prompt)
        if len(tokens) > 1900:
            tokens = tokens[-1900:]

        x = torch.tensor([tokens], dtype=torch.long, device="cuda")
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=400, temperature=0.7, top_k=100)

        response = enc.decode(y[0].tolist()[len(tokens):])
        for end_tok in [SPECIAL_TOKENS['assistant_end'], SPECIAL_TOKENS['user_start'], "<|endoftext|>"]:
            if end_tok in response:
                response = response[:response.index(end_tok)]
                break
        response = response.strip()
        print(f"AI: {response}\n")
        history += f"{SPECIAL_TOKENS['assistant_start']}{response}{SPECIAL_TOKENS['assistant_end']}"


# ---------------------------------------------------------------------------
# 8. GENERATE
# ---------------------------------------------------------------------------

def generate_text(ckpt_path, prompt="Once upon a time", max_tokens=300,
                  temperature=0.8, top_k=200):
    enc = get_tokenizer()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"])).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()
    x = torch.tensor([enc.encode(prompt)], dtype=torch.long, device="cuda")
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    print(enc.decode(y[0].tolist()))


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroShot-350M Training Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    tp = sub.add_parser("train", help="Full pipeline: base → mid → SFT")
    tp.add_argument("--resume", type=str, default="")
    tp.add_argument("--batch_size", type=int, default=8)
    tp.add_argument("--max_steps", type=int, default=30_000)
    tp.add_argument("--out_dir", type=str, default="checkpoints")
    tp.add_argument("--skip_base", action="store_true")
    tp.add_argument("--skip_mid", action="store_true")
    tp.add_argument("--no-compile", dest="compile", action="store_false", default=True)

    # chat
    cp = sub.add_parser("chat", help="Chat with trained model")
    cp.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_sft_final.pt")

    # generate
    gp = sub.add_parser("generate", help="Generate text")
    gp.add_argument("--checkpoint", type=str, required=True)
    gp.add_argument("--prompt", type=str, default="The meaning of life is")
    gp.add_argument("--max_tokens", type=int, default=300)
    gp.add_argument("--temperature", type=float, default=0.8)

    # finetune
    mp = sub.add_parser("finetune", help="Mid + SFT on existing base checkpoint")
    mp.add_argument("--checkpoint", type=str, required=True)
    mp.add_argument("--batch_size", type=int, default=8)
    mp.add_argument("--out_dir", type=str, default="checkpoints")
    mp.add_argument("--no-compile", dest="compile", action="store_false", default=True)

    # plot
    pp = sub.add_parser("plot", help="Generate loss curve from saved JSON logs")
    pp.add_argument("--out_dir", type=str, default="checkpoints")
    pp.add_argument("--combined", action="store_true", help="Combine all stages into one plot")

    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            out_dir=args.out_dir,
            compile_model=args.compile,
        )
        run_pipeline(cfg, ModelConfig(), resume=args.resume,
                     skip_base=args.skip_base, skip_mid=args.skip_mid)

    elif args.command == "chat":
        chat_interactive(args.checkpoint)

    elif args.command == "generate":
        generate_text(args.checkpoint, args.prompt, args.max_tokens, args.temperature)

    elif args.command == "finetune":
        cfg = TrainConfig(
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            compile_model=args.compile,
        )
        run_pipeline(cfg, ModelConfig(), resume=args.checkpoint, skip_base=True)

    elif args.command == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = args.out_dir
        json_files = sorted(Path(out_dir).glob("loss_*.json"))

        if not json_files:
            print(f"No loss JSON files found in {out_dir}")
            exit(1)

        if args.combined:
            # combined plot — all stages on one chart
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            fig.suptitle("ZeroShot-350M — Full Training Pipeline", fontsize=16, fontweight="bold")
            colors = {"base": "#2196F3", "mid": "#FF9800", "sft": "#4CAF50"}
            offset = 0

            for jf in json_files:
                with open(jf) as f:
                    data = json.load(f)
                stage = data["stage"]
                steps = [s + offset for s in data["steps"]]
                ax.plot(steps, data["losses"], color=colors.get(stage, "#999"),
                        linewidth=1.0, alpha=0.5)
                # smoothed
                if len(data["losses"]) > 10:
                    w = min(30, len(data["losses"]) // 5)
                    sm = np.convolve(data["losses"], np.ones(w) / w, mode="valid")
                    sm_steps = steps[w - 1:]
                    ax.plot(sm_steps, sm, color=colors.get(stage, "#999"),
                            linewidth=2.5, label=f"{stage} (smoothed)")
                if steps:
                    offset = steps[-1]

            ax.set_xlabel("Step (cumulative)", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            path = os.path.join(out_dir, "loss_curve_combined.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Combined plot saved → {path}")

        else:
            # individual plots per stage
            for jf in json_files:
                with open(jf) as f:
                    data = json.load(f)
                tracker = LossTracker(out_dir, data["stage"])
                tracker.steps = data["steps"]
                tracker.losses = data["losses"]
                tracker.lrs = data["learning_rates"]
                tracker.tok_per_sec = data["tokens_per_sec"]
                png = tracker.save_plot()
                if png:
                    print(f"Plot saved → {png}")
                else:
                    print(f"matplotlib not available, JSON saved at {jf}")
