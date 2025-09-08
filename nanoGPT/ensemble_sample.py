import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ensure we can import the NanoGPT modules
# assumes this file is placed at the repo root alongside model.py, config.py, etc.
from model import GPT, GPTConfig

# tiktoken for GPT-2 BPE encoding
try:
    import tiktoken
except ImportError:
    print("Please: pip install tiktoken")
    sys.exit(1)


def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # Support both NanoGPT-style checkpoints and raw state_dicts
    if isinstance(ckpt, dict) and 'model_args' in ckpt and 'model' in ckpt:
        model_args = ckpt['model_args']
        state_dict = ckpt['model']
    else:
        # Fallback: assume raw state dict; try to infer minimal model args
        model_args = None
        state_dict = ckpt
    return model_args, state_dict


def build_model_from_ckpt(ckpt_path, device, dtype):
    model_args, state_dict = load_checkpoint(ckpt_path, map_location='cpu')

    # Try to infer GPTConfig from checkpoint; minimal fallback if not present
    if model_args is None:
        # Heuristic defaults (adjust to match your training config if needed)
        # You can also parse config from filename or include a sidecar JSON
        raise ValueError(
            f"Checkpoint {ckpt_path} does not include 'model_args'. "
            "Please checkpoint with NanoGPT format or provide a compatible config."
        )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Some checkpoints may have keys with 'module.' prefix (if using DDP/DataParallel)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading {ckpt_path}: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading {ckpt_path}: {unexpected}")

    # cast and move
    if dtype == 'float16':
        model = model.half()
    elif dtype == 'bfloat16':
        model = model.bfloat16()
    elif dtype == 'float32':
        pass
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    model.eval().to(device)
    return model, gptconf


@torch.no_grad()
def ensemble_generate(
    model_a, model_b, weight_a, weight_b, idx, max_new_tokens,
    temperature=1.0, top_k=0, device='cpu', dtype='float16', mix_mode='probs', seed=None
):
    """
    Autoregressive generation with ensemble logits from two models.
    mix_mode='probs': p = w_a*softmax(logits_a/T) + w_b*softmax(logits_b/T)
    mix_mode='logits': logits = w_a*logits_a/T + w_b*logits_b/T ; p = softmax(logits)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    idx = idx.to(device)

    for _ in range(max_new_tokens):
        # Forward both models on the full context (no kv-cache in NanoGPT by default)
        logits_a, _ = model_a(idx)
        logits_b, _ = model_b(idx)
        logits_a = logits_a[:, -1, :]  # last token
        logits_b = logits_b[:, -1, :]

        if temperature <= 0:
            # Greedy: ignore temperature and top-k, combine on raw logits then argmax
            # For greedy, mixing logits is sufficient
            mixed_logits = weight_a * logits_a + weight_b * logits_b
            next_token = torch.argmax(mixed_logits, dim=-1, keepdim=True)
        else:
            if mix_mode == 'probs':
                # mix in probability space (recommended)
                pa = F.softmax(logits_a / temperature, dim=-1)
                pb = F.softmax(logits_b / temperature, dim=-1)
                p = weight_a * pa + weight_b * pb

                if top_k > 0:
                    # zero out everything outside top_k, then renormalize
                    topk_vals, topk_idx = torch.topk(p, k=top_k, dim=-1)
                    mask = torch.zeros_like(p).scatter_(1, topk_idx, 1.0)
                    p = p * mask
                    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)

                # sample
                next_token = torch.multinomial(p, num_samples=1)

            elif mix_mode == 'logits':
                # mix in logit space then apply temperature
                mixed_logits = weight_a * logits_a + weight_b * logits_b
                mixed_logits = mixed_logits / temperature
                if top_k > 0:
                    # top-k on logits
                    v, ix = torch.topk(mixed_logits, top_k, dim=-1)
                    min_keep = v[:, [-1]]
                    mixed_logits = torch.where(
                        mixed_logits < min_keep, torch.full_like(mixed_logits, float('-inf')), mixed_logits
                    )
                p = F.softmax(mixed_logits, dim=-1)
                next_token = torch.multinomial(p, num_samples=1)
            else:
                raise ValueError(f"Unknown mix_mode: {mix_mode}")

        idx = torch.cat((idx, next_token), dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser(description="NanoGPT Ensemble Sampling (Two Checkpoints)")
    parser.add_argument('--ckpt_a', type=str, required=True, help='Path to checkpoint A (e.g., Grimms)')
    parser.add_argument('--ckpt_b', type=str, required=True, help='Path to checkpoint B (e.g., Critique)')
    parser.add_argument('--start', type=str, default="\n", help='Prompt string')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples per configuration')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--mix_mode', type=str, default='probs', choices=['probs', 'logits'],
                        help='Where to mix the models: probability space or logit space')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    device = args.device

    # Build models
    model_a, conf_a = build_model_from_ckpt(args.ckpt_a, device, args.dtype)
    model_b, conf_b = build_model_from_ckpt(args.ckpt_b, device, args.dtype)

    # Basic compatibility checks
    if conf_a.vocab_size != conf_b.vocab_size:
        print(f"[WARN] vocab_size mismatch: A={conf_a.vocab_size} B={conf_b.vocab_size}")
    if conf_a.block_size != conf_b.block_size:
        print(f"[WARN] block_size mismatch: A={conf_a.block_size} B={conf_b.block_size}. "
              f"Generation will be limited by the smaller block size.")

    # Tokenizer (GPT-2 BPE)
    enc = tiktoken.get_encoding("gpt2")
    def encode(s): return torch.tensor([enc.encode(s)], dtype=torch.long)
    def decode(t): return enc.decode(t.tolist())

    # Encode prompt
    start_ids = encode(args.start)
    if start_ids.size(1) >= min(conf_a.block_size, conf_b.block_size):
        print(f"[WARN] Prompt length {start_ids.size(1)} >= model block size; trimming prompt.")
        start_ids = start_ids[:, -min(conf_a.block_size, conf_b.block_size):]

    start_ids = start_ids.to(device)

    # Requested configurations
    configs = [
        ("30% A + 70% B", 0.30, 0.70),
        ("70% A + 30% B", 0.70, 0.30),
        ("50% A + 50% B", 0.50, 0.50),
    ]

    for label, w_a, w_b in configs:
        print("=" * 80)
        print(f"Ensemble configuration: {label}  (mix_mode={args.mix_mode}, temperature={args.temperature}, top_k={args.top_k})")
        print("-" * 80)

        for s in range(args.num_samples):
            # Re-encode the prompt for each sample to avoid context growth across samples
            idx = start_ids.clone()

            out = ensemble_generate(
                model_a, model_b,
                weight_a=w_a, weight_b=w_b,
                idx=idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                dtype=args.dtype,
                mix_mode=args.mix_mode,
                seed=args.seed,
            )
            # Slice only the newly generated tokens to show, or show full
            generated = out[0, start_ids.size(1):]
            text = decode(generated)
            print(f"[Sample {s+1}] {text}\n")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
