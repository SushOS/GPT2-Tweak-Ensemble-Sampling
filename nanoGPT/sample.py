"""
Sample from an ensemble of two trained models by averaging their probability distributions.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT, ensemble_generate

# -----------------------------------------------------------------------------
# Model A (e.g., Grimms)
init_from = 'gpt2'  # A: 'resume' or a gpt2 variant like 'gpt2', 'gpt2-medium', etc.
out_dir = 'nanoGPT/out-grimm'       # A: directory containing ckpt.pt if init_from == 'resume'

# Model B (e.g., Critique)
init_from_b = 'gpt2'   # B: 'resume' or a gpt2 variant
out_dir_b = 'nanoGPT/out-critique'      # B: directory containing ckpt.pt if init_from_b == 'resume'

# Decoding & prompt
start = "\n" # or "<|endoftext|>" etc.; also supports "FILE:path.txt"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200

# Ensemble weights; you can provide multiple pairs to compare different mixtures
# e.g., [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5)]
weight_pairs = [(0.5, 0.5)]

# System / performance
seed = 1337
device = 'cpu' # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # or 'float32'
compile = False
exec(open('configurator.py').read()) # allows CLI overrides like out_dir=... out_dir_b=... weight_pairs=[(0.3,0.7)]
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
device_type = 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model(init_from_kind: str, out_dir_path: str):
    if init_from_kind == 'resume':
        ckpt_path = os.path.join(out_dir_path, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model, checkpoint
    elif init_from_kind.startswith('gpt2'):
        model = GPT.from_pretrained(init_from_kind, dict(dropout=0.0))
        return model, None
    else:
        raise ValueError(f"Unsupported init_from: {init_from_kind}")

# Build both models
model_a, checkpoint_a = load_model(init_from, out_dir)
model_b, checkpoint_b = load_model(init_from_b, out_dir_b)

model_a.eval().to(device)
model_b.eval().to(device)

if compile:
    model_a = torch.compile(model_a)
    model_b = torch.compile(model_b)

# look for the meta pickle in case it is available in the dataset folder (use A's dataset if present)
load_meta = False
if init_from == 'gpt2' and checkpoint_a is not None and 'config' in checkpoint_a and 'dataset' in checkpoint_a['config']:
    meta_path = os.path.join('data', checkpoint_a['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # assume GPT-2 BPE encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# sanity: vocab sizes should match for probability mixing
if model_a.config.vocab_size != model_b.config.vocab_size:
    raise ValueError(f"vocab_size mismatch: A={model_a.config.vocab_size}, B={model_b.config.vocab_size}")

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x0 = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run ensemble generation
with torch.no_grad():
    with ctx:
        for (wa, wb) in weight_pairs:
            # print header for this configuration
            print("=" * 80)
            print(f"Ensemble mixture: weight_a={wa:.2f}, weight_b={wb:.2f}, temperature={temperature}, top_k={top_k}")
            print("-" * 80)
            for k in range(num_samples):
                x = x0.clone()
                y = ensemble_generate(
                    model_a, model_b, x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    weight_a=wa,
                    weight_b=wb,
                )
                print(decode(y[0].tolist()))
                print('---------------')
