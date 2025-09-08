# prepare_grimm.py
import os
import tiktoken
import numpy as np

# Paths
here = os.path.dirname(__file__)
input_file = os.path.join(here, "Grimm's_fairy_tales.txt")  # ensure this file is in the same directory

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Could not find {input_file}. Place 'Grimm-s_fairy_tales.txt' in the same folder as this script."
    )

# Read entire corpus
with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

# 90/10 split by character count (contiguous split, like the reference script)
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Tokenize with GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"[Grimm] train has {len(train_ids):,} tokens")
print(f"[Grimm] val has {len(val_ids):,} tokens")

# Save to bin files (uint16 is safe as GPT-2 vocab size is 50257 < 65535)
train_arr = np.array(train_ids, dtype=np.uint16)
val_arr = np.array(val_ids, dtype=np.uint16)

train_out = os.path.join(here, "train_grimm.bin")
val_out = os.path.join(here, "val_grimm.bin")

train_arr.tofile(train_out)
val_arr.tofile(val_out)

print(f"[Grimm] wrote {train_out} and {val_out}")
