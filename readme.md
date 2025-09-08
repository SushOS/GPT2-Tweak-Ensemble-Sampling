# Text generation using Ensemble models
1. Model 1 finetuned on the Grimms dataset
2. Model 2 finetuned on the Critique's dataset

## Commands to Run this Repository
### To prepare the grimm dataset
```bash
cd path/to/grimm/dataset/directory
python prepare_grimm.py
```

### To prepare the the_critique dataset
```bash
cd path/to/the_critique/dataset/directory
python prepare_kant.py
```

### To finetune the model on the Grimm's dataset
```bash
python3 train_grimm.py config/finetune_grimm.py --device=mps --init_from=gpt2
```

### To finetune the model on the The Critique's dataset
```bash
python3 train_kant.py config/finetune_kant.py --device=mps --init_from=gpt2
```

### Final command to execute the sample and generating the output

```bash
python sample.py \
    --init_from=gpt2 \     
    --start="What is the answer to life, the universe, and everything?" \--num_samples=5 --max_new_tokens=100
```