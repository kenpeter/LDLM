# LDLM - Language Diffusion Learning Model

Reference: https://github.com/ML-GSAI/LLaDA

## Diffusion Language Model Flow Diagram

```
DIFFUSION LANGUAGE MODEL FLOW
==============================

1. DATA PREPROCESSING
   ┌─────────────────┐
   │ Raw Text Input  │
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │  SimpleTokenizer │ (build_vocab, encode)
   │  - <pad>, <unk>  │
   │  - <start>, <end>│
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │ TextDataset     │ (PyTorch Dataset)
   │ Token sequences │
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │   DataLoader    │ (batch_size=4)
   └─────────┬───────┘

2. MODEL ARCHITECTURE
             │
   ┌─────────▼───────┐
   │DiffusionTransformer│
   │                 │
   │ ┌─────────────┐ │
   │ │ Embedding   │ │ (vocab → d_model=256)
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Positional   │ │ (sin/cos encoding)
   │ │Encoding     │ │
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Time         │ │ (embed timestep t)
   │ │Embedding    │ │
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Transformer  │ │ (6 layers, 8 heads)
   │ │Encoder      │ │
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Output       │ │ (d_model → vocab_size)
   │ │Projection   │ │
   │ └─────────────┘ │
   └─────────┬───────┘

3. DIFFUSION PROCESS
             │
   ┌─────────▼───────┐
   │TextDiffusionModel│
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │ FORWARD PASS    │
   │ (Training)      │
   │                 │
   │ x₀ (clean text) │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Add Noise at │ │
   │ │timestep t   │ │ (cosine schedule)
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │  x_t        │ │ (noisy text)
   │ │(noisy text) │ │
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Predict noise│ │ (model output)
   │ │  ε_θ(x_t,t) │ │
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Cross-entropy│ │
   │ │    Loss     │ │
   │ └─────────────┘ │
   └─────────────────┘

4. REVERSE DIFFUSION (SAMPLING)
   ┌─────────────────┐
   │ x_T (pure noise)│
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │ For t = T→0:    │
   │                 │
   │ ┌─────────────┐ │
   │ │Predict noise│ │ 
   │ │ε_θ(x_t, t)  │ │
   │ └─────┬───────┘ │
   │       │         │
   │ ┌─────▼───────┐ │
   │ │Remove noise │ │
   │ │x_{t-1}=f(x_t│ │
   │ │    ,ε_θ,t)  │ │
   │ └─────┬───────┘ │
   │       │         │
   │       ▼         │
   │  (repeat until  │
   │   t reaches 0)  │
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │ x₀ (clean text) │
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │ Decode tokens   │ (tokenizer.decode)
   │ to readable text│
   └─────────────────┘

5. TRAINING LOOP
   ┌─────────────────┐
   │ 100 epochs      │
   │                 │
   │ For each batch: │
   │ 1. Sample t     │
   │ 2. Add noise    │
   │ 3. Predict ε    │
   │ 4. Compute loss │
   │ 5. Backprop     │
   │ 6. Update θ     │
   │                 │
   │ Every 10 epochs:│
   │ - Print loss    │
   │ - Generate      │
   │   sample text   │
   └─────────────────┘
```

## Key Components

### Architecture Details
- **Model**: DiffusionTransformer (train.py:133-188)
- **Dimensions**: 256 embedding, 8 attention heads, 6 layers
- **Vocabulary**: Dynamic vocab with <pad>, <unk>, <start>, <end> tokens
- **Sequence Length**: 128 tokens maximum
- **Time Steps**: 1000 diffusion steps with cosine schedule

### Training Process
- **Forward**: Add noise to clean text (train.py:228-247)
- **Reverse**: Learn to predict and remove noise (train.py:249-277)  
- **Loss**: Cross-entropy on noise prediction (train.py:293-309)
- **Optimizer**: Adam, lr=1e-4, 100 epochs
