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





 Training Loop Order:

  COMPLETELY RANDOM - no small to big or big to small pattern!

  t = torch.randint(0, 1000, (batch_size,))  # Pure randomness each step

  Full Example with Corruption Details:

  Training Step 1:
  sentence_0: "the quick brown fox" 
  Original tokens: [2, 15, 23, 8, 3]  # <start>, the, quick, brown, fox, <end>
  Random t=234 → noise_prob ≈ 0.23 (23% corruption chance)

  Random mask generation:
  Position:  [0,    1,    2,     3,     4   ]
  Random:    [0.45, 0.12, 0.67,  0.89,  0.05]  # Random numbers 0-1
  Mask:      [False,True, False, False, True ]  # < 0.23 threshold
  Noise:     [67,   91,   44,    52,    12  ]  # Random vocab tokens

  Result: [2, 91, 23, 8, 12]  # Corrupted: "the [noise] quick brown [noise]"
  Target: [67, 91, 44, 52, 12] # What model should predict as noise

  Training Step 2:
  sentence_0: "the quick brown fox" (SAME sentence)
  Original tokens: [2, 15, 23, 8, 3]  # Same original
  Random t=891 → noise_prob ≈ 0.89 (89% corruption chance)

  Random mask generation:
  Position:  [0,   1,    2,    3,    4   ]
  Random:    [0.95, 0.82, 0.45, 0.91, 0.77]  # NEW random numbers
  Mask:      [False,True, True, False, True ]  # < 0.89 threshold
  Noise:     [34,   72,   19,   88,    41  ]  # NEW random noise

  Result: [2, 72, 19, 8, 41]  # More corrupted: "the [noise] [noise] brown [noise]"
  Target: [34, 72, 19, 88, 41] # NEW noise target

  Training Step 3:
  sentence_0: "the quick brown fox" (SAME sentence again)
  Original tokens: [2, 15, 23, 8, 3]  # Same original
  Random t=67 → noise_prob ≈ 0.07 (7% corruption chance)

  Random mask generation:
  Position:  [0,    1,    2,    3,    4   ]
  Random:    [0.12, 0.95, 0.03, 0.45, 0.89]  # NEW random numbers
  Mask:      [False,False,True, False,False]  # < 0.07 threshold
  Noise:     [55,   29,   77,   66,    84  ]  # NEW random noise

  Result: [2, 15, 77, 8, 3]  # Lightly corrupted: "the quick [noise] brown fox"
  Target: [55, 29, 77, 66, 84] # NEW noise target

  Key Insights:
  1. Random timesteps: t=234, t=891, t=67 (no pattern)
  2. Same sentence: Gets different corruption each time
  3. Random masks: Even same timestep would give different results
  4. Training target: Always the noise tokens that were added

  The model learns: "Given any corruption level + noisy text → predict what noise was added"