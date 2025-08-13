# torch for tensor operation
import torch
# Neural network modules and building blocks
import torch.nn as nn
# Functional interface for neural network operations
import torch.nn.functional as F
# Optimization algorithms (Adam, SGD, etc.)
import torch.optim as optim
# Data loading utilities for training
from torch.utils.data import Dataset, DataLoader
# Numerical operations (not used but commonly imported)
import numpy as np
# Mathematical functions like sqrt, log, etc.
import math
# Type hints for better code documentation
from typing import Optional, Tuple
# Regular expressions for text processing
import re

# Dataset class to handle text data for PyTorch DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        # Store the list of text strings to be tokenized
        self.texts = texts
        # Tokenizer object to convert text to numerical tokens
        self.tokenizer = tokenizer
        # Maximum sequence length for padding/truncation
        self.max_length = max_length
    
    def __len__(self):
        # Return total number of text samples in dataset
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Get text at specified index
        text = self.texts[idx]
        # Convert text to tokens using tokenizer
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        # Return tokens as PyTorch tensor with long dtype for indexing
        return torch.tensor(tokens, dtype=torch.long)

# Simple tokenizer to convert text to numerical tokens and back
class SimpleTokenizer:
    def __init__(self):
        # Initialize vocabulary with special tokens and their indices
        self.vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        # Track current vocabulary size
        self.vocab_size = 4
        
    def build_vocab(self, texts):
        # Create set to store unique words from all texts
        words = set()
        # Extract all words from texts using regex
        for text in texts:
            # Find all word boundaries and convert to lowercase
            words.update(re.findall(r'\b\w+\b', text.lower()))
        
        # Add each unique word to vocabulary with unique index
        for word in sorted(words):
            if word not in self.vocab:
                # Assign current vocab_size as index for new word
                self.vocab[word] = self.vocab_size
                # Increment vocab size for next word
                self.vocab_size += 1
        
        # Create reverse mapping from index to word for decoding
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
    
    def encode(self, text, max_length=128):
        # Extract words from text using regex
        words = re.findall(r'\b\w+\b', text.lower())
        # Start with start token
        tokens = [self.vocab.get('<start>', 2)]
        
        # Convert each word to its token index
        for word in words:
            # Get word index, use <unk> if word not in vocabulary
            tokens.append(self.vocab.get(word, self.vocab.get('<unk>', 1)))
            # Stop if we reach max length minus 1 (save space for end token)
            if len(tokens) >= max_length - 1:
                break
        
        # Add end token
        tokens.append(self.vocab.get('<end>', 3))
        
        # Pad sequence to max_length with padding tokens
        while len(tokens) < max_length:
            tokens.append(self.vocab.get('<pad>', 0))
            
        # Ensure we don't exceed max_length
        return tokens[:max_length]
    
    def decode(self, tokens):
        # Convert token indices back to words
        words = []
        for token in tokens:
            # Get word from token index, use <unk> if not found
            word = self.idx2word.get(token, '<unk>')
            # Skip special tokens when building output text
            if word not in ['<pad>', '<start>', '<end>']:
                words.append(word)
        # Join words with spaces to form readable text
        return ' '.join(words)

# Positional encoding to give transformer information about token positions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create tensor to store positional encodings
        pe = torch.zeros(max_len, d_model)
        # Create position indices [0, 1, 2, ...] and add dimension
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create scaling factors for different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices in embedding dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in embedding dimension
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it moves with model but isn't a parameter
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]

# Main transformer model for diffusion process
class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=128):
        super().__init__()
        # Store model dimensions
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding layer to convert token indices to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding to add position information
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Create transformer encoder layer with attention mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # embedding dimension
            nhead=nhead,      # number of attention heads
            dim_feedforward=d_model * 4,  # feedforward network size
            dropout=0.1,      # dropout for regularization
            batch_first=True  # batch dimension comes first
        )
        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Neural network to embed time step information
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),      # expand time scalar to d_model dims
            nn.SiLU(),                  # smooth activation function
            nn.Linear(d_model, d_model) # transform to final embedding
        )
        
        # Final layer to predict vocabulary probabilities
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, t):
        # Get input dimensions
        batch_size, seq_len = x.shape
        
        # Convert tokens to embeddings and scale by sqrt(d_model)
        x_emb = self.embedding(x) * math.sqrt(self.d_model)
        # Add positional information
        x_emb = self.pos_encoding(x_emb)
        
        # Embed time step and convert to proper shape
        t_emb = self.time_embedding(t.float().unsqueeze(-1))
        # Expand time embedding to match sequence length
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add time information to token embeddings
        x_emb = x_emb + t_emb
        
        # Pass through transformer layers
        output = self.transformer(x_emb)
        # Project to vocabulary size for token prediction
        logits = self.output_projection(output)
        
        return logits

# Main diffusion model that handles forward and reverse processes
class TextDiffusionModel:
    def __init__(self, vocab_size, device='cpu', max_timesteps=1000):
        # Store device (CPU or GPU) for tensor operations
        self.device = device
        # Number of diffusion time steps
        self.max_timesteps = max_timesteps
        # Size of vocabulary for token generation
        self.vocab_size = vocab_size
        
        # Create and move transformer model to device
        self.model = DiffusionTransformer(vocab_size).to(device)
        
        # Create noise schedule using cosine schedule
        self.betas = self._cosine_beta_schedule(max_timesteps)
        # Alpha values are 1 - beta (how much original signal to keep)
        self.alphas = 1.0 - self.betas
        # Cumulative product of alphas (total signal remaining)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute square roots for efficient sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        # Create cosine-based noise schedule for smoother diffusion
        steps = timesteps + 1
        # Create linear space from 0 to timesteps
        x = torch.linspace(0, timesteps, steps)
        # Apply cosine function to create smooth schedule
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        # Normalize so first value is 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Calculate beta values from consecutive alpha ratios
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clip to prevent numerical instabilities
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_diffusion(self, x0, t):
        # Generate random noise tokens with same shape as input
        noise = torch.randint(0, self.vocab_size, x0.shape, device=self.device)
        
        # Get noise schedule values for current time steps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # Probability of replacing with noise (increases with time)
        noise_prob = sqrt_one_minus_alphas_cumprod_t
        # Probability of keeping original (decreases with time)
        original_prob = sqrt_alphas_cumprod_t
        
        # Create random mask to decide which tokens to noise
        mask = torch.rand_like(x0.float()) < noise_prob
        # Apply noise where mask is True, keep original where False
        xt = torch.where(mask, noise, x0)
        
        # Return noised input and the noise that was added
        return xt, noise
    
    def reverse_diffusion_step(self, xt, t):
        # Perform one step of reverse diffusion (denoising)
        with torch.no_grad():
            # Predict noise using the trained model
            pred_noise_logits = self.model(xt, t)
            # Get most likely noise tokens (not used in this implementation)
            pred_noise = torch.argmax(pred_noise_logits, dim=-1)
            
            # If not at final step, apply denoising
            if t[0] > 0:
                # Get schedule parameters for current time step
                beta_t = self.betas[t].view(-1, 1)
                alpha_t = self.alphas[t].view(-1, 1)
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
                
                # Calculate how much noise to remove
                noise_factor = beta_t / sqrt_one_minus_alphas_cumprod_t
                
                # Probabilistically remove some noise
                mask = torch.rand_like(xt.float()) < noise_factor
                # Replace noisy tokens with random tokens (simplified denoising)
                x_prev = torch.where(mask, 
                                   torch.randint(0, self.vocab_size, xt.shape, device=self.device),
                                   xt)
            else:
                # At final step, return as-is
                x_prev = xt
            
            return x_prev
    
    def sample(self, shape, tokenizer):
        # Start with pure noise (random tokens)
        x = torch.randint(0, self.vocab_size, shape, device=self.device)
        
        # Iteratively denoise from max timesteps down to 0
        for i in reversed(range(self.max_timesteps)):
            # Create time tensor for current step
            t = torch.full((shape[0],), i, device=self.device)
            # Apply one denoising step
            x = self.reverse_diffusion_step(x, t)
        
        # Return final denoised sequence
        return x
    
    def compute_loss(self, x0):
        # Get batch size from input
        batch_size = x0.shape[0]
        # Sample random time steps for each batch item
        t = torch.randint(0, self.max_timesteps, (batch_size,), device=self.device)
        
        # Apply forward diffusion to get noisy input and target noise
        xt, noise = self.forward_diffusion(x0, t)
        
        # Predict noise using model
        pred_noise_logits = self.model(xt, t)
        
        # Calculate cross-entropy loss between predicted and actual noise
        loss = F.cross_entropy(pred_noise_logits.view(-1, self.vocab_size), 
                              noise.view(-1))
        
        return loss

# Main training function
def train_model():
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Sample training texts for the model to learn from
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming the world",
        "machine learning models can generate creative text",
        "diffusion models work by gradually adding and removing noise",
        "natural language processing enables computers to understand text",
        "deep learning has revolutionized artificial intelligence",
        "transformer architectures have changed how we process language",
        "generative models can create new and original content",
        "neural networks learn patterns from large datasets",
        "language models can complete sentences and generate stories"
    ]
    
    # Create tokenizer and build vocabulary from training texts
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader for batch processing
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize diffusion model with vocabulary size
    diffusion_model = TextDiffusionModel(tokenizer.vocab_size, device=device)
    # Use Adam optimizer with learning rate 1e-4
    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=1e-4)
    
    # Set number of training epochs
    num_epochs = 100
    
    print("Starting training...")
    # Main training loop over epochs
    for epoch in range(num_epochs):
        # Initialize loss tracking for this epoch
        total_loss = 0
        num_batches = 0
        
        # Process each batch of data
        for batch in dataloader:
            # Move batch to GPU/CPU
            batch = batch.to(device)
            
            # Clear gradients from previous step
            optimizer.zero_grad()
            # Calculate loss using diffusion model
            loss = diffusion_model.compute_loss(batch)
            # Backpropagate gradients
            loss.backward()
            # Update model parameters
            optimizer.step()
            
            # Track loss for averaging
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        
        # Every 10 epochs, print progress and generate sample
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Generate sample text to see training progress
            sample_shape = (1, 128)  # 1 sequence of length 128
            generated = diffusion_model.sample(sample_shape, tokenizer)
            # Convert tokens back to readable text
            generated_text = tokenizer.decode(generated[0].cpu().numpy())
            print(f"Generated: {generated_text}")
            print("-" * 50)
    
    print("Training completed!")
    
    # Generate final samples to showcase trained model
    print("\nGenerating final samples...")
    for i in range(3):
        # Generate sequence of length 128
        sample_shape = (1, 128)
        # Sample from trained diffusion model
        generated = diffusion_model.sample(sample_shape, tokenizer)
        # Convert to readable text
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        print(f"Sample {i+1}: {generated_text}")

# Run training when script is executed directly
if __name__ == "__main__":
    train_model()