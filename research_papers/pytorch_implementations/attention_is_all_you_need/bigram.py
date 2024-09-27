import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
BATCH_SIZE = 32  # Number of independent sequences to process in parallel
BLOCK_SIZE = 8  # Maximum context length for predictions
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200

torch.manual_seed(1337)

# Load the input text data
with open('text_data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract unique characters from the text
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# Create character-to-integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # Encoder: string to integer list
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: integer list to string

# Train and validation split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Function to get a batch of data for training and validation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# Function to estimate the loss on train and validation sets
@torch.no_grad()
def estimate_loss():
    loss_dict = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        loss_dict[split] = losses.mean()
    model.train()
    return loss_dict


# Bigram Language Model class
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Focus on the last time step (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append new token
        return idx


# Initialize the model and optimizer
model = BigramLanguageModel(VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for iteration in range(MAX_ITERS):

    # Evaluate loss every EVAL_INTERVAL steps
    if iteration % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iteration}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass and loss calculation
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate new text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
