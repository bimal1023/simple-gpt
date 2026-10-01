# Step 1: Data, tokenizer, and a bigram baseline model
# Run: python step1_bigram.py   (needs input.txt in the same folder)

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- config ----------------
batch_size = 32      # how many sequences we train on in parallel
block_size = 8       # context length (max chars the model sees)
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Apple GPU if available
torch.manual_seed(1337)
print(f"Using device: {device}")

# ---------------- data + tokenizer ----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))          # every unique character = our vocabulary
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # char -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

print(f"Dataset: {len(text):,} chars, vocab size: {vocab_size}")
print("encode('hello') ->", encode("hello"))

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))                 # 90% train, 10% validation
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    """Grab random chunks; y is x shifted by one (the 'next char' targets)."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------- model ----------------
class BigramLanguageModel(nn.Module):
    """Predicts the next char using ONLY the current char (no context yet)."""
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly looks up a row of logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)   # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                 # last time step only
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)   # append and repeat
        return idx

model = BigramLanguageModel(vocab_size).to(device)

# ---------------- before training ----------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Sample BEFORE training ---")
print(decode(model.generate(context, 200)[0].tolist()))

# ---------------- training ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for it in range(max_iters):
    if it % eval_interval == 0:
        l = estimate_loss()
        print(f"step {it}: train loss {l['train']:.4f}, val loss {l['val']:.4f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------- after training ----------------
print("\n--- Sample AFTER training ---")
print(decode(model.generate(context, 500)[0].tolist()))
