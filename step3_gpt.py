# Step 3: The full (tiny) GPT
#   multi-head attention + feed-forward MLP + residual connections + LayerNorm + dropout,
#   stacked into N Transformer blocks.


import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- config (sized for an 8 GB Mac) ----------------
batch_size = 32
block_size = 64      # context length
n_embd = 128         # vector size per token
n_head = 4           # attention heads per block (head_size = 128 / 4 = 32)
n_layer = 4          # number of Transformer blocks stacked
dropout = 0.1        # randomly zero 10% of activations during training to fight overfitting
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)
print(f"Using device: {device}")

# ---------------- data + tokenizer (same as before) ----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()     # turns dropout OFF for evaluation
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

# ---------------- building blocks ----------------
class Head(nn.Module):
    """One head of masked self-attention (same as step 2, plus dropout)."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """
    NEW: several heads in parallel, each with its own q/k/v.
    One head might track 'which speaker is talking', another 'am I mid-word'.
    Their outputs are concatenated, then mixed back together by a linear projection.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """
    NEW: attention = tokens COMMUNICATE; feed-forward = each token THINKS about what it gathered.
    Expand 4x, apply a nonlinearity, project back down. Applied to every token independently.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    NEW: one Transformer block = communicate, then think.
    Two tricks make deep stacks trainable:
      - Residual connections (x + ...): each sublayer only ADDS a refinement,
        so gradients flow straight through the '+' back to early layers.
      - LayerNorm (pre-norm): normalizes each token's vector before each sublayer,
        keeping activations in a stable range.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # communicate
        x = x + self.ffwd(self.ln2(x))   # think
        return x

# ---------------- the GPT ----------------
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])  # NEW: stack N blocks
        self.ln_f = nn.LayerNorm(n_embd)                                              # NEW: final norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------- training ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start = time.time()
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        print(f"step {it}: train loss {l['train']:.4f}, val loss {l['val']:.4f}  ({time.time() - start:.0f}s)")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save the trained weights so you can reuse them without retraining
torch.save(model.state_dict(), 'gpt_step3.pt')
print("Saved weights to gpt_step3.pt")

# ---------------- sample ----------------
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Sample AFTER training ---")
print(decode(model.generate(context, 800)[0].tolist()))