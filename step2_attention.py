
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 32      # context length
n_embd = 64          # size of each token's vector
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3 
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)
print(f"Using device: {device}")

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

#  NEW: one head of self-attention 
class Head(nn.Module):
    """
    Every token emits three vectors:
      query = "what am I looking for?"
      key   = "what do I contain?"
      value = "what I'll share if you attend to me"
    Affinity between tokens = query . key. A causal mask hides the future.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular matrix of ones: position t may only see positions <= t
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        # 1) how much should each token attend to each other token?
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T), scaled so softmax stays soft
        # 2) no peeking at the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 3) turn scores into weights that sum to 1 across each row
        wei = F.softmax(wei, dim=-1)
        # 4) weighted average of the values
        v = self.value(x)    # (B, T, head_size)
        return wei @ v       # (B, T, head_size)

# model 
class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # WHAT the char is
        self.position_embedding_table = nn.Embedding(block_size, n_embd)   # NEW: WHERE it sits
        self.sa_head = Head(n_embd)                                        # NEW: talk to past tokens
        self.lm_head = nn.Linear(n_embd, vocab_size)                       # vectors -> scores per char

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                 # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # (T, n_embd)
        x = tok_emb + pos_emb        # broadcast add: each token now knows what AND where it is
        x = self.sa_head(x)          # each token gathers context from earlier tokens
        logits = self.lm_head(x)     # (B, T, vocab_size)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]   # NEW: crop, since position table only has block_size rows
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = AttentionLanguageModel().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# training 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        print(f"step {it}: train loss {l['train']:.4f}, val loss {l['val']:.4f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# sample
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Sample AFTER training ---")
print(decode(model.generate(context, 500)[0].tolist()))