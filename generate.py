
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.8,
                    help='<1 = safer/more repetitive, >1 = wilder/more random')
parser.add_argument('--top_k', type=int, default=20,
                    help='only sample from the k most likely next chars (0 = no limit)')
parser.add_argument('--tokens', type=int, default=300, help='how many chars to generate')
args = parser.parse_args()


block_size = 64
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1   # ignored at generation time (model.eval() turns dropout off)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------------- rebuild the same vocabulary ----------------
# The model only knows integers 0..64, so we rebuild the exact same char<->int mapping.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]   # silently skip chars the model never saw
decode = lambda l: ''.join(itos[i] for i in l)

# ---------------- same model classes as step 3 ----------------
class Head(nn.Module):
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :]          # scores for the next char

            # NEW: temperature. Dividing scores by T<1 sharpens the distribution
            # (confident, repetitive); T>1 flattens it (creative, chaotic).
            logits = logits / temperature

            # NEW: top-k. Keep only the k highest-scoring chars, so the model
            # never picks a really unlikely one by bad luck.
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # stream each char to the screen as it is generated
            print(itos[idx_next.item()], end='', flush=True)
        return idx

# ---------------- load the trained weights ----------------
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('gpt_step3.pt', map_location=device))
model.eval()
print(f"Loaded gpt_step3.pt on {device} | temperature={args.temperature}, top_k={args.top_k}")
print("Type a prompt (e.g. 'ROMEO:' or 'To be or not'). Empty line = start from scratch. Ctrl+C to quit.\n")

# ---------------- interactive loop ----------------
try:
    while True:
        prompt = input(">>> ")
        ids = encode(prompt) if prompt else encode('\n')
        context = torch.tensor([ids], dtype=torch.long, device=device)
        print(prompt, end='')
        model.generate(context, args.tokens, args.temperature, args.top_k)
        print("\n")
except (KeyboardInterrupt, EOFError):
    print("\nBye!")