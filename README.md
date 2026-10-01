# simple-gpt

A small character-level GPT built from scratch in PyTorch and trained on Tiny Shakespeare. The model is built up in steps, from a bigram baseline to a full Transformer, so each part of the architecture can be seen working on its own.

## Architecture

```
Input text            "To be or not t"
     |
Tokenizer             characters -> integers (65-char vocabulary)
     |
Embeddings            token embedding (what the char is)
                    + position embedding (where it sits)
     |
Transformer block x N
  |  LayerNorm -> Masked multi-head self-attention -> + residual
  |  LayerNorm -> Feed-forward MLP                -> + residual
     |
Final LayerNorm
     |
Linear head           scores (logits) for every possible next char
     |
Softmax + sample      next char -> appended to input, repeat
```

How it works:

- **Self-attention** lets each character look back at earlier characters in its context window. Each token produces a query, a key, and a value. The query-key scores decide how much each token attends to the others, and a causal mask prevents looking at future tokens.
- **Multi-head attention** runs several attention heads in parallel, so different heads can learn different patterns.
- **The feed-forward MLP** processes each token independently after attention has gathered context.
- **Residual connections and LayerNorm** keep training stable as blocks are stacked.
- **Training** minimizes cross-entropy loss between the predicted and actual next character.

## Files

| File | What it adds | Val loss |
|---|---|---|
| `step1_bigram.py` | Tokenizer, batching, bigram model (no context) | 2.49 |
| `step2_attention.py` | Position embeddings, single-head self-attention | 2.34 |
| `step3_gpt.py` | Multi-head attention, MLP, residuals, LayerNorm, 4 stacked blocks (816K params) | 1.62 |
| `generate.py` | Loads trained weights; prompt-based generation with temperature and top-k | - |

Step 3 trains in about 2 minutes on an Apple Silicon Mac (MPS).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch numpy
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

```bash
python step3_gpt.py                                 # train and save gpt_step3.pt
python generate.py                                  # interactive prompt
python generate.py --temperature 0.5 --top_k 10     # safer, more repetitive
python generate.py --temperature 1.2 --top_k 0      # more creative
```

## Sample output (step 3)

```
KING RICHARD III:
Yough not, and honour madam, is a your bosom.
Again Such our corres back shall shephecas?
```

## Reference

Based on Andrej Karpathy's "Let's build GPT" and nanoGPT.