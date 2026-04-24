import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from colorama import *
import os
import time

def afficher_caractere(carac):
    print(Fore.GREEN+Style.DIM+carac)
    time.sleep(0.2)
    os.system('clear' if os.name == 'posix' else 'cls')

def waitingAnimation(caracteres = ["...","⁕⁎.","⁎⁕⁎",".⁎⁕"]):
    print("\n"+Fore.GREEN+"————————————————————"+Fore.RESET)
    for caractere in caracteres:
        afficher_caractere(caractere)
    print(Fore.RESET)
# transformer_blocks.py

# ----------------------------
# Self-Attention Head
# ----------------------------
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

# ----------------------------
# Multi-Head Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# ----------------------------
# Feed Forward Network
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

with open("C:/Users/chevr/Desktop/FIchiers/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input= "C:/Users/chevr/Desktop/FIchiers/corpus.txt",
    model_prefix="tokenizer",
    vocab_size=95,     
    model_type="bpe"
)

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")
    
ids = sp.encode(text, out_type=int) 
data = torch.tensor(ids, dtype=torch.long) 

print(data)

vocab_size = sp.get_piece_size() 
print(vocab_size)

block_size = 24
embedding_dim = 128
n_heads = 16
n_layers = 16
lr = 1e-3 
epochs = 500


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y

class KIPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

model = KIPP()

if input("do you want to load KIPP (y/n)") != "n":
    try:
        model.load_state_dict(torch.load("kipp_model.pth"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found. Starting training from scratch.")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

if input("do you want to train KIPP (y/n)") != "n":
    epochs = int(input("Enter the number of epochs for training: "))
    for step in range(epochs):
        xb, yb = get_batch() 
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

print("KIPP is trained, now you can try it \n | tell him 'by' to close the tchating")

while True:
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")
    user = input(">>")
    if user == "by":
        break
    waitingAnimation()
    userInput = torch.tensor([sp.encode(user)], dtype=torch.long)
    out = model.generate(userInput, max_new_tokens=20)
    generated_ids = out[0].tolist()
    print(sp.decode(generated_ids))

if input("do you want to save KIPP (y/n)") != "n":
    while True:
        try:
            torch.save(model.state_dict(), "kipp_model.pth")
            print("Model saved")
            break
        except:
            print("an error occured, retrying...")
