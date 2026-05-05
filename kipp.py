import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from colorama import *
import os
import time
import datetime
path = "C:/Users/chevr/Desktop/FIchiers/KIPP-AI/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def afficher_caractere(carac):
    print(Fore.GREEN+Style.DIM+carac)
    time.sleep(0.2)
    os.system('clear' if os.name == 'posix' else 'cls')

def waitingAnimation(caracteres = ["...","⁕⁎.","⁎⁕⁎",".⁎⁕"]):
    for caractere in caracteres:
        afficher_caractere(caractere)
    print(Fore.RESET)

def WriteInLog(data,path=path):
    data = str(data)
    with open(path+"log.txt", "r", encoding="utf-8") as prevlog:
        prev = prevlog.read()
    with open(path+"log.txt", "w", encoding="utf-8") as newlog:
        newlog.write(prev+"\n"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" : "+data)

def show(text):
    os.system('clear' if os.name == 'posix' else 'cls')
    print(text)
    WriteInLog(text)



# ----------------------------
# Self-Attention Head
# ----------------------------
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False).to(device)
        self.query = nn.Linear(embedding_dim, head_size, bias=False).to(device)
        self.value = nn.Linear(embedding_dim, head_size, bias=False).to(device)
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
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)]).to(device)
        self.proj = nn.Linear(num_heads * head_size, embedding_dim).to(device)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1).to(device)
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
        ).to(device)
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads).to(device)
        self.ffwd = FeedForward(embedding_dim).to(device)
        self.ln1 = nn.LayerNorm(embedding_dim).to(device)
        self.ln2 = nn.LayerNorm(embedding_dim).to(device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

show(Fore.GREEN+Style.BRIGHT+"Initializing KIPP-AI..."+Fore.RESET)
show("\n\n<<<<<<<<<< Launching KIPP-AI >>>>>>>>>>\nTorch version:"+str(torch.__version__)+"\n"+"CUDA available:"+str( torch.cuda.is_available())+"\n"+"GPU name:"+str( torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")+"\n\n")
show(Back.GREEN+"\n"+" "*39+"\n ██ ▄█▀ ██ █████▄ █████▄     ▄████▄ ██ \n ████   ██ ██▄▄█▀ ██▄▄█▀ ▄▄▄ ██▄▄██ ██ \n ██ ▀█▄ ██ ██     ██         ██  ██ ██ \n"+" "*39+Back.RESET+"\n\n")
time.sleep(2)

with open(path+"corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input= path+"corpus.txt",
    model_prefix="tokenizer",
    vocab_size=95,     
    model_type="bpe"
)

sp = spm.SentencePieceProcessor()
sp.load(path+"tokenizer.model")
    
ids = sp.encode(text, out_type=int) 
data = torch.tensor(ids, dtype=torch.long).to(device)

show("tensor: " + str(data))

vocab_size = sp.get_piece_size() 
show("vocab_size: " + str(vocab_size))

epochs = 500
block_size    = 64       # [OPT 7] 24→64 : plus de contexte = meilleure cohérence
embedding_dim = 256      # [OPT 8] 128→256 : modèle plus expressif
n_heads       = 8        # head_size = 32 (256/8), plus efficace que 128/16=8
n_layers      = 6        # 6 blocs bien dimensionnés > 16 blocs sous-dimensionnés
lr            = 1e-3     # [OPT 9] 1e-3→3e-4 : meilleur pour la convergence
batch_size    = 32       # [OPT 10] 16→32 : meilleur gradient, mieux pour GPU
vocab_size    = 512 


class KIPP(nn.Module):
    def __init__(self,device=device):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 
        self.device = device
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

        show(f"<<<<<<<<<< START OF LOG ({datetime.datetime.now().strftime('%Y-%m-%d')}) >>>>>>>>>>")

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
            probs = F.softmax(logits, dim=-1).to(device)
            next_idx = torch.multinomial(probs, 1).to(device)
            idx = torch.cat((idx, next_idx), dim=1).to(device)
        return idx

    def save(self, path=path+"kipp_model.pth"):
        try:
            torch.save(self.state_dict(), path).to(device)
            show("Model saved successfully.")
        except Exception as e:
            show(f"an error occured : {e}")

    def load(self, path=path+"kipp_model.pth"):
        try:
            self.load_state_dict(torch.load(path))
            show("Model loaded successfully.")
        except FileNotFoundError:
            show("No pre-trained model found. Starting training from scratch.")

    
    def train(self):
        # --- 1. Selection du device (GPU si disponible, sinon CPU) ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        show(f"Training on: {device}")

        # --- 2. Decoupage des donnees en train (90%) et validation (10%) ---
        n          = int(0.9 * len(data))
        train_data = data[:n]
        val_data   = data[n:]

        # --- Fonction utilitaire : recupere un batch aleatoire ---
        def get_batch_split(split, batch_size=32):
            d  = train_data if split == "train" else val_data
            ix = torch.randint(len(d) - block_size, (batch_size,))
            x  = torch.stack([d[i:i+block_size]     for i in ix]).to(device)
            y  = torch.stack([d[i+1:i+block_size+1] for i in ix]).to(device)
            return x, y

        # --- Fonction d'evaluation : calcule la loss moyenne sans modifier les gradients ---
        @torch.no_grad()
        def estimate_loss(eval_iters=200):
            nn.Module.train(self, False)   # mode evaluation (pas de dropout, batchnorm fixe)
            out = {}
            for split in ("train", "val"):
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    xb, yb    = get_batch_split(split)
                    _, loss   = self(xb, yb)
                    losses[k] = loss.item()
                out[split] = losses.mean().item()
            nn.Module.train(self, True)    # retour en mode entrainement
            return out

        # --- 3. Optimiseur AdamW avec weight decay pour regulariser ---
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)

        # --- Scheduler Cosine : le learning rate descend progressivement vers lr/10 ---
        # Cela evite les oscillations en fin d'entrainement
        total_steps = 50_000
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr / 10
        )

        # --- 4. Parametres d'entrainement ---
        eval_interval    = 5000    # evaluer toutes les N etapes
        patience         = 5      # nombre d'evaluations sans amelioration avant d'arreter
        convergence_goal = 0.005  # seuil de val loss pour considerer la convergence atteinte

        # --- Variables de suivi ---
        step             = 0
        loss_sum         = 0.0
        best_val_loss    = float("inf")
        patience_counter = 0

        show("Starting training until convergence...")
        nn.Module.train(self, True)

        # --- 5. Boucle principale d'entrainement ---
        try:
            while True:

                # -- Forward pass : calcul des logits et de la loss --
                xb, yb       = get_batch_split("train")
                logits, loss = self(xb, yb)

                # -- Backward pass --
                optimizer.zero_grad(set_to_none=True)  # reset des gradients (plus rapide que zero_grad())
                loss.backward()

                # -- Gradient clipping : evite l'explosion du gradient --
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # -- Mise a jour des poids et du learning rate --
                optimizer.step()
                scheduler.step()

                loss_sum += loss.item()
                step     += 1

                # --- 6. Evaluation periodique ---
                if step % eval_interval == 0:
                    losses     = estimate_loss()
                    current_lr = scheduler.get_last_lr()[0]

                    show(
                        f"Step {step:>6} | "
                        f"train loss: {losses['train']:.4f} | "
                        f"val loss: {losses['val']:.4f} | "
                        f"lr: {current_lr:.2e}"
                    )

                    # -- Sauvegarde uniquement si la val loss s'ameliore --
                    if losses["val"] < best_val_loss:
                        best_val_loss    = losses["val"]
                        patience_counter = 0
                        try:
                            self.save()
                            show(f"Best model saved (val loss: {best_val_loss:.4f})")
                        except Exception as e:
                            show(f"Save failed: {e}")
                    else:
                        # -- Pas d'amelioration : on incremente le compteur de patience --
                        patience_counter += 1
                        show(f"No improvement ({patience_counter}/{patience})")

                    # -- Early stopping : arret si trop longtemps sans amelioration --
                    if patience_counter >= patience:
                        show(f"Early stopping triggered at step {step}.")
                        break

                    # -- Convergence atteinte si la val loss passe sous le seuil fixe --
                    if losses["val"] < convergence_goal:
                        show(f"Convergence reached at step {step}.")
                        break

        except KeyboardInterrupt:
            show("Training interrupted by user.")
            self.save()

        # --- 7. Resume final ---
        show(
            f"Training complete | "
            f"{step} steps | "
            f"avg loss: {loss_sum / max(step, 1):.4f} | "
            f"best val loss: {best_val_loss:.4f}"
        )

    def end(self):
        while True:
            try:
                self.save()
                show(f"<<<<<<<<<< END OF LOG ({datetime.datetime.now().strftime('%Y-%m-%d')}) >>>>>>>>>>")
                break
            except Exception as e:
                show(f"an error occured : {e}, retrying...")
    
    def chat(self):
        model = self
        while True:
            sp = spm.SentencePieceProcessor()
            sp.load("tokenizer.model")
            user = input(">>")
            if user == "/quit":
                break
            waitingAnimation()
            userInput = torch.tensor([sp.encode(user)], dtype=torch.long).to(device)
            out = model.generate(userInput, max_new_tokens=20).to(device)
            generated_ids = out[0].tolist()
            show(sp.decode(generated_ids))

model = KIPP()
model.load()
model.train()
model.chat()
model.end()