import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from TransformerBlock import Block
import os

def open_file(filepath: str, mode: str, new_text: str = None, encoding: str = "utf-8") -> str | None:
    """
    Opens a file for reading or appending text.

    Args:
        filepath (str): Full path to the file.
        mode (str): 'r' to read content, 'a' to append text.
        new_text (str, optional): Text to append if mode == 'a'.
        encoding (str): File encoding (default "utf-8").

    Returns:
        str | None: File content if mode == 'r', else None.

    Raises:
        ValueError: If the provided mode is neither 'r' nor 'a'.
    """
    if mode == 'r':
        with open(filepath, mode, encoding=encoding) as file:
            return file.read()
    elif mode == 'a':
        with open(filepath, mode, encoding=encoding) as file:
            if new_text is not None:
                file.write(new_text)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

class GPT(nn.Module):
    """
    GPT-style language model (decoder-only transformer).
    Trains its own BPE tokenizer (SentencePiece) on a text corpus,
    then stacks multiple Transformer blocks to predict the next token.
    Provides methods for creating, loading, saving, training, and chatting with the model.
    """

    def __init__(self, source_dir: str):
        """
        Initializes the tokenizer, loads the corpus, and builds all
        network layers (embeddings, Transformer blocks, output head).

        Args:
            source_dir (str): Directory containing "corpus.txt" and where
                tokenizer/checkpoints will be saved.
        """
        super().__init__()
        self.source_dir = source_dir
        self.token_ids = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = None
        self.vocab_size = None
        self.block_size = 64
        self.embedding_dim = 256
        self.num_heads = 8
        self.num_layers = 6
        self.learning_rate = 1e-3
        self.position_embedding = nn.Embedding(self.block_size, self.embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(self.embedding_dim, self.block_size, self.num_heads).to(self.device)
              for _ in range(self.num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.embedding_dim)
        self.loaded_model = None

    def load_corpus(self, corpus_name: str) -> None:
        """
        Trains a SentencePiece tokenizer on the corpus and loads the data.

        Args:
            corpus_name (str): Name of the corpus file (e.g., "corpus.txt").

        Raises:
            FileNotFoundError: If the corpus file is not found.
            RuntimeError: If an error occurs during loading.
        """
        try:
            self.tokenizer = spm.SentencePieceProcessor()
            spm.SentencePieceTrainer.Train(
                input=self.source_dir + corpus_name,
                model_prefix="tokenizer",
                vocab_size=115,
                model_type="bpe"
            )
            self.tokenizer.load(self.source_dir + "tokenizer.model")
            self.vocab_size = self.tokenizer.get_piece_size()
            self.token_ids = self.tokenizer.encode(
                open_file(self.source_dir + corpus_name, "r"), out_type=int
            )
            self.data = torch.tensor(self.token_ids, dtype=torch.long).to(self.device)
            self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.output_head = nn.Linear(self.embedding_dim, self.vocab_size)
            self.to(self.device)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Corpus '{corpus_name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Failed to load corpus '{corpus_name}': {e}")

    def forward(self, indices: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, float | None]:
        """
        Forward pass: transforms token indices into logits for next-token prediction.

        Args:
            indices (torch.Tensor): Token indices, shape (batch, time).
            targets (torch.Tensor, optional): Expected indices to compute loss, shape (batch, time).

        Returns:
            tuple: (logits, loss) where loss is None if targets is None.
        """
        if not hasattr(self, "token_embedding"):
            raise RuntimeError("No corpus loaded. Call load_corpus() before training.")

        batch_size, seq_length = indices.shape
        token_embeddings = self.token_embedding(indices)
        position_embeddings = self.position_embedding(torch.arange(seq_length, device=indices.device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        loss = None
        if targets is not None:
            batch_size, seq_length, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * seq_length, vocab_size),
                targets.view(batch_size * seq_length)
            )
        return logits, loss

    def create_model(self, model_name: str) -> None:
        """
        Saves the current (often untrained) model state to a new file.

        Args:
            model_name (str): Name of the file to create in self.source_dir.
        """
        try:
            torch.save(self.state_dict(), f=self.source_dir + model_name + ".pth")
            print(f"Model {model_name}.pth created.")
        except FileExistsError:
            print(f"{model_name}.pth already exists.")
        except Exception as e:
            print(f"Error: {e}")

    def load_model(self, model_name: str) -> None:
        """
        Loads model weights from a saved checkpoint.

        Args:
            model_name (str): Name of the checkpoint file to load (without .pth extension).
        """
        try:
            self.load_state_dict(torch.load(self.source_dir + model_name + ".pth"))
            self.loaded_model = model_name
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No model found. Creating a new one...")
            self.create_model(model_name)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading file:\n{e}")

    def save_model(self) -> str:
        """
        Saves the current model state under the loaded model's name (self.loaded_model).

        Returns:
            str: "saved" on success, "Error" on failure.
        """
        try:
            if self.loaded_model is None:
                raise ValueError("No model loaded. Use load_model() first.")
            torch.save(self.state_dict(), f=self.source_dir + self.loaded_model + ".pth")
            return "saved"
        except Exception as e:
            print(f"Save error: {e}")
            return "Error"

    def train(
        self,
        convergence_threshold: float,
        eval_interval: int = 100,
        patience: int = 1000
    ) -> None:
        """
        Full training loop for the model.
        Splits data into 90% train / 10% validation, performs forward/backward passes
        with AdamW and cosine scheduler, periodically evaluates validation loss,
        saves the best model, and stops either by early stopping (patience) or
        when the convergence threshold is reached.

        Can be interrupted with Ctrl+C (KeyboardInterrupt), in which case the model is saved.

        WARNING: If convergence_threshold < 0.1, training may never stop.
        """
        if self.data is None:
            raise RuntimeError("No corpus loaded. Call load_corpus() before train().")
        if convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be > 0.")

        train_limit = int(0.9 * len(self.data))
        train_data = self.data[:train_limit]
        val_data = self.data[train_limit:]

        def get_batch(split: str, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns a batch of inputs and targets."""
            data = train_data if split == "train" else val_data
            if len(data) <= self.block_size:
                raise ValueError(
                    f"'{split}' split too small: {len(data)} tokens for block_size={self.block_size}"
                )
            random_indices = torch.randint(len(data) - self.block_size, (batch_size,))
            input_batch = torch.stack([data[i:i + self.block_size] for i in random_indices])
            target_batch = torch.stack([data[i + 1:i + self.block_size + 1] for i in random_indices])
            return input_batch, target_batch

        @torch.no_grad()
        def estimate_loss(eval_iterations: int = 200) -> dict[str, float]:
            """Estimates loss on train/val splits."""
            nn.Module.train(self, False)
            results = {}
            for split in ("train", "val"):
                losses = torch.zeros(eval_iterations)
                for i in range(eval_iterations):
                    input_batch, target_batch = get_batch(split)
                    _, loss = self(input_batch, target_batch)
                    losses[i] = loss.item()
                results[split] = losses.mean().item()
            nn.Module.train(self, True)
            return results

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2
        )
        total_steps = 50_000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.learning_rate / 10
        )
        step = 0
        loss_sum = 0.0
        best_val_loss = 20.0
        patience_counter = 0
        progress_percentage = 0
        print(f"Training: target -> loss < {convergence_threshold}")
        nn.Module.train(self, True)

        try:
            while True:
                # Forward/backward pass
                input_batch, target_batch = get_batch("train")
                logits, loss = self(input_batch, target_batch)

                # Optimization
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item()
                step += 1

                # Periodic evaluation
                if step % eval_interval == 0:
                    losses = estimate_loss()
                    current_lr = scheduler.get_last_lr()[0]

                    if losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
                        patience_counter = 0
                        progress_percentage = min(100, int((convergence_threshold / losses["val"]) * 100))
                        print(
                            f"Step {step:>6} | "
                            f"train loss: {losses['train']:.4f} | "
                            f"val loss: {losses['val']:.4f} | "
                            f"lr: {current_lr:.2e} | "
                            f"status: {self.save_model()}\n"
                            f"[{progress_percentage * '#' + (100 - progress_percentage) * ' '}] "
                            f"{progress_percentage:.1f}%"
                        )
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Too many iterations without improvement. Stopped at step: {step}.")
                        break
                    if losses["val"] < convergence_threshold:
                        print(f"Convergence reached at step: {step}.")
                        break
        except Exception as e:
            print(f"\nError during training: {type(e).__name__}:\n{e}")
            self.save_model()

    def generate(self, indices: torch.Tensor, num_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens autoregressively from a starting sequence.

        Args:
            indices (torch.Tensor): Starting token sequence, shape (batch, time).
            num_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: Original sequence with generated tokens appended.
        """
        for _ in range(num_new_tokens):
            truncated_indices = indices[:, -self.block_size:]
            logits, _ = self(truncated_indices)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            indices = torch.cat((indices, next_token), dim=1)
        return indices

    def chat(self, text: str = "", num_new_tokens: int = 20) -> None:
        """
        Encodes user text, generates a sequence of tokens with the model,
        and prints the decoded output.

        Args:
            text (str): Input message to send to the model.
            num_new_tokens (int): Number of tokens to generate.
        """
        try:
            if self.tokenizer is None:
                raise RuntimeError("No tokenizer loaded. Ensure you have loaded a corpus.")
            if len(text) < 1:
                print("Empty message.")
                return

            self.tokenizer.load(self.source_dir + "tokenizer.model")
            user_input = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long).to(self.device)
            output = self.generate(user_input, num_new_tokens)
            generated_ids = output[0].tolist()
            print(self.tokenizer.decode(generated_ids))
        except Exception as e:
            print(f"Generation error: {e}")

    def get_info(self) -> tuple[str, ...]:
        """
        Returns diagnostic information about the model: data tensor,
        vocabulary size, and total number of parameters.

        Returns:
            tuple[str]: Formatted info lines.

        Raises:
            NameError: If info cannot be retrieved (e.g., no model loaded).
        """
        try:
            return (
                "--------- Model Information ---------\n",
                f"tensor: {self.data}\n",
                f"vocab size: {self.vocab_size}\n",
                f"total parameters: {sum(p.numel() for p in self.parameters()):,}"
            )
        except Exception:
            raise NameError("Cannot display model info. Ensure a model is loaded.")

if __name__ == "__main__":
    model = GPT("C:/Users/chevr/Desktop/FIchiers/KIPP-AI/").to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_str = "bonjour"
    for name in ["Atlas", "Gibraltar"]:
        model.create_model(name)
        model.load_model(name)
        model.load_corpus(name + "Corpus.txt")
        model.train(0.2)
        for i in range(len(test_str)):
            model.chat(test_str[:i+1])