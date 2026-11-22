from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tokenizer import Tokenizer

torch.manual_seed(1337)  # type: ignore
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_printoptions(profile="full", linewidth=150)  # type: ignore


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int  # comes from the tokenizer at runtime
    block_size: int  # (dim T) max context_length for prediction
    n_embd: int  # (dim C) size of a token embedding


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int  # (dim B) independent sequences to process parallelly in one pass
    max_iters: int  # training iterations
    learning_rate: float
    eval_interval: int  # print the loss every eval_interval iterations
    eval_iters: int


class BigramLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # lookup table of embeddings
        # dim = vocab_size x vocab_size
        # row i is the embedding of token i. it has vocab_size numbers
        # one entry i,j represents how likely token j follows token i
        self.token_embedding_table: torch.nn.Embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        # print("Embedding table:")
        # print(f"  {self.token_embedding_table}")
        # print(f"  {self.token_embedding_table.weight.detach()}")
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self, xb: torch.Tensor, yb: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # integer array indexing
        # each token is within vocab_size
        # for each token, pick out its embedding table row
        # that row is the embedding
        # dim = batch_size x block_size x vocab_size (B,T,C)
        # for each char in each batch the logits for all possible following chars
        logits = self.token_embedding_table(xb)
        # print(f"Logits: {logits}")
        if yb is None:
            loss = None
        else:
            # compute negative log likelihood loss
            # goal: return logits and loss
            # isuse: pytorch expects dimensions differently
            B, T, C = logits.shape
            # so we need to juggle them around
            # flatten logits and targets
            logits = logits.view(B * T, C)
            yb = yb.view(B * T)
            # result: blocks and batches concatenated into one
            loss = F.cross_entropy(logits, yb)
        return logits, loss

    def print_embeddings(self):
        for i, row in enumerate(self.token_embedding_table.weight.detach()):
            print(f"{i}: {row}")

    def generate(self, idcs: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive loop
        Extend the B batches of size T by a new token one at a time.
        Continually generate across batches in the time dimension.
        From (B,T) to (B,T+1), then (B,T+2) and so on.
        """
        for _ in range(max_new_tokens):
            # do a prediction
            logits, _ = self(idcs)
            # get the last token from each batch
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idcs = torch.cat((idcs, idx_next), dim=1)
        return idcs


def main():
    train_config = TrainConfig(
        batch_size=4,
        max_iters=10000,
        learning_rate=1e-2,
        eval_interval=300,
        eval_iters=200,
    )

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    model_config = ModelConfig(vocab_size=len(tokenizer.vocab), block_size=8, n_embd=32)
    # encoding = tokenizer.encode(s="Hello World")
    # decoding = tokenizer.decode(encoding)
    # print(f"Encoded 'Hello World': {encoding}")
    # print(f"Decoded 'Hello World': {decoding}")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Data shape: {data.shape} Data Type: {data.dtype}")
    n_chars = 6
    print(f"First {n_chars} characters: {data[:n_chars]}")

    print("Splitting dataset: ")
    split_n = int(0.9 * len(data))
    train_data = data[:split_n]
    val_data = data[split_n:]
    print(f"  Test: 1 - {len(train_data):_}")
    print(f"  Val: {len(train_data) + 1:_} - {len(data):_}")

    x = train_data[: model_config.block_size]
    y = train_data[1 : model_config.block_size + 1]  # offset by one

    xb, yb = get_batch(
        train_data,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
    )
    print("Example input:")
    print(f"  {xb.size()}")
    print(f"  {xb}")
    # print("Decoded input:")
    # for row in xb.tolist():
    #     print(f"  {tokenizer.decode(row)!r}")
    print("Example target:")
    print(f"  {yb.size()}")
    print(f"  {yb}")
    # print("Decoded target:")
    # for row in yb.tolist():
    #     print(f"  {tokenizer.decode(row)!r}")

    m = BigramLM(model_config)
    logits, loss = m(xb, yb)  # calls m.forward()
    print("Bigram Model:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss}")

    print("Example generation before train:")
    start = torch.zeros(
        size=(1, 1), dtype=torch.long
    )  # need to make this a 64, because for us 64 is \n
    tokens = m.generate(idcs=start, max_new_tokens=100)[0]
    print(f"  {tokenizer.decode(tokens.tolist())!r}")
    # m.print_embeddings()

    # -- do training loop --
    optimizer = torch.optim.AdamW(m.parameters(), lr=train_config.learning_rate)
    print(f"Starting training loop with {train_config.max_iters} iterations...")
    train(
        data=train_data,
        model=m,
        optimizer=optimizer,
        config=train_config,
        block_size=model_config.block_size,
    )
    print("Training finished.")

    # -- do prediction --
    start = torch.zeros(
        size=(1, 1), dtype=torch.long
    )  # need to make this a 64, because for us 64 is \n
    print("Generation after training: ")
    tokens = m.generate(idcs=start, max_new_tokens=200)[0]
    print(f"  {tokenizer.decode(tokens.tolist())!r}")


def train(
    data: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    block_size: int,
) -> None:
    for i in range(config.max_iters):
        xb, yb = get_batch(
            data=data, block_size=block_size, batch_size=config.batch_size
        )
        _, loss = model(xb, yb)
        if i % config.eval_interval == 0:
            print(f"Loss:  {loss}")
        optimizer.zero_grad(set_to_none=True)  # reset gradients
        loss.backward()  # compute gradients
        optimizer.step()  # adjust weights using gradients


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    # print(f"Starting indeces for current batch: {ix}")
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


if __name__ == "__main__":
    main()
