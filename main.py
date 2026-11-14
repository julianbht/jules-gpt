import torch
from torch import nn
from torch.nn import functional as F

from tokenizer import Tokenizer

torch.manual_seed(1337)  # type: ignore
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_printoptions(profile="full", linewidth=150)  # type: ignore

# hyperparameters
block_size = 8  # (dim C) max context_length for prediction
batch_size = 4  # (dim B) independent sequences to process parallelly in one pass
max_iters = 500  # training iterations
max_new_tokens = 200
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200


def main():
    train()


def train():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = Tokenizer(text)
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

    x = train_data[:block_size]
    y = train_data[1 : block_size + 1]  # offset by one

    xb, yb = get_batch(train_data, block_size=block_size, batch_size=batch_size)
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

    m = BigramLM(len(tokenizer.vocab))
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

    # -- try to do an actual training loop --
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    print(f"Starting training loop with {max_iters} iterations...")
    for steps in range(max_iters):
        xb, yb = get_batch(
            data=train_data, batch_size=batch_size, block_size=block_size
        )
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # print(loss.item())

    print("Training finished.")

    start = torch.zeros(
        size=(1, 1), dtype=torch.long
    )  # need to make this a 64, because for us 64 is \n
    print("Generation after training: ")
    tokens = m.generate(idcs=start, max_new_tokens=max_new_tokens)[0]
    print(f"  {tokenizer.decode(tokens.tolist())!r}")


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    # print(f"Starting indeces for current batch: {ix}")
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # lookup table of embeddings
        # dim = vocab_size x vocab_size
        # row i is the embedding of token i. it has vocab_size numbers
        # one entry i,j represents how likely token j follows token i
        self.token_embedding_table: torch.nn.Embedding = nn.Embedding(
            vocab_size, vocab_size
        )
        # print("Embedding table:")
        # print(f"  {self.token_embedding_table}")
        # print(f"  {self.token_embedding_table.weight.detach()}")

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


if __name__ == "__main__":
    main()
