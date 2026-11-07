import torch
from torch import nn

from tokenizer import Tokenizer

torch.manual_seed(1337)  # type: ignore


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

    block_size = 8  # max context_length for prediction
    x = train_data[:block_size]
    y = train_data[1 : block_size + 1]  # offset by one
    # for t in range(block_size):
    #     print(f"For {x[: t + 1]} target is {y[t]}")

    block_size = 8  # max context_length for prediction
    batch_size = 4  # independent sequences to process parallelly in one pass
    xb, yb = get_batch(train_data, block_size=block_size, batch_size=batch_size)
    print("Example input:")
    print(f"  {xb.size()}")
    print(f"  {xb}")
    print("Example target:")
    print(f"  {yb.size()}")
    print(f"  {yb}")

    m = BigramLM(len(tokenizer.vocab))
    out = m(xb, yb)  # calls m.forward()
    print("Bigram Model:")
    print(f"  {out.shape}")


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    print(f"Starting indeces for current batch: {ix}")
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
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        # integer array indexing
        # each token is within vocab_size
        # for each token, pick out its embedding table row
        # that row is the embedding
        logits = self.token_embedding_table(idx)
        # dim = block_size x batch_size x vocab_size
        # for each char in each batch the logits for all possible following chars
        return logits


if __name__ == "__main__":
    main()
