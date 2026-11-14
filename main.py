import torch
from torch import nn

from bigram import BigramLM
from tokenizer import Tokenizer

torch.manual_seed(1337)  # type: ignore
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_printoptions(profile="full", linewidth=150)  # type: ignore

# hyperparameters
block_size = 8  # (dim T) max context_length for prediction
batch_size = 4  # (dim B) independent sequences to process parallelly in one pass
max_iters = 10000  # training iterations
max_new_tokens = 200  # how many tokens to predict in one generation loop
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200


def main():
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

    # -- do training loop --
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    print(f"Starting training loop with {max_iters} iterations...")
    train(data=train_data, model=m, max_iters=max_iters, optimizer=optimizer)
    print("Training finished.")

    # -- do prediction --
    start = torch.zeros(
        size=(1, 1), dtype=torch.long
    )  # need to make this a 64, because for us 64 is \n
    print("Generation after training: ")
    tokens = m.generate(idcs=start, max_new_tokens=max_new_tokens)[0]
    print(f"  {tokenizer.decode(tokens.tolist())!r}")


def train(
    data: torch.Tensor,
    model: nn.Module,
    max_iters: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    for i in range(max_iters):
        xb, yb = get_batch(data=data, batch_size=batch_size, block_size=block_size)
        _, loss = model(xb, yb)
        if i % 500 == 0:
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
