import torch

from tokenizer import Tokenizer


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
    print(f"First 5 characters: {data[:5]}")


if __name__ == "__main__":
    main()
