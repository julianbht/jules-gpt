def main():
    train()


def train():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Input has this many characters: {len(text):_}")
    vocab = extract_vocab(text)
    print(f"Vocab: {''.join(vocab)}")
    print(f"Vocab Size: {len(vocab)}")
    # [print(i, ch) for i, ch in enumerate(vocab)]
    stoi_map = {ch: i for i, ch in enumerate(vocab)}
    itos_map = {i: ch for i, ch in enumerate(vocab)}
    encoding = encode(s="Hello World", stoi_map=stoi_map)
    decoding = decode(encoding, itos_map=itos_map)
    print(f"Encoded 'Hello World': {encoding}")
    print(f"Decoded 'Hello World': {decoding}")


def encode(s: str, stoi_map: dict[str, int]) -> list[int]:
    return [stoi_map[c] for c in s]


def decode(l: list[int], itos_map: dict[int, str]) -> str:
    return "".join([itos_map[i] for i in l])


def extract_vocab(text: str) -> list[str]:
    return sorted(set(text))


if __name__ == "__main__":
    main()
