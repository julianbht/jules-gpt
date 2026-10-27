def main():
    train()


def train():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Input has this many characters: {len(text):_}")
    vocab = extract_vocab(text)
    print(f"Vocab: {''.join(vocab)}")
    print(f"Vocab Size: {len(vocab)}")


def extract_vocab(text: str) -> list[str]:
    return sorted(set(text))

def

if __name__ == "__main__":
    main()
