class Tokenizer:
    def __init__(self, text: str):
        self.vocab = self.extract_vocab(text)
        self.stoi_map = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos_map = {i: ch for i, ch in enumerate(self.vocab)}

        print("Tokenizer:")
        print(f"  Input has this many characters: {len(text):_}")
        print(f"  Vocab: {''.join(self.vocab)!r}")
        print(f"  Vocab Size: {len(self.vocab)}")
        # [print(repr(i), repr(ch)) for i, ch in enumerate(self.vocab)]

    def encode(self, s: str) -> list[int]:
        return [self.stoi_map[c] for c in s]

    def decode(self, l: list[int]) -> str:
        return "".join([self.itos_map[i] for i in l])

    def extract_vocab(self, text: str) -> list[str]:
        return sorted(set(text), reverse=True)
