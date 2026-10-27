def main():
    train()


def train():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"length: {len(text)}")


if __name__ == "__main__":
    main()
