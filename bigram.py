import torch
from torch import nn
from torch.nn import functional as F


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
