import torch
from torch import nn
from torch.nn import functional as F

# ------
# Batched Averaging (for Attention mechansim)
# ------

# Create example matrix that matches our shape
torch.manual_seed(1337)  # type: ignore
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)

# What is our goal here?
# We try to average the C vector of the previous tokens
# Or: We try to average the logits of the previous tokens
# This will allow us to attend to the previous tokens in the current prediction
# in a very basic way

# version 1: for loop (inefficient)
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])
print(xbow[0])

a = torch.tril(torch.ones(3, 3))
print(a)
print(torch.sum(a, 1, keepdim=True))
a = a / torch.sum(a, 1, keepdim=True)
print(a)

# version 2: vectorized with broadcasting
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
print(wei)
xbow2 = wei @ x  # (T,T) @ (B,T,C) --> (B,T,T) @ (B,T,C) = (B,T,C)
print(torch.allclose(xbow, xbow2, atol=1e-6))

# version 3: vectorized with softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3, atol=1e-6))

# ------
# Trying out attention
# ------

# single head of masked (causal) self attention

torch.manual_seed(1337)  # type: ignore
B, T, C = 4, 6, 3  # type: ignore
# x holds random token embedding vectors
x = torch.randn(B, T, C)

# create linear layers
# initializes the weight matrices with random weights
# input: vectors of size C. output: vectors of size head_size.
head_size = 2
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# create query, key, value vector for all embeddings in the batch
q = query(x)  # (B, T, hs)
k = key(x)  # (B, T, hs) (hs = head_size)
v = value(x)  # (B, T, hs)

# dot product all keys and all queries of the entire
# batch at the same time
scores = q @ k.transpose(-2, -1)  # (B, T, hs) @ (B, hs, T) ---> (B, T, T)

# Use lower triangular to "hide (mask) the future"
tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float("-inf"))

# at this point, scores represent the affinities between the tokens
# use softmax to turn them into probability weights
# (which are usually called attention weights)
weights = F.softmax(scores, dim=-1)

# finally use the attention weights we worked hard to acquire
# weigh each v with it
out = weights @ v  # (B, T, T) @ (B, T, hs) ---> (B, T, hs)
print(out.shape)
