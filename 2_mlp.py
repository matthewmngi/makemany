import torch, random
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))

s_i = {s:i+1 for i, s in enumerate(chars)}
s_i["."] = 0

i_s = {i:s for s, i in s_i.items()}

# building dataset

block_size = 3

def build_dataset(words):
    
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        
        for ch in w + ".":
            ix = s_i[ch]
            X.append(context)
            Y.append(ix)
            # print("".join(i_s[i] for i in context), "---->", i_s[ix])
            context = context[1:] + [ix]
            
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)

w1 = torch.randn((30, 200))
b1 = torch.randn(200)

w2 = torch.randn((200, 27))
b2 = torch.randn(27)

parameters = [C, w1, b1, w2, b2]

for p in parameters:
    p.requires_grad = True

""" lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri = [] """
lossi = []
stepi = []

# training loop
for i in range(50000):
    # mini batch
    ix = torch.randint(0, Xtr.shape[0], (32,))
    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ w1 + b1)

    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    # backward pass
    for p in parameters:
        p.grad = None
        
    loss.backward()
    
    # update
    # lr = lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad
      
    # track stats  
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())
    
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ w1 + b1)
logits = h @ w2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

g2 = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ w1 + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g2).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    
    print("".join(i_s[i] for i in out))