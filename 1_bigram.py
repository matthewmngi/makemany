import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))

s_i = {s:i+1 for i, s in enumerate(chars)}
s_i["."] = 0

i_s = {i:s for s, i in s_i.items()}
g = torch.Generator().manual_seed(2147483647)
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s_i[ch1]
        ix2 = s_i[ch2]
        N[ix1, ix2] += 1

"""
plt.figure(figsize=(8, 8))
plt.imshow(N, cmap="YlGnBu")
"""

P = (N +1 ).float()
P /= P.sum(1, keepdim=True)

"""
for i in range(27):
    for j in range(27):
        chstr = i_s[i] + i_s[j]
        plt.text(j, i, chstr, ha="center", color="gray", fontsize="xx-small")
        #plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray", fontsize="xx-small")

plt.axis("off")
plt.show()
"""


for i in range(5):
    out = []
    ix = 0

    while True:
        p = P[ix]
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(i_s[ix])
        
        if ix == 0: 
            break

    print("".join(out))

"""
log_likelihood = 0.0
n = 0


for w in ["matthewqz"]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s_i[ch1]
        ix2 = s_i[ch2]
        
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        
        log_likelihood += logprob
        n += 1
        
        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")
        
print(log_likelihood/n)
"""

# training set
xs, ys = [], []

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s_i[ch1]
        ix2 = s_i[ch2]
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)


W = torch.randn((27, 27), generator=g, requires_grad=True)
loss = 0.0

for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # one-hot encoded inputs to the neural net
    logits = xenc @ W

    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(len(xs)), ys].log().mean()
    #print(loss.item())

    # backward pass
    W.grad = None # zero grad
    loss.backward()

    W.data += -50 * W.grad
    
print(loss)

for i in range(5):
    out = []
    ix = 0
    
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(i_s[ix])
        
        if ix == 0:
            break
    
    print("".join(out))
    
