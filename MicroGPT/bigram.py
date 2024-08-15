import torch
import torch.nn as nn
from torch.nn import functional as F 

#hyperparameters
batch_size = 32 #how many chunks we will process at 1 time?
block_size = 8 #size of chunk, sequence length, context length
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters =200
n_embd = 32

#reproduceability
torch.manual_seed(1337)

#read dataset, text
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

#all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


#split dataset into test/train
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch (split):
    data = train_data if split =='train' else val_data
    #index of x, generate randome indices to grab a parallel chunk from
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #generates a 1D tensor of size batch_size where 
    # each element is a random integer between 0 and 
    # len(data) - block_size - 1 (both inclusive).


    #This slices the data tensor to extract a subsequence starting from 
    # index i up to i + block_size. block_size defines the length of these subsequences.
    x = torch.stack([data[i:i+block_size] for i in ix])
    # This list comprehension creates a list of these subsequences for each index i in the list ix

    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y 


#turn off gradient calculatios, because we'll use adam????
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)



#bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        #create an instance of the embedding layer from torch.nn
        #each row is a token from the vocab that is embedded
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #num of embedding dims
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idk is random indeces of data to pick out chunks
        #
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1,:] 

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx,idx_next), dim=1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
     
    if iter % eval_interval ==0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



