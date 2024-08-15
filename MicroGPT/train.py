with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()


print (f"length of data in chars: {len(text)}")

#first 1000 characters
# print (text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#TOKENIZE INPUT TEXT
#this will be done on a character by character level because it is simpler
#other larger models use different methods like tiktokken or sub sentence

#creating dictionaries with index,character
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
#lambda is shorthand in line functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#testing out encoders/decoders
print(encode("hii there"))
print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
#commenting so it doesnt keep printing
#print(data[:1000])

#splitting to test/train

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#training the model/transformer with chunks of data
#instead of all training data at once
block_size = 8
train_data[:block_size+1]
#it wasnt printing
print(train_data[:block_size+1])


#Going more in-depth with small training chunks

#x is input to transformers, aka previous context 
x = train_data[:block_size]
#y is target, aka what most likely comes next given the context
y = train_data[1:block_size+1]
for t in range (block_size):
    context = x[:t+1]
    target = y[t]
    #showing all the examples of context, targets and probabilities in 1 chunk
    print(f"when input is {context} the target is {target}")

#random seed for reproduceability 
torch.manual_seed(1337)
batch_size = 4 #how many chunks at a time together for training
block_size = 8 #context length

def get_batch(split):
    data = train_data if split == 'train' else val_data

    #generate random indeces to pull chunks from in the data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    #input element in those 'random chunk' arrays 
    x = torch.stack([data[i:i+block_size] for i in ix])
    #y is x offset by 1
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #also, stack those chunks that 8 chars long on top of eachother to see view as a matrix
    return x,y

xb, yb = get_batch('train')
print("inputs:")
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----------')

for b in range(batch_size):
    #  4 rows with 8 columns
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target is: {target}") #----------------------*******************


#CALLING BIGRAM TYPE OF LLM ALREADY
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        #creates an instance of embedding layer from torch.nn
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        #each row is a token from the vocab that is embedded 

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #Batch aka 4, Time aka 8, Channel aka 65

        if targets is None:
            loss = None
        else:
        #reshaping logits to properly get cross entropy in pytorch
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range (max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

#calling the model and putting in our inputs (context and target)
m = BigramLanguageModel(vocab_size)
logits,loss = m(xb, yb)
print(logits.shape)
print(loss)

#create a singular 1x1 tensor of 0, which is the new line character
#ask for 100 tokens/symbols
#create it into a list for the ongoing generation
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


#creating an optimizer object
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

batch_size=32
for steps in range (10000):

    #sample batch of data
    xb,yb = get_batch('train')

    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=400)[0].tolist()))


