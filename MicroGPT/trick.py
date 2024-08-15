import torch

torch.manual_seed(1337)
B,T,C = 4,8,2 #4 chunks at a time, each is 8 tokens long
x = torch.randn(B,T,C)
x.shape
print(x.shape)

#getting average of x[b,t] = mean _{i<=t} x[b,i]

xbow = torch.zeros((B,T,C)) #bag of words
for b in range (B): #iterating over every batch dimension, so a row(chunk) in a 4x8 batch matrix
    for t in range (T):
        xprev = x[b,:t+1] # {t,C}
        xbow[b,t] = torch.mean(xprev,0)

#print(x[0])
#print(xbow[0])

torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a,1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b

print('a==')
print(a)
print('--')
print('b==')
print(b)
print('--')
print('c==')
print(c)


