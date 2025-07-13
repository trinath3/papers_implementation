#imports 
import torch 
from torch import nn 
import math 

dim_input = 512
n_heads  = 8


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,Q,K,V):
        x = torch.matmul(Q,K.transpose(-2,-1))
        x = x/math.sqrt(K.size(-1))
        x = nn.functional.softmax(x,dim=-1)
        x = torch.matmul(x,V)
        return x


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = nn.Linear(dim_input,dim_input // n_heads)
        self.k_linear = nn.Linear(dim_input,dim_input// n_heads)
        self.v_linear = nn.Linear(dim_input,dim_input// n_heads)
        self.attention = Attention()

    def forward(self,x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        return self.attention(Q,K,V)

class multihead_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_heads)])
        self.linear = nn.Linear(dim_input,dim_input)

    
    def forward(self,x):
        x = [head(x) for head in self.heads]
        x = torch.cat(x,dim=-1)
        return self.linear(x)
        








        




