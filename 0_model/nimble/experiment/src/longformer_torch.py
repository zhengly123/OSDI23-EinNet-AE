import torch
import torch.nn as nn
import torch.nn.functional as F
import math


n_layers = 1
seq_len = 10000
embed = 512
n_heads = 8
feat_len = 64
w = 1000
dilation = 4


def dilated_indices(seq_begin: int, seq_end: int, w, dilation: int):
    D = torch.LongTensor([i for i in range(0, 2 * w * dilation + 1, dilation)])
    sD = torch.stack([D] * (seq_end - seq_begin))
    
    F = torch.LongTensor([[i] for i in range(seq_begin, seq_end)])
    
    return sD + F


def dilated_attention_(q: torch.Tensor, pad_k: torch.Tensor, pad_v: torch.Tensor,
                       w: int, dilation: int, seq_begin: int, seq_end: int):
    bs, n_heads, _, feat_len = q.size()
    seq_len = seq_end - seq_begin
    sqrt_d = math.sqrt(feat_len)

    indices = dilated_indices(seq_begin, seq_end, w, dilation)

    # diag_k = torch.gather(pad_k, dim=2, index=indices).reshape(bs, n_heads, seq_len, 2 * w + 1, feat_len)
    # diag_v = torch.gather(pad_v, dim=2, index=indices).reshape(bs, n_heads, seq_len, 2 * w + 1, feat_len)
    diag_k = pad_k[:, :, indices, :]
    diag_v = pad_v[:, :, indices, :]
    
    q = q[:, :, seq_begin:seq_end, :]
    attn = torch.einsum("bijp,bijkp->bijk", q, diag_k)
    attn = F.softmax(attn, dim=-1) / sqrt_d
    
    return torch.einsum("bijk,bijkp->bijp", attn, diag_v)


def dilated_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: int, dilation: int):
    bs, n_heads, seq_len, feat_len = q.size()
    
    paddings = (0, 0, w * dilation, w * dilation)
    
    pad_k = F.pad(k, paddings, "constant", 0)
    pad_v = F.pad(v, paddings, "constant", 0)
    
    if bs == 1:
        N = 5
    elif bs == 16:
        N = 100
    else:
        raise ValueError("Unsupported BS = {}".format(bs))
    
    partitions = [i for i in range(0, seq_len, seq_len // N)]
    if len(partitions) > 1 and (seq_len - partitions[-1]) < seq_len // N // 2:
        partitions[-1] = seq_len
    else:
        partitions.append(seq_len)
    
    outputs = []
    for seq_begin, seq_end in zip(partitions[:-1], partitions[1:]):
        output = dilated_attention_(
            q, pad_k, pad_v, w, dilation, seq_begin, seq_end)
        outputs.append(output)
    
    return torch.cat(outputs, dim=2)


class LongFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(n_heads * feat_len, n_heads * feat_len, bias=False)
        self.k = nn.Linear(n_heads * feat_len, n_heads * feat_len, bias=False)
        self.v = nn.Linear(n_heads * feat_len, n_heads * feat_len, bias=False)
        
        self.ffn1 = nn.Linear(embed, embed)
        self.ffn2 = nn.Linear(embed, embed)
        
    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        q = torch.transpose(
            torch.reshape(self.q(x), (-1, seq_len, n_heads, feat_len)),
            dim0=1, dim1=2
        )
        
        k = torch.transpose(
            torch.reshape(self.k(x), (-1, seq_len, n_heads, feat_len)),
            dim0=1, dim1=2
        )
        
        v = torch.transpose(
            torch.reshape(self.v(x), (-1, seq_len, n_heads, feat_len)),
            dim0=1, dim1=2
        )
        
        x1 = dilated_attention(q, k, v, w, dilation)
        
        x1 = torch.transpose(x1, dim0=1, dim1=2)
        x1 = torch.reshape(x1, (-1, seq_len, embed))
        
        f = F.relu(self.ffn1(x1))
        g = F.relu(self.ffn2(f))
        
        return x + g


class LongFormerPart1(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(n_heads * feat_len, n_heads * feat_len, bias=False)
        self.k = nn.Linear(n_heads * feat_len, n_heads * feat_len, bias=False)
        self.v = nn.Linear(n_heads * feat_len, n_heads * feat_len, bias=False)
        
    def forward(self, x: torch.Tensor):
        q = torch.transpose(
            torch.reshape(self.q(x), (-1, seq_len, n_heads, feat_len)),
            dim0=1, dim1=2
        )
        
        k = torch.transpose(
            torch.reshape(self.k(x), (-1, seq_len, n_heads, feat_len)),
            dim0=1, dim1=2
        )
        
        v = torch.transpose(
            torch.reshape(self.v(x), (-1, seq_len, n_heads, feat_len)),
            dim0=1, dim1=2
        )
        
        return q, k, v


class LongFormerPart2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ffn1 = nn.Linear(embed, embed)
        self.ffn2 = nn.Linear(embed, embed)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor):
        g = torch.transpose(g, dim0=1, dim1=2)
        g = torch.reshape(g, (-1, seq_len, embed))
        
        g = F.relu(self.ffn1(g))
        g = F.relu(self.ffn2(g))
        
        return x + g


if __name__ == "__main__":
    with torch.no_grad():
        lf = LongFormer().cuda()
        lf.eval()
        x = torch.rand((1, seq_len, embed)).cuda()
        y = lf(x)
    print(y.size())
