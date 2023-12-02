import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.profiler import profile, record_function, ProfilerActivity

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.bias.data.fill_(0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class ViTSelfAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_dropout: float = 0,
                 dropout: float = 0,
                 bias: bool = True):
        super().__init__()
        self.attention_head_size = hidden_size // num_heads
        self.query_key_value = nn.Linear(hidden_size,
                                             3 * hidden_size,
                                             bias=bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # the size of x is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # the size of qkv is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE*3)
        
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        # the size of q is (BATCH_SZIE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE//NUM_HEADS)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # the size of x is (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        # the size of x after matmul is (BATCH_SZIE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE//NUM_HEADS)
        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        # the size of x after reshape is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = x.reshape(new_context_layer_shape)
        # the size of x after dense is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = self.dense(x)
        x = self.dropout(x)

        return x

def test_scaled_dot_product():
    seq_len, d_k = 4, 2
    pl.seed_everything(42)
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    values, attention = scaled_dot_product(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

def test_attention(model, data):
    output = model(data)
    output.mean().backward()
    

BATCH_SIZE = 16
SEQ_LENGTH = 2048
NUM_HEADS = 4
HIDDEN_SIZE = 64

if __name__ == "__main__":
    n_iter = 50
    from tqdm import tqdm
    bar = tqdm(range(n_iter))
    model = ViTSelfAttention(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS).cuda()
    print(f'nparams: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data = torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE).cuda()
    for i in bar:
        test_attention(model, data)
