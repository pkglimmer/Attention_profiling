import os
import torch
from torch.nn import Linear, Module
import torch.distributed as dist
import torch.multiprocessing as mp


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def init_process(rank, size, qk, k_share, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(qk, k_share)

class SplitAttentionLayer(Module):
    """
    (single head self attention)
    Generate Q, K, V matrices with dimensions n x d.
    Attention = Softmax(Q K^T) V
    """
    def __init__(self, n, d, n_head = 1):
        super(SplitAttentionLayer, self).__init__()
        # random init
        self.DEBUG = True
        self.Q = torch.randn(n, d)
        self.K = torch.randn(n, d)
        self.V = torch.randn(n, d)

        self.A1 = torch.zeros(n // 2, n).to('cuda:0')
        self.A2 = torch.zeros(n // 2, n).to('cuda:1')
        self.n, self.d = n, d

    def _dist_qk(self, qk, k_share):
        # n0, d0 = qk[0].shape
        rank = qk.device.index
        req0, req1 = None, None
        dist.send(tensor=qk[1], dst=1-rank)
        print(f'Rank {rank} started sending')
        req1 = dist.irecv(tensor=k_share, src=1-rank)
        print(f'Rank {rank} started receiving')
        req1.wait()
        if self.DEBUG:
            print(f'rank {rank}, qk[0] shape: {qk[0].shape}, qk[1].shape: \
                {qk[1].shape}, k_share.shape {k_share.shape}')
        print(f'qk.device {qk.device}')
        # A = torch.matmul(qk[0], torch.cat((qk[1].T, k_share.T), 1))
        

    def forward(self):
        # split Q, K
        qk0 = torch.stack( (self.Q[:self.n//2, :], self.K[:self.n//2, :]), 1)
        qk1 = torch.stack( (self.Q[self.n//2:, :], self.K[self.n//2:, :]), 1)

        size = 2
        qk0 = qk0.to(f'cuda:0')
        qk1 = qk1.to(f'cuda:1')
        k_share_0 = torch.zeros_like(qk0[0]).to(f'cuda:0')
        k_share_1 = torch.zeros_like(qk0[0]).to(f'cuda:1')


        mp.set_start_method('spawn')
        p1 = mp.Process(target=init_process, args=(0, size, qk0, k_share_0, self._dist_qk))
        p1.start()
        p2 = mp.Process(target=init_process, args=(1, size, qk1, k_share_1, self._dist_qk))
        p2.start()
    
        return

def main():
    attn = SplitAttentionLayer(16, 4)


if __name__ == "__main__":
    main()
