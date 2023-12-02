# Peikun September 23rd
# Script for experiment only

import os
import time
import torch
from torch.nn import Linear, Module
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def init_process(rank, size, send_tensor, recv_tensor, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, send_tensor, recv_tensor)

def split_matrix(M):
    """
    Args:
        M (2D tensor): Q or K or V, shape = (n, d)

    Returns:
        M1, M2 (2D tensor): shape (2, n//2, d)
        M1[0] is the upper half of M, M1[1] is all zeros;
        M2[1] is the lower half of M, M2[0] is all zeros.
    """
    n_split = 2
    n, d = M.shape
    assert n % n_split == 0
    M1, M2 = torch.zeros_like(M), torch.zeros_like(M)
    M1[:n//n_split, :] = M[:n//n_split, :]
    M2[n//n_split:, :] = M[n//n_split:, :]
    M1 = M1.view(n_split, n//n_split, d)
    M2 = M2.view(n_split, n//n_split, d)
    return M1, M2

class SplitAttentionLayer(Module):
    """
    (single head self attention)
    Generate Q, K, V matrices with dimensions n x d.
    Attention = Softmax(Q K^T) V
    """
    def __init__(self, n, d, n_head = 1):
        super(SplitAttentionLayer, self).__init__()
        
        # random init of Q, K, V
        self.n, self.d = n, d
        self.Q = torch.randn(n, d)
        self.K = torch.randn(n, d)
        self.V = torch.randn(n, d)

        # use torch.chunk to split tensors along dimension 0
        self.Q0, self.Q1 = split_matrix(self.Q)
        self.K0, self.K1 = split_matrix(self.K)
        self.V0, self.V1 = split_matrix(self.V)
        
        # Assign tensors to their devices
        device0, device1 = 'cuda:0', 'cuda:1'
        self.Q0 = self.Q0.to(device0)
        self.K0 = self.K0.to(device0)
        self.V0 = self.V0.to(device0)
        self.Q1 = self.Q1.to(device1)
        self.K1 = self.K1.to(device1)
        self.V1 = self.V1.to(device1)
        
        self.A1 = torch.zeros(n // 2, n).to(device0)
        self.A2 = torch.zeros(n // 2, n).to(device1)

    def _share_tensor(self, rank, send_tensor, recv_tensor):
        assert send_tensor.shape == recv_tensor.shape
        print(f"rank: {rank}")
        dist.broadcast(send_tensor, src=rank)
        # send_req = dist.isend(tensor = send_tensor, dst = 1-rank)

        # print(f'Rank {rank} started sending')
        # if rank == 0:
        #     dist.recv(self.K0[1], 1)
        # else:
        #     dist.recv(self.K1[0], 0)
        
        # print(f'Rank {rank} started receiving')
        # # print('Rank ', rank, ' has data shape ', recv_tensor.shape)
        # send_req.wait()
        # if rank == 0:
        #     print(f'self.K0[1] {self.K0[1]}')
        

    def forward(self):
        size = 2

        mp.set_start_method('spawn')
        processes = []
        
        print(f'K1 {self.K0}')
        print(f'K2 {self.K1}')
        
        p1 = mp.Process(target=init_process, args=(0, size, self.K0[0], self.K0[1], self._share_tensor))
        p1.start()
        processes.append(p1)
        
        p2 = mp.Process(target=init_process, args=(1, size, self.K1[1], self.K1[0], self._share_tensor))
        p2.start()
        processes.append(p2)
        
        for p in processes:
            p.join()

        print(f'K1 {self.K0}')
        print(f'K2 {self.K1}')

        return

def main():
    attn = SplitAttentionLayer(4, 2)
    attn.forward()


if __name__ == "__main__":
    main()
