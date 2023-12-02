import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Module

n, d = 4, 2
Q = torch.randn(n, d)
K = torch.randn(n, d)
V = torch.randn(n, d)
assert n % 2 == 0

# step 0: partition x, compute Qi Ki Vi submatrices
#         (XW_Q, XW_K, XW_V)


# step 1: 
def worker_qk(rank, queue):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    print(f'rank: {rank}')
    start, end = rank * (n//2), (rank+1) * (n//2)  # 0: 0:n//2, 1: n//2:n
    Q_ = Q[start:end, :]
    Q_ = Q_.to(f'cuda:{rank}')
    K_ = K[start:end, :]
    K_ = K_.to(f'cuda:{rank}')
    Q_K_ = torch.matmul(Q_, K_.T)

    print(f"rank{rank}, Q_K_: \n{Q_K_}")

    queue.put(Q_K_)
    time.sleep(3)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    result_queue = mp.Queue()
    world_size = 2
    for rank in range(world_size):
        mp.Process(target=worker_qk, args=(rank, result_queue)).start()

    # print(f"{'='*10} results {'='*10}")
    for rank in range(world_size):
        temp = result_queue.get()
        print(f"\n\n(main process){rank}rank, QK^T:\n{temp}")
        del temp
