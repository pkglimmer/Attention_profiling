import colossalai
import colossalai.nn as col_nn
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device, print_rank_0

# https://github.com/hpcaitech/ColossalAI-Examples/commit/936c73932d20bb180ca3b7ebd4806bb369e0815e
# (Peikun) To run this script, use command:
# torchrun --nproc_per_node=4 colossal_2d.py --from_torch

CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=4, mode='2d'),
))


class MLP(torch.nn.Module):

    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)
        print_rank_0(f'Weight of the first linear layer: {self.dense_1.weight.shape}')
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        print_rank_0(f'Weight of the second linear layer: {self.dense_2.weight.shape}')
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        print_rank_0(f'Output of the first linear layer: {x.shape}')
        x = self.activation(x)
        x = self.dense_2(x)
        print_rank_0(f'Output of the second linear layer: {x.shape}')
        x = self.dropout(x)
        return x


def main():
    colossalai.logging.disable_existing_loggers()
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    if args.from_torch:
        colossalai.launch_from_torch(config=CONFIG)
    else:
        # standard launch
        colossalai.launch(config=CONFIG,
                          rank=args.rank,
                          world_size=args.world_size,
                          local_rank=args.local_rank,
                          host=args.host,
                          port=args.port)

    m = MLP()

    x = torch.randn((16, 256), device=get_current_device())
    
    print(f'Debug\n get_current_device(): {get_current_device()} \
        \n gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL): {gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)}\
        \n gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW): {gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)}\n')
    # partition input
    torch.distributed.broadcast(x, src=0)
    x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)]
    x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)]
    print_rank_0(f'Input: {x.shape}')

    x = m(x)
    print(f'Final output: {x.shape}')


if __name__ == '__main__':
    main()