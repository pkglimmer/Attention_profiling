import torch

g = torch.randn(16, 8, 8, 8)
k = torch.randn(16, 4, 512, 16)
v = torch.randn(16, 4, 512, 16)




4
Input Linear: 0.5869
Gather: 3.9454
QKT: 1.6261
Softmax: 1.6946
SA: 2.1470
Output Linear: 0.1173


Input Linear: 0.6325
Gather: 2.2106
QKT: 3.1234
Softmax: 3.3224
SA: 3.6545
Output Linear: 0.1675