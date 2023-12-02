# Test scripts (10.15)

vanilla: /home/yuxin/code/attention.py
linear layer matmul parallel: /home/yuxin/code/Titans/tests/test_layer/test_attention/test_inputW_parallel.py
Ring Attention (somehow very slow): /home/yuxin/code/ColossalAI/tests/test_layers/test_sequence/checks_seq/check_layer.py

Useful implementations:
Tensor Parallel:
    - /home/yuxin/anaconda3/envs/pk/lib/python3.10/site-packages/colossalai/nn/layer/parallel_1d/layers.py
    - (matmul 2d) /home/yuxin/anaconda3/envs/pk/lib/python3.10/site-packages/colossalai/nn/layer/parallel_2d/_operation.py
    - (split forward, gather backward) /home/yuxin/anaconda3/envs/pk/lib/python3.10/site-packages/colossalai/nn/layer/parallel_1d/_utils.py
Ring:
    - (Transformer Attention Ring) /home/yuxin/anaconda3/envs/pk/lib/python3.10/site-packages/colossalai/nn/layer/parallel_sequence/layers.py
    - (RingQK, RingAV) /home/yuxin/anaconda3/envs/pk/lib/python3.10/site-packages/colossalai/nn/layer/parallel_sequence/_operation.py


# Implementation by steps

The general structure of the Attention layer (nn.Module) follows the original implementation.

0. Compute Q, K, V by forward passing input x through the first linear layer. scatter Q, K, V to gpus.

2. (start timing) All gather K, V. Ai = Softmax(Qi x K.T)

3. Oi = Ai x V

# Vanilla Attention Distributed Implementation (11.21.2022)

## scripts
* profiling script path: /home/yuxin/code/Titans/tests/test_layer/test_attention/profile_vanilla_parallel.py
* /home/yuxin/code/Titans/titans/layer/attention/vit_attention.py - `ViTSelfAttention_Parallel` class

## logs
/home/yuxin/code/taylor_logs/vanilla_attention_summary.csv



# Taylor Attention Distributed Implementation (11.18.2022)

## Implementation

Some thoughts about the implementation details are listed below.

0. Code

* Taylor attention, distributed: 
    - Raw, without timing: /home/yuxin/code/taylor_scripts/taylor_layer.py 
    - Profiling with time.time(): 
Profiling script: 

1. `Linear1D_row` in colossalai layers

In the `Linear1D_row` class, the linear layer weight size is in accord with the partitioned input (across row dimension). However, `Linear1D_row` is a layer (torch.nn module), which can serve as a component for executing step 0 (Q, K, V calculations). The later steps would no longer require layers/modules, as we would directly apply the communication operations in our newly defined taylor attention module.

2. Profiler

https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md
https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

## Result 

* Strong scaling is defined as how the solution time varies with the number of processors for a **fixed total problem size**. 
* Weak scaling is defined as how the solution time varies with the number of processors for a **fixed problem size per processor**.


# profiler:
https://colab.research.google.com/drive/1kAzv1TWUFRU5HxLsJGVMiBd6jtGzPGAQ?usp=sharing


# Default settings

* ViT
    - Sequence length (n) = 512
    - HIDDEN_SIZE = 64
    - NUM_HEADS ()

* Bert
    - DEPTH = 12
    - NUM_ATTENTION_HEADS = 12
    - HIDDEN_SIZE = 768
    - /home/yuxin/code/ColossalAI-Examples/language/bert/sequene_parallel/config.py


* Timing
    - https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1
    - or just use shell `time` command
    - tqdm progressbar



# References 

1. Attention and visualizations: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
2. (**timeit** included) Single-Machine Model Parallel Best Practices: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
3. All gather examples:
    - SlowFasthttps://github.com/facebookresearch/SlowFast/blob/adc55ca38a62a8bf3899a2d7c2dfbc9f9eee1ade/slowfast/models/contrastive.py
    - https://github.com/facebookresearch/pytorchvideo/blob/825e8aa9430e6dc431c9b6548355820ba08dbf09/pytorchvideo_trainer/pytorchvideo_trainer/module/distributed_utils.py
4. Parallelism from HugginFace: https://huggingface.co/docs/transformers/perf_train_gpu_many#naive-model-parallelism-vertical-and-pipeline-parallelism









