# LMQFormer: A Laplace-Prior-Guided Mask Query Transformer for Lightweight Snow Removal <br> (Accepted by IEEE TCSVT)

<img src=".\img\result.png" alt="result" style="zoom:50%;" />


Qualitative results on synthetic datasets. From top to bottom rows: Snow100K, SRRS, CSD, SnowKitti2012 and SnowCityScapes. Please zoom in for better visual quality.

# Abstract:

Snow removal aims to locate snow areas and recover clean images without repairing traces. Unlike the regularity and semitransparency of rain, snow with various patterns and degradations seriously occludes the background. As a result, the state-of-the-art snow removal methods usually retains a large parameter size. In this paper, we propose a lightweight but high-efficient snow removal network called Laplace Mask Query Transformer (LMQFormer). Firstly, we present a Laplace-VQVAE to generate a coarse mask as prior knowledge of snow. Instead of using the mask in dataset, we aim at reducing both the information entropy of snow and the computational cost of recovery. Secondly, we design a Mask Query Transformer (MQFormer) to remove snow with the coarse mask, where we use two parallel encoders and a hybrid decoder to learn extensive snow features under lightweight requirements. Thirdly, we develop a Duplicated Mask Query Attention (DMQA) that converts the coarse mask into a specific number of queries, which constraint the attention areas of MQFormer with reduced parameters. Experimental results in popular datasets have demonstrated the efficiency of our proposed model, which achieves the state-of-the-art snow removal quality with significantly reduced parameters and the lowest running time.

[[Paper Download]]([LMQFormer: A Laplace-Prior-Guided Mask Query Transformer for Lightweight Snow Removal | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10092769))

You can also refer our works on other low-level vision applications!

[LDRM: Degradation Rectify Model for Low-light Imaging via Color-Monochrome Cameras | Proceedings of the 31st ACM International Conference on Multimedia](https://dl.acm.org/doi/abs/10.1145/3581783.3613792)


# Network Architecture

<img src=".\img\network.png" alt="network" style="zoom:50%;" />


# Setup and environment

#### To generate the recovered result you need:

1. Python 3.7
2. CPU or NVIDIA GPU + CUDA CuDNN
3. Pytorch 1.8.0
4. python-opencv

#### Testing

We trained our model in five snow removal datasets, including Snow100K, SRRS, CSD, SnowKitti2012 and SnowCityScapes.

Please replace weights_dir data_dir and result_dir in test.py, and put your testset in data_dir.

#### Pre-trained model
It can be downloaded from：

Link: https://pan.baidu.com/s/1ED2hIoAiPciAHAqDt_pNMg 

Extract code: LMQF


# Citations
Please cite this paper in your publications if it is helpful for your tasks:    

Bibtex:
```
@ARTICLE{10092769,
  author={Lin, Junhong and Jiang, Nanfeng and Zhang, Zhentao and Chen, Weiling and Zhao, Tiesong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={LMQFormer: A Laplace-Prior-Guided Mask Query Transformer for Lightweight Snow Removal}, 
  year={2023},
  volume={33},
  number={11},
  pages={6225-6235},
  doi={10.1109/TCSVT.2023.3264824}}

```
