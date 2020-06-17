# Sequential-GCN-for-Active-Learning
## Requirements:
python 3.6+

torch 1.0+

pip libraries: tqdm, sklearn, scipy, math

## Run:
```bash 
python main.py # it will start the AL framework for CIFAR-10 on UncertainGCN method over 5 stages of 1000 points
```
Please have a look over the config file before running. Also, check the args of the code.
CUDA-GPU implementation, not tested on CPU. Different random seed might produce different results.

## Active Learning methods implemented:
Active Learning for Convolutional Neural Networks: A Core-Set Approach: https://arxiv.org/pdf/1708.00489.pdf
Learning Loss for Active Learning: https://arxiv.org/pdf/1905.03677.pdf
Variational Adversial Active Learning: https://arxiv.org/pdf/1904.00370.pdf

