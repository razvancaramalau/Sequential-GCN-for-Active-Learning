# Sequential GCN for Active Learning
[Link to paper.](https://arxiv.org/pdf/2006.10219.pdf)
## Requirements:
python 3.6+

torch 1.0+

pip libraries: tqdm, sklearn, scipy, math

## Run:
For running UncertainGCN on CIFAR-10 over 5 sampling stages of 1000 images:
```bash 
python main.py -m UncertainGCN -d cifar10 -c 5 # Other available datasets cifar100, fashionmnist, svhn
```
CoreGCN The geometric method that uses GCN training can be run 
Please have a look over the config file before running. Also, check the args of the code.
CUDA-GPU implementation, not tested on CPU. Different random seed might produce different results.

## Active Learning methods implemented:
Active Learning for Convolutional Neural Networks: A Core-Set Approach: https://arxiv.org/pdf/1708.00489.pdf

Learning Loss for Active Learning: https://arxiv.org/pdf/1905.03677.pdf

Variational Adversial Active Learning: https://arxiv.org/pdf/1904.00370.pdf

## Contact
If there are any questions or concerns feel free to send a message at: r.caramalau18@imperial.ac.uk
