# Sequential-GCN-for-Active-Learning
Requirements:
python 3.6+
torch 1.0+
pip3 libraries: tqdm, sklearn, scipy, math

Run:
python main.py # it will start the AL framework for CIFAR-10 on UncertainGCN method over 5 stages of 1000 points

Please have a look over the config file before running. Also, check the args of the code.
CUDA-GPU implementation, not tested on CPU. Different random seed might produce different results.
