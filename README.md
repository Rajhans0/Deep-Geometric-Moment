# Code for paper Improving Shape Awareness and Interpretability in Deep Networks Using Geometric Moments
# Accepted at CVPR 2023 Workshop: Deep Learning for Geometric Computing

# Deep-Geometric-Moment ResNet-18 code for cifar datasets
# model.py contains the implementation of DGM-ResNet-18 model

# Part of the code is taken from the pytorch implementation of resnet model by Wei YANG, can be found here: https://github.com/bearpaw/pytorch-classification.git



#need following library to run the code:
1) Python
2) torch
3) torchvision
4) numpy
5) pillow
6) scikit-learn
7) tqdm
8) matplotlib

# Example command to run the code:

python main.py -c checkpoints/cifar10/chkpt -d cifar10 --lr 0.1 --epoch 150 --train-batch 128 --test-batch 100 --gpu-id 0,1
