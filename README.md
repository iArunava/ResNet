# ResNet

In this repository, I have duplicated the ResNet paper. Each of the modules of the ResNet is seperated in a file so users can utilize each
block of the resnet seperately and use the ResNet Blocks in their architectures. The link to the paper can be found here: https://arxiv.org/pdf/1512.03385.pdf

The Repository contains the code to create all 5 ResNet architectures:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

```
>>> # To get the ResNet18 use
>>> ResNet(nc, ResNetBlock2L, [2, 2, 2, 2])
>>>
>>> # To get the ResNet34 use
>>> ResNet(nc, ResNetBlock2L, [3, 4, 6, 3])
>>>
>>> # To get the ResNet50 use
>>> ResNet(nc, ResNetBlock3L, [3, 4, 6, 3])
>>>
>>> # To get the ResNet101 use
>>> ResNet(nc, ResNetBlock3L, [3, 4, 23 3])
>>>
>>> To get the ResNet152 use
>>> ResNet(nc, ResNetBlock3L, [3, 8, 36, 3])
```

## Pretrained Models

1. ResNet18 - CIFAR10 - 83.9% - [Download](https://drive.google.com/file/d/1JLZ5h15yF7e6QBzXrZrzEE6i98SrQylA/view?usp=sharing)

## References

1. Deep Residual Learning for Image Recognition He et al. [Paper](https://arxiv.org/pdf/1512.03385.pdf)
2. Identity Mappings in Deep Residual Networks He et al. [Paper](https://arxiv.org/pdf/1603.05027.pdf)

## License

The code in this repository is free to use and modify for both commercial and non-commercial use.
If possible, just refer back to this repository.
