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
