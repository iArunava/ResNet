import torch
import torch.nn as nn

from .BNConv import BNConv

class ResidualBlock(nn.Module):
  
  def __init__(self, inc, outc, downsample, expansion=4, stride=2,
                     conv_first=True, relu_after_add=True, num_conv=2):
    '''
    This class defines the Residual Basic Block with 2 Conv Layers

    Arguments;
    - inc :: # of input channels
    - outc :: # of output channels for the final conv layers
    - downsample :: Whether this block is to downsample the input
    - expansion :: The expansion for the channels
                  Default: 4
    - stride :: if downsample is True then the specified stride is used.
               Default: 2
    - conv_first :: Whether to apply conv before bn or otherwise.
                   Default: True
    
    - num_conv :: # of conv layers in the block for the main branch
    '''
    
    super(ResidualBlock2L, self).__init__()
    
    assert(downsample == True or downsample == False)
    assert(relu_after_add == True or relu_after_add == False)
    self.downsample = downsample
    self.expansion = expansion
    self.relu_after_add = relu_after_add
    oc_convi = outc // self.expansion
    
    stride = stride if self.downsample else 1
    
    layers = [BNConv(in_channels=inc,
                        out_channels=oc_convi,
                        kernel_size=3,
                        padding=1,
                        stride=stride,
                        eps=eps,
                        momentum=momentum,
                        conv_first=conv_first)]
    
    for _ in range(num_conv-2):
        layers += [BNConv(in_channels=oc_convi,
                            out_channels=oc_convi,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            eps=eps,
                            momentum=momentum,
                            conv_first=conv_first)]
    
    layers += [BNConv(in_channels=oc_convi,
                        out_channels=outc,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        eps=eps,
                        momentum=momentum,
                        conv_first=conv_first)]
    
    self.main = nn.Sequential(*layers)

    if self.downsample:
        self.side = BNConv(in_channels=inc,
                            out_channels=outc,
                            kernel_size=3,
                            padding=1,
                            stride=stride,
                            eps=2e-5,
                            momentum=0.9,
                            conv_first=conv_first)
    
    if self.relu_after_add:
        self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):

    x = self.main(x) + self.side(x) if self.downsample else x
    return self.relu(x) if self.relu_after_add else x
