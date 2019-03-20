import torch
import torch.nn as nn

from .BNConv import BNConv

class ResidualBlock2L(nn.Module):
  
  def __init__(self, ic_conv, oc_conv, downsample, expansion=4, stride=2,
                     conv_first=True, relu_after_add=True):
    '''
    This class defines the Residual Basic Block with 2 Conv Layers

    Arguments;
    - ic_conv : # of input channels
    - oc_conv : # of output channels for the final conv layers
    - downsample : Whether this block is to downsample the input
    - expansion : The expansion for the channels
                  Default: 4
    - stride : if downsample is True then the specified stride is used.
               Default: 2
    - conv_first : Whether to apply conv before bn or otherwise.
                   Default: True
    '''
    
    super(ResidualBlock2L, self).__init__()
    
    assert(downsample == True or downsample == False)
    assert(relu_after_add == True or relu_after_add == False)
    self.downsample = downsample
    self.expansion = expansion
    self.relu_after_add = relu_after_add
    oc_convi = oc_conv // self.expansion
    
    stride = stride if self.downsample else 1
    
    # TODO: update this block for handling conv_first=False
    # Where the relu block needs to come before the BNConv Layer

    self.side = nn.Sequential(
                    BNConv(in_channels=ic_conv,
                                out_channels=oc_convi,
                                kernel_size=3,
                                padding=1,
                                stride=stride,
                                eps=2e-5,
                                momentum=0.9,
                                conv_first=True),

                    nn.ReLU(inplace=True),
            
                    BNConv(in_channels=oc_convi,
                                out_channels=oc_conv,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                eps=2e-5,
                                momentum=0.9,
                                conv_first=True)
                )

    if self.downsample:
        self.main = BNConv(in_channels=ic_conv,
                            out_channels=oc_conv,
                            kernel_size=3,
                            padding=1,
                            stride=stride,
                            eps=2e-5,
                            momentum=0.9,
                            conv_first=True)
    
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):
    '''
    '''
    
    xs = self.side(x)

    xm = self.main(x) if self.downsample else x
    
    x = xm + xs
    x = self.relu(x) if self.relu_after_add else x
    
    return x
