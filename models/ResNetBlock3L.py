import torch
import torch.nn as nn

from .BNConv import BNConv

class ResidualBlock3L(nn.Module):
  def __init__(self, ic_conv, oc_conv, downsample, expansion=4, stride=2,
                     conv_first=True):
    '''
    '''
    
    super(ResidualBlock3L, self).__init__()
    
    assert (downsample == True or downsample == False)
    self.downsample = downsample
    self.expansion = expansion
    oc_convi = oc_conv // self.expansion
    
    stride = stride if self.downsample else 1
    
    # TODO: update this block for handling conv_first=False
    # Where the relu block needs to come before the BNConv Layer
    self.side = nn.Sequential (
                    BNConv(in_channels=ic_conv,
                             out_channels=oc_convi,
                             kernel_size=1,
                             padding=0,
                             stride=1,
                             eps=2e-5,
                             momentum=0.9,
                             conv_first=True),

                    nn.ReLU(inplace=True),
                    
                    BNConv(in_channels=oc_convi,
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
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            eps=2e-5,
                            momentum=0.9,
                            conv_first=True)
                )
    
    self.relu = nn.ReLU(inplace=True)

    if self.downsample:
        self.main =  BNConv(in_channels=ic_conv,
                                out_channels=oc_conv*self.expansion,
                                kernel_size=3,
                                padding=1,
                                stride=stride,
                                eps=2e-5,
                                momentum=0.9,
                                conv_first=True)

    
  def forward(self, x):
    '''
    This method defines the forward pass for the Residual Block
    with 3 layers.

    Arguments:
    - x : The input to the network

    Returns:
    - The output of the network
    '''
    
    xs = self.side(x)
    xm = self.main(x) if self.downsample else x
    
    x = xm + xs
    x = self.relu(x)

    return x
