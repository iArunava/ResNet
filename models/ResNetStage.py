import torch
import torch.nn as nn

from .ResidualBlock import ResidualBlock

class ResNetBlock(nn.Module):
  
  def __init__(self, block:nn.Module, ic_conv:int, oc_conv:int, num_layers:int, stride:int=2):
    '''
    '''
    super(ResNetBlock,self).__init__()
    
    self.rblocks = nn.ModuleList([block(ic_conv=ic_conv,
                                        oc_conv=oc_conv,
                                        stride=stride)])
    
    self.rblocks.extend([block(ic_conv=oc_conv,
                                oc_conv=oc_conv) for i in range (num_layers-1)])
    
  def forward(self, x:torch.Tensor):
    for block in self.rblocks:
        x = block(x)
    return x
