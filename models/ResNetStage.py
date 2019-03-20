import torch
import torch.nn as nn

from .ResNetBlock2L import ResNetBlock2L
from .ResNetBlock3L import ResNetBlock3L

class ResNetBlock(nn.Module):
  
  def __init__(self, block, ic_conv, oc_conv, num_layers, stride=2):
    '''
    '''
    super(ResNetBlock,self).__init__()
    
    self.rblocks = nn.ModuleList([block(ic_conv=ic_conv,
                                        oc_conv=oc_conv,
                                        stride=stride)])
    
    self.rblocks.extend([block(ic_conv=oc_conv,
                                oc_conv=oc_conv) for i in range (num_layers-1)])
    
  def forward(self, x):
    for block in self.rblocks:
        x = block(x)
    return x
