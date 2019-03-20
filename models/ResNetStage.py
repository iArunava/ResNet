import torch
import torch.nn as nn

from .ResidualBlock import ResidualBlock

class ResNetBlock(nn.Module):
  
  def __init__(self, block:nn.Module, inc:int, outc:int, num_layers:int, 
                     stride:int=2, conv_first=False, inplace=False):
    '''
    '''
    super(ResNetBlock,self).__init__()
    
    self.rblocks = nn.ModuleList([block(inc=inc, outc=outc, stride=stride, 
                                        conv_first=conv_first, inplace=inplace)])
    
    self.rblocks.extend([block(inc=outc, outc=outc,
                                conv_first=conv_first, inplace=inplace) 
                                for i in range (num_layers-1)])
    
  def forward(self, x:torch.Tensor):
    for block in self.rblocks:
        x = block(x)
    return x
