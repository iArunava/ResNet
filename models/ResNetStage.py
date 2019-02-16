class ResNetBlock(nn.Module):
  
  def __init__(self, block, ic_conv1, oc_conv1, num_layers, stride=2):
    '''
    '''
    super(ResNetBlock,self).__init__()
    
    self.layers = []
    
    
    self.layers.append(block(ic_conv1=ic_conv1,
                        oc_conv1=oc_conv1,
                        downsample=True,
                        stride=stride))
    
    num_layers -= 1
    #ic_conv1 = ic_conv1*4
    for _ in range(num_layers):
      self.layers.append(block(ic_conv1=oc_conv1*4,
                          oc_conv1=oc_conv1,
                          downsample=False))
    
    self.stage = nn.Sequential(*self.layers)
    
  def forward(self, x):
    '''
    '''
    
    x = self.stage(x)
    return x
