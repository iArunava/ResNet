class ResNetBlock(nn.Module):
  
  def __init__(self, block, ic_conv1, oc_conv1, num_layers, downsample):
    '''
    '''
    super(ResNetBlock,self).__init__()
    
    self.layers = []
    
    if downsample:
        layers.append(block(ic_conv1=ic_conv1,
                            oc_conv1=oc_conv1,
                            downsample=True))
        num_layers -= 1
        
    for _ in range(num_layers):
      layers.append(block(ic_conv1=ic_conv1,
                          oc_conv2=oc_conv2,
                          downsample=False))
    
    self.stage = nn.Sequential(*layers)
    
  def forward(self, x):
    '''
    '''
    
    x = self.stage(x)
    return x
