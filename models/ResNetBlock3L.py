class ResNetBlock3L(nn.Module):
  def __init__(self, ic_conv1, oc_conv1, downsample, stride=2):
    '''
    '''
    
    super().__init__()
    
    assert (downsample == True or downsample == False)
    self.downsample = downsample
    
    if self.downsample:
      stride = stride
    else:
      stride = 1
    
    self.conv1 = BNConv(in_channels=ic_conv1,
                         out_channels=oc_conv1,
                         kernel_size=1,
                         padding=0,
                         stride=1)
    
    self.conv2 = BNConv(in_channels=oc_conv1,
                        out_channels=oc_conv1,
                        kernel_size=3,
                        padding=1,
                        stride=stride)
    
    self.conv3 = BNConv(in_channels=oc_conv1,
                        out_channels=oc_conv1*4,
                        kernel_size=1,
                        padding=0,
                        stride=1)
    
    self.convs = BNConv(in_channels=ic_conv1,
                        out_channels=oc_conv1*4,
                        kernel_size=3,
                        padding=1,
                        stride=stride)
      
    self.relu = nn.ReLU()
    
  def forward(self, x):
    '''
    '''
    
    xm = self.relu(self.conv1(x))
    xm = self.relu(self.conv2(xm))
    xm = self.conv3(xm)
    
    xs = self.convs(x) if self.downsample else x
    
    x = xm + xs
    x = self.relu(x)

    return x
