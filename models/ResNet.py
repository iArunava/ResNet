from .ResNetBlock2L import ResNetBlock2L
from .ResNetBlock3L import ResNetBlock3L
from .ResNetBlock import ResNetStage

class ResNet(nn.Module):
  
  def __init__(self, nc, block, layers, s1_channels=64):
    '''
    The class that defines the ResNet module
    
    Arguments:
    - s1_channels : # of channels for the output of the first stage
    
    '''
    super(ResNet, self).__init__()
    
    # Stage 1
    self.conv1 = BNConv(in_channels=3,
                        out_channels=s1_channels,
                        kernel_size=7,
                        padding=3,
                        stride=2)
    
    # Stage 2
    self.maxpool = nn.MaxPool2d(kernel_size=2,
                                stride=2)
    
    self.stage2 = ResNetBlock(block, ic_conv1=s1_channels,
                              oc_conv1=s1_channels,
                              num_layers=layers[0],
                              stride=1)
    
    self.stage3 = ResNetBlock(block, ic_conv1=s1_channels*4,
                              oc_conv1=s1_channels*2,
                              num_layers=layers[1])
    
    self.stage4 = ResNetBlock(block, ic_conv1=s1_channels*8,
                              oc_conv1=s1_channels*4,
                              num_layers=layers[2])
    
    self.stage5 = ResNetBlock(block, ic_conv1=s1_channels*16,
                              oc_conv1=s1_channels*8,
                              num_layers=layers[3])
    
    self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    
    self.fc = nn.Linear(2048, 1000)
    
    self.out = nn.Linear(1000, nc)
    
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x):
    '''
    '''
    
    batch_size = x.size(0)
    
    x = self.conv1(x)
    x = self.maxpool(x)
    print ('pool', x.shape)
    x = self.stage2(x)
    print ('2', x.shape)
    x = self.stage3(x)
    print (3, x.shape)
    x = self.stage4(x)
    print (4, x.shape)
    x = self.stage5(x)
    print (5, x.shape)
    x = self.avgpool(x)    
    print ('avg', x.shape)
    
    x = x.view(batch_size, -1)
    print (x.shape)
    
    x = self.fc(x)
    x = self.out(x)
    out = self.softmax(x)
    
    return out
