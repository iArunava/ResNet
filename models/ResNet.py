from .ResNetBlock2L import ResNetBlock2L
from .ResNetBlock3L import ResNetBlock3L
from .ResNetBlock import ResNetStage

class ResNet(nn.Module):
  
  def __init__(self, nc, block, layers, s1_channels=64, apool=True):
    '''
    The class that defines the ResNet module
    
    Arguments:
    - s1_channels : # of channels for the output of the first stage

    '''
    super(ResNet, self).__init__()
    
    self.apool = apool
    
    self.conv1 = BNConv(in_channels=3,
                        out_channels=s1_channels,
                        kernel_size=7,
                        padding=3,
                        stride=2,
                        eps=2e-5,
                        momentum=0.9,
                        conv_first=True)
    
    # Stage 2
    self.maxpool = nn.MaxPool2d(kernel_size=3,
                                stride=2)
    
    self.stage2 = ResNetBlock(block, ic_conv=s1_channels,
                              oc_conv=s1_channels*4,
                              num_layers=layers[0],
                              stride=1)
    
    self.stage3 = ResNetBlock(block, ic_conv=s1_channels*4,
                              oc_conv=s1_channels*8,
                              num_layers=layers[1])
    
    self.stage4 = ResNetBlock(block, ic_conv=s1_channels*8,
                              oc_conv=s1_channels*16,
                              num_layers=layers[2])
    
    self.stage5 = ResNetBlock(block, ic_conv=s1_channels*16,
                              oc_conv=s1_channels*32,
                              num_layers=layers[3])
    
    if self.apool:
      self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    
    self.fc = nn.Linear(2048, nc)
    
    #'''
    for module in self.modules():
      #if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.xavier_normal_(module.weight, gain=0.02)
      if isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        #nn.init.constant_(module.bias, 0)
    #'''
    
  def forward(self, x):
    '''
    '''
    
    batch_size = x.size(0)
    
    x = self.conv1(x)
    #print (x.shape)
    x = self.maxpool(x)
    #print (x.shape)
    x = self.stage2(x)
    #print (x.shape)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)
    #print (x.shape)
    x = self.avgpool(x) if self.apool else x
    
    x = x.view(batch_size, -1)
    
    x = self.fc(x)
    
    return x
