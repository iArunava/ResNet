import torch
import torch.nn as nn

class BNConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias=False, eps=1e-5, momentum=0.1, conv_first=True, 
                 act='relu', nslope=0.2, norm='instance', inplace=True):
        
        '''
        This is the BNConv module that helps in defining the 
        Conv->Norm->RelU
        RelU->Norm->Conv
        
        Arguments:
        - in_channels = # of input channels
        - out_channels = # of output channels
        - kernel_size = the kernel_size
        - stride - the stride
        - padding - the padding
        - bias = whether to add bias to Convolutional Layer
        - eps = the epsilon value
        - momentum = the momentum value
        - conv_first = If True: conv->norm->act
                       If False: act->norm->conv
                       Default: True
        - nslope = the slope if leaky relu is used.
        - norm = the norm type
        - inplace = The value for the inplace argument inactivation
                    Default: True
        '''

        super(BNConv, self).__init__()
        
        act = act.lower()
        norm = norm.lower()
        
        if act == 'relu':
            act = nn.ReLU(inplace=inplace)
        elif act == 'leakyrelu':
            act = nn.LeakyReLU(negative_slope=nslope, inplace=inplace)
        else:
            raise ('Activation value not understood')

        if norm == 'batch':
            norm = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        elif norm == 'instance':
            norm = nn.InstanceNorm2d(out_channels, eps=eps, momentum=momentum)
        else:
            raise ('Norm value not understood')
        
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=bias),
                    
                  norm,
                  act]

        self.net = nn.Sequential(*layers) if conv_first else nn.Sequential(*layers[::-1])
            
    def forward(self, x): return self.net(x)
