import torch
import torch.nn as nn

class BNConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias=False, eps=1e-5, momentum=0.1, conv_first=True):
        '''
        '''

        super(BNConv, self).__init__()
        
        if conv_first:
            self.main = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              stride=stride,
                                              padding=padding,
                                              bias=bias),
                        
                        nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
                    )
        else:
            self.main = nn.Sequential(
                            nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum),

                            nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias)
                        )


    def forward(self, x):
        '''
        Method that defines the forward pass through the BNConv network.

        Arguments:
        - x : The input to the network

        Returns:
        - The output of the network BNConv
        '''

        x = self.main(x)

        return x
