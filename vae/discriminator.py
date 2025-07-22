import torch
import torch.nn as nn
from config import Config

class Discriminator(nn.Module):
    def __init__(self, in_channels = Config.in_channels, base_channels = Config.d_features):
        super().__init__()
        
        
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=4, stride=2, padding=1, bias=False),  #Shape: [batch_dim,img_channels,32,32]---->[batch_dim,base_channels,16,16]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=4, stride=2, padding=1, bias=False),  #Shape: [batch_dim,base_channels,16,16]------>[batch_dim,base_channels*2,8,8]
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=4, stride=2, padding=1, bias=False),  #Shape: [batch_dim,base_channels*2, 8,8]----->[batch_dim,base_channels*4, 4, 4]
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            #Final Convolution which will map base_channels*4 to Binary_Output
            nn.Conv2d(in_channels=base_channels*4, out_channels=1, kernel_size=4, stride=1, bias=False),
        )
        
    def forward(self, x):
        #view() is used to Reshape the image from [batch_dim, 1,1,1]---->[batch_dim, 1]
        out = self.discriminator(x)
        return out.view(-1)