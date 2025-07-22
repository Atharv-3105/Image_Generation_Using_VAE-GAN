import torch
import torch.nn as nn
import torch.nn.functional as F 
from vae.config import Config


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=Config.in_channels, out_channels=Config.encoder_channels[0], kernel_size=Config.kernel_size,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(Config.encoder_channels[0]),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=Config.encoder_channels[0], out_channels=Config.encoder_channels[1],kernel_size=Config.kernel_size,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(Config.encoder_channels[1]),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=Config.encoder_channels[1], out_channels=Config.encoder_channels[2], kernel_size=Config.kernel_size,
                      stride=2, padding=1),
            nn.BatchNorm2d(Config.encoder_channels[2]),
            nn.ReLU(True),
             
        )
        
        self.flatten = nn.Flatten(),
        self.mu = nn.Linear(in_features=Config.encoder_channels[2] * 4 * 4, out_features=Config.latent_dim)
        self.logvar = nn.Linear(in_features=Config.encoder_channels[2] * 4 * 4, out_features=Config.latent_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=Config.latent_dim, out_features=Config.decoder_channels[0]*4*4)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=Config.decoder_channels[0], out_channels=Config.decoder_channels[1],
                               kernel_size=Config.kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(Config.decoder_channels[1]),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(in_channels=Config.decoder_channels[1], out_channels=Config.decoder_channels[2],
                               kernel_size=Config.kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(Config.decoder_channels[2]),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=Config.decoder_channels[2], out_channels=Config.out_channels,
                               kernel_size=Config.kernel_size, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z) #[B,latent_dim]------>[B,128*4*4] Converts the input to a 1-D tensor
        x = x.view(-1, Config.decoder_channels[0], 4, 4) #Reshapes the Input_tensor back to 4-D tensor
        x = self.deconv(x)
        return x



class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed , mu , logvar