import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import wandb
import os
import numpy as np

from config import Config
from model import  VAE
from discriminator import Discriminator
from utils import plot_training_curve, create_interpolation_gif

#--------------------Set them SEEDS----------------------
torch.manual_seed(Config.seed)

#----------------Initialize WandB-------------------------
wandb.init(
    project=Config.wandb_project,
    name="GAN_TRAINING_FIXED",
    config = {k:v for k,v in Config.__dict__.items() if not k.startswith("__")}
)

#-----------------Load The Dataset--------------------------
transform = transforms.Compose([
    transforms.Resize((Config.image_size , Config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])
train_dataset = datasets.CIFAR10(root = Config.data_path, train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size= Config.batch_size, shuffle = True)

#---------------------Initialize the Model----------------------


vae = VAE().to(Config.device)
vae.load_state_dict(torch.load("saves/checkpoints/vae_epoch_50.pt",map_location=Config.device))
decoder = vae.decoder
decoder.to(Config.device)

discriminator = Discriminator(
    in_channels= Config.in_channels,
    base_channels=Config.d_features
).to(Config.device)

#----------------------Optimizer & Loss Function---------------------
gen_optimizer = optim.Adam(decoder.parameters(), lr=Config.gan_lr, betas=(0.5,0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=Config.gan_lr, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

#-------------------------Training-----------------------------------
decoder.train()
discriminator.train()

for epoch in range(Config.gan_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch:[{epoch + 1}/{Config.gan_epochs}]")
    epoch_g_losses, epoch_d_losses = [], []
    
    for i, (real_imgs, _) in loop:
        
        real_imgs = real_imgs.to(Config.device)
        batch_size = real_imgs.size(0)
        
        #Define the Labels
        real_labels = torch.ones(batch_size,1, device=Config.device)
        fake_labels = torch.zeros(batch_size, 1, device=Config.device)
        
        #----------------Train Discriminator----------------
        z = torch.randn(batch_size, Config.latent_dim).to(Config.device)
        fake_imgs_detached = decoder(z).detach()
        
        real_preds = discriminator(real_imgs)
        fake_preds = discriminator(fake_imgs_detached)
        
        d_loss_real = criterion(real_preds, real_labels)
        d_loss_fake = criterion(fake_preds, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        disc_optimizer.zero_grad()
        d_loss.backward()
        disc_optimizer.step()
        
        #-------------------Train Generator-------------------------
        z = torch.randn(batch_size, Config.latent_dim).to(Config.device)
        fake_imgs = decoder(z)
        preds = discriminator(fake_imgs)
        
        g_loss = criterion(preds, real_labels)
        
        gen_optimizer.zero_grad()
        g_loss.backward()
        gen_optimizer.step()
        
        loop.set_postfix(D_Loss = d_loss.item(), G_Loss = g_loss.item())
        epoch_d_losses.append(d_loss.item())
        epoch_g_losses.append(g_loss.item())

    # Calculate average losses for the epoch
    avg_d_loss = np.mean(epoch_d_losses)
    avg_g_loss = np.mean(epoch_g_losses)

    #-------------------WandB Logging & Saving Samples-----------------
    log_dict = {
        'Epoch': epoch + 1,
        'Discriminator_Loss': avg_d_loss,
        'Generator_Loss': avg_g_loss,
    }

    if (epoch + 1) % Config.sample_interval == 0:
        os.makedirs(Config.sample_dir, exist_ok=True)
        sample_path = os.path.join(Config.sample_dir, f"sample_at_epoch_{epoch+1}.png")
        
        # Use the latest generated images for sampling
        save_image(fake_imgs[:16], sample_path, nrow=4, normalize=True)
        fake_grid = make_grid(fake_imgs[:16], nrow=4, normalize=True)
        real_grid = make_grid(real_imgs[:16], nrow=4, normalize=True)
        
        gif_save_path = os.path.join(Config.sample_dir, f"interpolation_epoch_{epoch + 1}.gif")
        create_interpolation_gif(decoder, Config.latent_dim, steps=10, gif_save_path=gif_save_path)
        
        log_dict.update({
            "Sample_Image_Fake": wandb.Image(fake_grid, caption=f"Fake_Samples_Generated_at_Epoch_{epoch + 1}"),
            "Sample_Image_Real": wandb.Image(real_grid, caption=f"Real_Samples_at_Epoch_{epoch + 1}"),
        })
        
        if os.path.exists(gif_save_path):
            log_dict.update({
                "Latent Interpolation": wandb.Video(gif_save_path, format="gif"),
            })

    wandb.log(log_dict)
    
    #--------------------Save Checkpoint-------------------------------
    if (epoch + 1) % Config.checkpoint_interval == 0:
        checkpoint_dir = "./saves/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch + 1,
            "decoder_state_dict": decoder.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "gen_optimizer_state_dict": gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_dir}/gan_checkpoint_epoch_{epoch+1}.pt")
           
wandb.finish()
