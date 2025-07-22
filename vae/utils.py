import os
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import imageio
import numpy as np
from config import Config


def save_reconstructed_image(model, dataloader, epoch, max_samples = 8):
    
    model.eval()
    os.makedirs(Config.reconstruction_save_path, exist_ok=True)
    
    with torch.no_grad():
        images = next(iter(dataloader))[0][:max_samples].to(Config.device)
        recon , _, _ = model(images)
        
        #Combine them into GRIDS
        grid = make_grid(recon, nrow=max_samples, normalize=True)
        file_path = os.path.join(Config.reconstruction_save_path, f"recon_epoch:{epoch}.png")
        save_image(grid, file_path)
        
    model.train()
    return grid


def plot_training_curve(loss_list, save_path = "saves/vae_loss_plot.png"):
    plt.figure(figsize=(8,5))
    plt.plot(loss_list, label = "Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
def interpolate_latents(z_start, z_end, steps):
    ratios = torch.linspace(0,1, steps).unsqueeze(1).to(z_start.device)
    z_interpolate = z_start * (1 - ratios) + z_end * ratios
    return z_interpolate

def create_interpolation_gif(decoder, latent_dim, steps, gif_save_path):
    decoder.eval()
    with torch.no_grad():
        z_start = torch.randn(1, latent_dim).to(next(decoder.parameters()).device)
        z_end = torch.randn(1, latent_dim).to(next(decoder.parameters()).device)
        
        z_interpolate = interpolate_latents(z_start, z_end, steps)
        imgs = decoder(z_interpolate).cpu()
        
        imgs = (imgs + 1) / 2
        grid_imgs = [make_grid(img.unsqueeze(0), nrow=1).permute(1,2,0).numpy() for img in imgs]
        grid_imgs = [(img * 255).astype(np.uint8) for img in grid_imgs]
        
        imageio.mimsave(gif_save_path, grid_imgs, fps = 5)
        
    
    