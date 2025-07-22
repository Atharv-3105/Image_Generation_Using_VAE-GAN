import os
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import wandb
from tqdm import tqdm
from torchvision.utils import save_image

from config import Config
from vae.model import VAE
from vae.utils import save_reconstructed_image, plot_training_curve


#---------------------Set Device & SEED-----------------------------
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(Config.seed)

#----------------------Initialize WandB & Model------------------------
wandb.init(project=Config.wandb_project,
           name = f"{Config.wandb_run_name}_2",
           config = {k:v for k,v in Config.__dict__.items() if not k.startswith("__")})

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
reconstruction_criterion = nn.MSELoss(reduction="sum")

#---------------------------Define Loss Function-------------------------
def loss_fn(recon_x, x, mu, logvar):
    recon_loss = reconstruction_criterion(recon_x, x)
    kl_divergence = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_divergence
    
    return total_loss, recon_loss, kl_divergence

#----------------------------Load Dataset & Make DataLoader-----------------
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root = Config.dataset_path, train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle = True)


#--------------------------Training Loop---------------------------------
os.makedirs("saves/reconstructions", exist_ok=True)
os.makedirs("saves/checkpoints", exist_ok=True)

total_loss_list = []

model.train()
for epoch in range(1, Config.num_epochs + 1):
    epoch_loss = 0
    
    loop = tqdm(train_loader, desc = f"Epoch: [{epoch}/{Config.num_epochs}]", leave=False)
    
    for x, _ in loop:
        x = x.to(device)
        optimizer.zero_grad()
        
        recon_x , mu , logvar = model(x)
        loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        loop.set_postfix(loss = loss.item() / x.size(0))
        
        wandb.log(
            {
                "Batch Loss": loss.item() / x.size(0),
                "Reconstruction Loss": recon_loss.item() / x.size(0),
                "KL Loss": kl_loss.item() / x.size(0),
            }
        )
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    total_loss_list.append(avg_loss)
    
    wandb.log({"Epoch Loss": avg_loss, "Epoch":epoch})
    
    #================Save Reconstruction==================
    if epoch % Config.save_reconstruction_interval == 0:
        model.eval()
        with torch.no_grad():
            x , _ = next(iter(train_loader))
            x = x.to(device)[:8]
            recon_x, _, _ = model(x)
            compare = torch.cat([x, recon_x])
            save_image(compare.cpu(), f"saves/reconstructions/recon_epoch:{epoch}.png", nrow = 8)
            
    #===============Save Checkpoint=======================
    if epoch % 25 == 0:
        checkpoint = {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
        }
    torch.save(checkpoint, f"saves/checkpoints/vae_epoch:{epoch}.pt")
    
plot_training_curve(total_loss_list)
wandb.save("saves/vae_training_loss.png")
        

        

