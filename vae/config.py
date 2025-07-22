import torch

class Config():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    
    dataset_name = "CIFAR-10"
    image_size = 32
    in_channels = 3
    out_channels = 3
    
    encoder_channels = [32,64,128]
    decoder_channels = [128,64,32]
    kernel_size = 4

    #VAE_HYPERPARAMETERS
    latent_dim = 128
    hidden_dim = 512
    
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 50
    
    #GAN_HYPERPARAMETER
    d_features = 64
    gan_lr = 2e-4
    gan_epochs = 50
    checkpoint_interval = 10
    sample_interval = 5
    
    #For Saving
    log_interval = 100
    save_reconstruction_interval = 2
    data_path = "./data"
    reconstruction_save_path ="./saves/reconstructions"
    model_save_path = "./saves/checkpoints" 
    dataset_path = "./data" 
    sample_dir = "./saves/gan_samples"
    
    #WandB_Config
    wandb_project = "VAE-CIFAR-10"
    wandb_run_name = "vae_run"
    
    