import torch 
import random
from vae.model import VAE
from vae.config import Config
from torchvision.utils  import make_grid
from torchvision.transforms.functional import to_pil_image

CAPTIONS = [
    "I like my vectors random, but my attention... focused.",
    "Not everything generated is fake — some things just feel real.",
    "This model doesn’t overfit — unless you give it a reason to.",
    "Every latent vector has a story — some more attractive than others.",
    "All it takes is one seed to create something unforgettable.",
    "There's beauty in high-dimensional spaces — especially when they collapse into one.",
    "You bring the noise, I’ll bring the structure.",
    "Every sample is unique — just like the one viewing this.",
    "The distribution may be Gaussian, but the attraction is anything but normal.",
    "Some interpolations are smooth, others... electric.",
    "Don't mistake low resolution for low effort — this one’s got layers.",
    "Some images are blurry, some are sharp — kind of like first impressions.",
    "My decoder doesn't hallucinate — but it sure dreams well.",
    "Latent or not, some variables just spark curiosity.",
    "Behind every noise vector is a shape waiting to be seen — much like a good conversation."
]

post_gen_comments = [
    "Not quite 4K, but hey — this image was born from pure noise.",
    "A little blurry? Maybe. But so is the line between chaos and creation.",
    "Every pixel here started as a whisper from randomness.",
    "Who knew numbers could dream up something so close to reality?",
    "Generated from scratch — no filters, just math and magic.",
    "It’s not blurry, it’s abstract realism. Trust the process.",
    "Think of it as a photo taken by probability itself.",
    "This might be the best-looking static you’ve ever seen.",
    "Just imagine — this image didn’t exist a second ago.",
    "Born from chaos, raised by GANs, blurry but brilliant.",
]


def get_random_caption():
    return random.choice(CAPTIONS)

@torch.no_grad()
def prepare_decoder():
    vae = VAE().to(Config.device)
    path = "saves\checkpoints\gan_checkpoint_epoch_50.pt"
    checkpoint = torch.load(f"{Config.model_save_path}/gan_checkpoint_epoch_50.pt", map_location=Config.device)
    vae.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    vae.decoder.eval()
    return vae.decoder

@torch.no_grad()
def generate_image_from_latent(decoder, z_tensor):
    decoded = decoder(z_tensor).cpu().squeeze(0)
    grid = make_grid(decoded, normalize=True)
    return to_pil_image(grid)