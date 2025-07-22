import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from vae.model import VAE
from vae.config import Config

#Visualization Libraries
import numpy as np
from PIL import Image 
import streamlit as st 
import random
from utils import prepare_decoder, generate_image_from_latent, get_random_caption, post_gen_comments

#Set the Page Config
st.set_page_config(page_title="✨VAE+GAN Image Generator", layout = "centered")

#CSS-Style
st.markdown("""
    <style>
    @keyframes fadeIn{
        from {opacity: 0;}
        to {opacity:1;}
    }       
    .fade-in-text{
        animation: fadeIn 2s ease-in-out;
        font-size: 17px;
        color: #cccccc;
        margin-top: 1rem;
    }        
    </style>        
            
""", unsafe_allow_html=True)

#Set the Page title
st.title("🧠 VAE + GAN Image Generator (CIFAR-10)")
st.markdown("""
Welcome to the **VAE + GAN Image Generator**!

This project uses a **Variational AutoEncoder (VAE)** which is first trained on CIFAR-10 as part of Pre-Training,
Then the VAE_Decoder is fine-tuned using a **GAN Discriminator** to generate sharper, more realistic images from the CIFAR-10 Dataset.

- 🎛️ You can control a **128-dimensional latent vector** using sliders below.
- 📸 The Decoder will convert your latent vector into a CIFAR-10-like image.
- 🧪 Try different combinations to explore how the latent space affects the output.
- 💡 For peeps who are not into **AI/ML**; This app generates Images from scratch using only **Random Numbers**.


---
            
            
            
""")

#Load trained VAE decoder
decoder = prepare_decoder()


#Latent-Vector Controls

st.markdown("### 🎛️ Choose How to Generate the Latent Vector")

option = st.radio(
    "Select input method:",
    ["Manual (16 sliders)", "Surprise Me! (Random latent vector)"],
    horizontal=True
)

if option == "Manual (16 sliders)":
    latent_vector = torch.zeros(Config.latent_dim)
    for i in range(16):
        latent_vector[i] = st.slider(f"z[{i}]", -3.0, 3.0, 0.0, step=0.1)
    latent_vector[16:] = torch.randn(Config.latent_dim - 16)
else:
    latent_vector = torch.randn(Config.latent_dim)
    st.success("Random Latent Vector generated!! Hit 'Generate Image' to see the magic!!")
#----------Generate button----------
if st.button("🔮 Generate Image"):
        generated_img = generate_image_from_latent(decoder, latent_vector)
        selected_caption = get_random_caption()
        line = random.choice(post_gen_comments)
        
        resized_img = generated_img.resize((256,256), resample = Image.BICUBIC)
        
        st.image(resized_img, caption=line, use_container_width=False)
        
        # st.markdown(
        #     f"<div class = 'fade-in-text'>{line}</div>",
        #     unsafe_allow_html=True
        # )
        
#-------------Footer Section-------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 50px; color: white;">
        <p>Made with ❤️ by <strong>Atharva</strong></p>
        <p>💬<em>Who knew randomness could be this creative? If that excites you, let’s connect.</em></p>
        <p>
            <a href="https://github.com/Atharv-3105" target="_blank">
                    <img src="https://img.icons8.com/ios-glyphs/30/github.png" alt="GitHub"/>
            </a>
            &nbsp;&nbsp;
            <a href="https://www.linkedin.com/in/atharv3105/" target="_blank">
                          <img src="https://img.icons8.com/ios-filled/30/linkedin.png" alt="LinkedIn"/>
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
