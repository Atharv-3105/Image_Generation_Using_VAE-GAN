# 🎨 Image Generation Using VAE + GAN

Welcome to the **VAE + GAN Image Generator**, a two-phase project that explores how random noise can be transformed into realistic images using a combination of deep learning techniques — **Variational Autoencoders (VAE)** and **Generative Adversarial Networks (GANs)**.

![vae_gan_demo](./saves/gan_samples/interpolation_epoch_30.gif)

---

## 🧠 Project Overview

This project is structured in **two main phases**:

### Phase 1: 🔧 Variational Autoencoder (VAE) Pretraining
- A standard **VAE model** is trained on the **CIFAR-10 dataset**.
- The model learns to **encode images into latent vectors** and then reconstruct them back.
- Objective: To learn a meaningful latent representation of images.
- ✍️ **Loss Function**: Combines **Reconstruction Loss** (MSE) with **KL Divergence** to ensure smooth latent space and image fidelity.
- Output: Well-trained `encoder` and `decoder`.
- 📊 **Training Logs** (wandb): [VAE Training Dashboard](https://wandb.ai/atharv3105-dr-a-p-j-abdul-kalam-technical-university/VAE-CIFAR-10/runs/rv8tmpo7?nw=nwuseratharv3105)

### Phase 2: 🎭 GAN Fine-Tuning
- The **pretrained VAE decoder** is used as the **GAN Generator**.
- A **Discriminator** is trained adversarially to improve the quality of the generated images.
- Generator (VAE Decoder) is fine-tuned to produce **sharper, more realistic images** than VAE alone.
- Discriminator learns to distinguish between real CIFAR-10 images and generated ones.
- ✍️ **Loss Functions**:
  - **Binary Cross Entropy (BCE)** for both Generator and Discriminator.
  - Generator tries to "fool" the Discriminator using BCE with target labels as **real**.
  - Discriminator is trained with separate BCE losses for real and fake inputs.
- 📊 **Training Logs** (wandb): [GAN Training Dashboard](https://wandb.ai/atharv3105-dr-a-p-j-abdul-kalam-technical-university/VAE-GAN-CIFAR-10-Colab/runs/w3wghhmj?nw=nwuseratharv3105)

---

## 🚀 Functionality

- 🔀 **Latent Space Exploration**: Control a 128-dimensional latent vector and visualize how it generates images.
- 🎮 **Random Sampling**: Automatically generate random images from noise.
- 📈 **Fine-tuned Decoder**: Unlike basic VAEs, the decoder here is fine-tuned with GAN training for higher-quality outputs.
- 📦 **Streamlit App**: Interactive UI built using Streamlit. Deployed to the cloud.
- 🧠 **Educational Insight**: See how structured noise becomes creativity!

---

## 🖼 Streamlit App Preview

🌐 Try the app: [Deployed Link](https://project-vae.streamlit.app/)

Features:
- Control latent dimensions with sliders (or reduced controls for simplicity).
- Generate CIFAR-10-like images on the fly.
- Enjoy random witty captions with each generation.

---


---

## 🧪 Technical Stack

- **Framework**: PyTorch, Streamlit
- **Dataset**: CIFAR-10
- **Image Size**: 32×32 (resized to 256×256 in the app)
- **Latent Vector**: 128 dimensions
- **Training**: VAE pretrained → GAN Fine-tuning

---

## 🔁 Future Work

- ✅ Train with higher resolution datasets
- ✅ Experiment with different GAN loss functions (e.g., WGAN, LSGAN)
- ⏳ Improve decoder’s fine-tuning for sharper outputs
- 💡 Add feature to interpolate between latent vectors live in the app

---

## 🧙🏻‍♂️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Atharv-3105/Image_Generation_Using_VAE-GAN.git
cd Image_Generation_VAE+GAN

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app/app.py


