#!/usr/bin/env python
# coding: utf-8

# 
# # Image reconstruction with a diffusion model
# 
# This code shows you how to use the DDRM diffusion algorithm to reconstruct images and also compute the
# uncertainty of a reconstruction from incomplete and noisy measurements.
# 
# The paper can be found at https://arxiv.org/pdf/2209.11888.pdf.
# 
# The DDRM method requires that:
# 
# * The operator has a singular value decomposition (i.e., the operator is a :class:`deepinv.physics.DecomposablePhysics`).
# * The noise is Gaussian with known standard deviation (i.e., the noise model is :class:`deepinv.physics.GaussianNoise`).
# 

# In[17]:


import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
import numpy as np
from deepinv.utils.demo import load_url_image
import torchvision.io as io


# ## Load example image from the internet
# 
# This example uses an image of Lionel Messi from Wikipedia.
# 
# 

# In[18]:


device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
"""
url = (
    "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
    "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
)
x = load_url_image(url=url, img_size=32).to(device)
print(f" type of x is : {x.type}")
print(x.shape)
"""


# In[38]:


# Local file path
file_path = r"D:\unfolded-proximal-nn\implementation\ProxImage\Python_tutorial\images\IMG_4572.JPG"

# Image size
img_size = 32

# Load local image
x = io.read_image(file_path).to(device)
x = torch.nn.functional.interpolate(x.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False)
x = x.squeeze(0)

# Print type and shape
print(f"Type of x is: {x.type()}")
print(f"Shape of x: {x.shape}")


# torch.Size([1, 3, 32, 32]) represents a 4-dimensional tensor where the first dimension is the batch size (1 in this case), the second dimension is the number of channels (3 in this case), and the last two dimensions are the height and width of each image (both 32 in this case).

# ## Define forward operator and noise model
# 
# We use image inpainting as the forward operator and Gaussian noise as the noise model.
# 
# 

# In[39]:


sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(
    mask=0.5,# half values are masked 
    tensor_size=x.shape,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
)


# ## Define the MMSE denoiser
# 
# The diffusion method requires an MMSE denoiser that can be evaluated a various noise levels.
# Here we use a pretrained DRUNET denoiser from the `denoisers <denoisers>` module.
# 
# 

# In[40]:


denoiser = dinv.models.DRUNet(pretrained="download").to(device)


# The MMSE denoiser assumes a statistical model for both the clean signal and the noise. Typically, the noise is assumed to be additive white Gaussian noise (AWGN), which is a common assumption in many signal processing applications.

# ## Create the Monte Carlo sampler
# 
# We can now reconstruct a noisy measurement using the diffusion method.
# We use the DDRM method from :class:`deepinv.sampling.DDRM`, which works with inverse problems that
# have a closed form singular value decomposition of the forward operator.
# The diffusion method requires a schedule of noise levels ``sigmas`` that are used to evaluate the denoiser.
# 
# 

# In[41]:


sigmas = np.linspace(1, 0, 100) if torch.cuda.is_available() else np.linspace(1, 0, 10)

diff = dinv.sampling.DDRM(denoiser=denoiser, etab=1.0, sigmas=sigmas, verbose=True)


# ## Generate the measurement
# We apply the forward model to generate the noisy measurement.
# 
# 

# In[42]:


y = physics(x)


# ## Run the diffusion algorithm and plot results
# The diffusion algorithm returns a sample from the posterior distribution. (the probability distribution after taking into account both prior knowledge and observed data.)
# We compare the posterior mean with a simple linear reconstruction.
# 
# 

# In[43]:


xhat = diff(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_lin):.2f} dB")
print(f"Diffusion PSNR: {dinv.utils.metric.cal_psnr(x, xhat):.2f} dB")

# plot results
error = (xhat - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
imgs = [x_lin, x, xhat]
plot(imgs, titles=["measurement", "ground truth", "DDRM reconstruction"])


# ## Create a Monte Carlo sampler
# Running the diffusion gives a single sample of the posterior distribution.
# In order to compute the posterior mean and variance, we can use multiple samples.
# This can be done using the :class:`deepinv.sampling.DiffusionSampler` class, which converts
# the diffusion algorithm into a fully fledged Monte Carlo sampler.
# We set the maximum number of iterations to 10, which means that the sampler will run the diffusion 10 times.
# 
# 

# In[44]:


f = dinv.sampling.DiffusionSampler(diff, max_iter=10)


# ## Run sampling algorithm and plot results
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.
# 
# 

# In[45]:


mean, var = f(y, physics)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_lin):.2f} dB")
print(f"Posterior mean PSNR: {dinv.utils.metric.cal_psnr(x, mean):.2f} dB")

# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [x_lin, x, mean, std / std.flatten().max(), error / error.flatten().max()]
plot(
    imgs,
    titles=["measurement", "ground truth", "post. mean", "post. std", "abs. error"],
)


# In[ ]:




