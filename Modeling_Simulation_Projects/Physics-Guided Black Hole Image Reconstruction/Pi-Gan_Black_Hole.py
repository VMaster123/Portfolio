import torch
import torch.nn as nn
import torch.optim as optim


# ----- Generator -----
class Generator(nn.Module):
    def __init__(self, latent_dim=128, img_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_size * img_size),
            nn.Tanh(),  # outputs between -1 and 1
        )
        self.img_size = img_size

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, self.img_size, self.img_size)


# ----- Discriminator -----
class Discriminator(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


# ----- Physics-Informed Loss -----
def physics_loss(reconstructed_img, measured_visibilities, mask):
    """
    Enforce consistency with interferometric data (Fourier domain).
    - reconstructed_img: generated black hole image
    - measured_visibilities: observed sparse Fourier samples
    - mask: binary mask of observed Fourier coefficients
    """
    # Forward Fourier transform of image
    fourier_img = torch.fft.fft2(reconstructed_img.squeeze(1))
    fourier_img = torch.fft.fftshift(fourier_img)

    # Compute loss only where data is measured
    return torch.mean(torch.abs(mask * (fourier_img - measured_visibilities)) ** 2)


# ----- Training Loop (Sketch) -----
latent_dim = 128
img_size = 64
generator = Generator(latent_dim, img_size)
discriminator = Discriminator(img_size)

optimizer_G = optim.Adam(generator.parameters(), lr=2e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4)
criterion = nn.BCELoss()

for epoch in range(10000):
    # 1. Sample noise
    z = torch.randn(32, latent_dim)

    # 2. Generate fake images
    fake_imgs = generator(z)

    # 3. Discriminator training
    real_imgs = torch.randn(
        32, 1, img_size, img_size
    )  # replace with true black hole priors
    optimizer_D.zero_grad()
    real_loss = criterion(discriminator(real_imgs), torch.ones(32, 1))
    fake_loss = criterion(discriminator(fake_imgs.detach()), torch.zeros(32, 1))
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # 4. Generator training with physics loss
    optimizer_G.zero_grad()
    adv_loss = criterion(discriminator(fake_imgs), torch.ones(32, 1))

    # Physics loss (requires telescope visibilities + mask)
    measured_vis = torch.randn_like(fake_imgs.squeeze(1))  # placeholder
    mask = torch.randint(0, 2, measured_vis.shape).float()
    phys_loss = physics_loss(fake_imgs, measured_vis, mask)

    g_loss = adv_loss + 0.1 * phys_loss  # λ balances adversarial vs physics
    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"[{epoch}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
