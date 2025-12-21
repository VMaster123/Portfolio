import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ----- Generator (CNN based) -----
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=64):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ----- Discriminator (CNN based) -----
class Discriminator(nn.Module):
    def __init__(self, img_size=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# ----- Physics-Informed Components -----


def generate_synthetic_bh(img_size=64):
    """Generates a synthetic black hole 'ring' image (Ground Truth)."""
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)

    # Ring parameters
    ring_radius = 0.5
    ring_width = 0.1
    ring = np.exp(-((r - ring_radius) ** 2) / (2 * ring_width**2))

    # Add some asymmetry (brightening on one side due to Doppler boosting)
    asymmetry = 1 + 0.5 * X
    ring = ring * asymmetry

    # Normalize to [-1, 1] for GAN
    ring = (ring / np.max(ring)) * 2 - 1
    return torch.FloatTensor(ring).unsqueeze(0).unsqueeze(0)


def create_uv_mask(img_size=64, sparsity=0.1):
    """Simulates sparse UV sampling of a telescope array like EHT."""
    mask = np.zeros((img_size, img_size))
    # Simulate a few 'tracks' in the UV plane
    num_tracks = 8
    for _ in range(num_tracks):
        angle = np.random.uniform(0, np.pi)
        length = np.random.uniform(0.1, 0.9)
        for r in np.linspace(0, length, img_size // 2):
            idx_x = int(img_size / 2 + r * np.cos(angle) * (img_size / 2 - 1))
            idx_y = int(img_size / 2 + r * np.sin(angle) * (img_size / 2 - 1))
            mask[idx_x, idx_y] = 1
            mask[img_size - idx_x - 1, img_size - idx_y - 1] = 1  # Conjugate symmetry

    return torch.FloatTensor(mask)


def physics_loss(gen_img, gt_visibilities, mask):
    """Computes Fourier domain consistency loss."""
    # FFT of generated image
    gen_vis = torch.fft.fftshift(torch.fft.fft2(gen_img.squeeze(1)))

    # Masked MSE loss in Fourier domain
    # gen_vis and gt_visibilities are complex; MSE on absolute differences
    diff = mask * (gen_vis - gt_visibilities)
    loss = torch.mean(torch.abs(diff) ** 2)
    return loss


def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


# ----- Training Configuration -----
img_size = 64
latent_dim = 100
n_epochs = 2000
batch_size = 32
lambda_phys = 5.0  # Weight for physics loss

# Initialize models
generator = Generator(latent_dim, img_size)
discriminator = Discriminator(img_size)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
adversarial_loss = nn.BCELoss()

# Data setup
ground_truth = generate_synthetic_bh(img_size)
uv_mask = create_uv_mask(img_size)
# Ground truth visibilities (Fourier transform of ground truth)
gt_visibilities = torch.fft.fftshift(torch.fft.fft2(ground_truth.squeeze(1)))

print("Starting Physics-Informed GAN training...")
start_time = time.time()

# ----- Training Loop -----
for epoch in range(n_epochs):

    # 1. Train Discriminator
    optimizer_D.zero_grad()

    # Sample real images (priors) - in a real case, these would be high-res BH simulations
    # Here we use the ground truth with some noise as 'real' data for demo
    real_imgs = ground_truth.repeat(batch_size, 1, 1, 1) + 0.1 * torch.randn(
        batch_size, 1, img_size, img_size
    )
    valid = torch.ones(batch_size, 1)
    fake = torch.zeros(batch_size, 1)

    # Discriminator loss
    real_loss = adversarial_loss(discriminator(real_imgs), valid)

    z = torch.randn(batch_size, latent_dim)
    gen_imgs = generator(z)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # 2. Train Generator
    optimizer_G.zero_grad()

    # Adversarial loss
    g_adv_loss = adversarial_loss(discriminator(gen_imgs), valid)

    # Physics-Informed loss (Consistency with sparse Fourier data)
    g_phys_loss = physics_loss(gen_imgs, gt_visibilities, uv_mask)

    # Total Generator Loss
    g_loss = g_adv_loss + lambda_phys * g_phys_loss

    g_loss.backward()
    optimizer_G.step()

    # 3. Logging & Numerical Results
    if epoch % 100 == 0:
        with torch.no_grad():
            test_z = torch.randn(1, latent_dim)
            reconstruction = generator(test_z)
            psnr = calculate_psnr(reconstruction, ground_truth)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{n_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} "
                f"| Physics Loss: {g_phys_loss.item():.6f} | PSNR: {psnr.item():.2f}dB | Time: {elapsed:.1f}s"
            )

# Final Results
print("\nTraining Complete.")
final_psnr = calculate_psnr(generator(torch.randn(1, latent_dim)), ground_truth)
print(f"Final Reconstruction PSNR: {final_psnr.item():.2f} dB")


# ----- Visualization of Results -----
def plot_results(gt, mask, recon):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth (Black Hole Ring)")
    plt.imshow(gt[0, 0], cmap="hot")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Sparse UV Mask (Telescope Coverage)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"PI-GAN Reconstruction\nPSNR: {final_psnr.item():.2f} dB")
    plt.imshow(recon[0, 0].detach().numpy(), cmap="hot")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_results.png")
    print("Results saved to 'reconstruction_results.png'")
    # plt.show() # Commented out for non-interactive environments


plot_results(ground_truth, uv_mask, generator(torch.randn(1, latent_dim)))
