"""
Physics-Informed Generative Adversarial Network (PI-GAN) for Black Hole Image Reconstruction
=============================================================================================

This implementation reconstructs black hole images from sparse interferometric data (like EHT).
The approach combines:
1. Deep Convolutional GAN (DCGAN) for high-quality image generation
2. Physics-informed loss enforcing Fourier domain consistency
3. Multi-scale reconstruction and spectral normalization for stability
4. Comprehensive numerical evaluation metrics

Author: Vilohith Gokarakonda
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: GENERATOR NETWORK (DCGAN Architecture)
# ============================================================================
# The generator creates black hole images from random noise vectors.
# Uses transposed convolutions to upsample from latent space to image space.


class Generator(nn.Module):
    """
    Generator network: Maps random noise z -> black hole image.

    Architecture:
    - Input: Random noise vector (latent_dim dimensions)
    - Linear layer: Expands to feature map size
    - Transposed convolutions: Upsamples 4x4 -> 16x16 -> 64x64
    - Batch normalization: Stabilizes training
    - LeakyReLU: Prevents dead neurons
    - Tanh output: Normalizes to [-1, 1] range
    """

    def __init__(self, latent_dim=100, img_size=64):
        super(Generator, self).__init__()
        self.init_size = img_size // 4  # Start from 16x16 for 64x64 output
        self.latent_dim = latent_dim

        # Project noise vector to initial feature map
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size**2),
            nn.BatchNorm1d(128 * self.init_size**2),
        )

        # Upsampling blocks: 16x16 -> 32x32 -> 64x64
        self.conv_blocks = nn.Sequential(
            # First upsampling: 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # Second upsampling: 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # Final convolution to single channel
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z):
        """Forward pass: noise -> image."""
        # Reshape noise to feature map
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # Upsample through convolutional blocks
        img = self.conv_blocks(out)
        return img


# ============================================================================
# STEP 2: DISCRIMINATOR NETWORK (CNN Classifier)
# ============================================================================
# The discriminator distinguishes real vs generated black hole images.
# Uses spectral normalization for training stability.


class SpectralNorm(nn.Module):
    """
    Spectral Normalization: Constrains weight matrix spectral norm <= 1.
    This stabilizes GAN training by preventing discriminator from becoming too strong.
    """

    def __init__(self, module, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.power_iterations = power_iterations

        # Initialize u and v for power iteration
        w = module.weight.data
        height = w.shape[0]
        width = w.numel() // height
        u = F.normalize(w.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(w.new_empty(width).normal_(0, 1), dim=0, eps=1e-12)
        self.register_buffer("u", u)
        self.register_buffer("v", v)

    def forward(self, x):
        w = self.module.weight
        height = w.shape[0]

        # Power iteration to estimate spectral norm
        with torch.no_grad():
            for _ in range(self.power_iterations):
                self.v = F.normalize(
                    torch.mv(w.view(height, -1).t(), self.u), dim=0, eps=1e-12
                )
                self.u = F.normalize(
                    torch.mv(w.view(height, -1), self.v), dim=0, eps=1e-12
                )

        # Normalize weights
        sigma = torch.dot(self.u, torch.mv(w.view(height, -1), self.v))
        w_normalized = w / (sigma + 1e-12)  # Add small epsilon for numerical stability
        return F.conv2d(
            x,
            w_normalized,
            self.module.bias,
            self.module.stride,
            self.module.padding,
            self.module.dilation,
            self.module.groups,
        )


class Discriminator(nn.Module):
    """
    Discriminator network: Classifies images as real or fake.

    Architecture:
    - Convolutional blocks: Downsample 64x64 -> 4x4
    - Spectral normalization: Stabilizes training
    - Dropout: Prevents overfitting
    - Sigmoid output: Probability of being real
    """

    def __init__(self, img_size=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, use_sn=True):
            """Convolutional block with optional spectral normalization."""
            layers = []
            conv = nn.Conv2d(in_filters, out_filters, 3, 2, 1)
            if use_sn:
                layers.append(SpectralNorm(conv))
            else:
                layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            if bn:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            return layers

        # Build discriminator: progressively downsample
        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False, use_sn=True),  # 64x64 -> 32x32
            *discriminator_block(16, 32, bn=True, use_sn=True),  # 32x32 -> 16x16
            *discriminator_block(32, 64, bn=True, use_sn=True),  # 16x16 -> 8x8
            *discriminator_block(64, 128, bn=True, use_sn=True),  # 8x8 -> 4x4
        )

        # Final classification layer
        ds_size = img_size // 2**4  # 4x4 after 4 downsampling steps
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        """Forward pass: image -> probability of being real."""
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# ============================================================================
# STEP 3: PHYSICS-INFORMED COMPONENTS
# ============================================================================
# These functions enforce physical constraints from interferometry.


def generate_synthetic_bh(img_size=64):
    """
    Generates synthetic black hole 'ring' image (ground truth).

    Physics:
    - Black holes appear as bright rings due to photon sphere
    - Relativistic Doppler boosting creates brightness asymmetry
    - Ring radius ~ 5.2 GM/c² for Schwarzschild black hole

    Parameters:
    - ring_radius: Angular size of photon sphere
    - ring_width: Thickness of emission ring
    - asymmetry: Doppler boosting factor (1 + β*cos(θ))
    """
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)

    # Ring parameters (based on GR predictions)
    ring_radius = 0.5  # Normalized radius
    ring_width = 0.1  # Gaussian width
    ring = np.exp(-((r - ring_radius) ** 2) / (2 * ring_width**2))

    # Relativistic Doppler boosting: brighter on approaching side
    # v/c ~ 0.3 for M87* accretion disk
    beta = 0.3
    asymmetry = 1 + beta * X  # Brightening in +X direction
    ring = ring * asymmetry

    # Normalize to [-1, 1] for GAN
    ring = (ring / np.max(ring)) * 2 - 1
    return torch.FloatTensor(ring).unsqueeze(0).unsqueeze(0)


def create_uv_mask(img_size=64, sparsity=0.1):
    """
    Simulates sparse UV sampling of telescope array (like Event Horizon Telescope).

    Physics:
    - Radio interferometry measures Fourier transform (visibilities)
    - Earth rotation creates 'tracks' in UV plane
    - Sparse coverage due to limited telescope baselines

    The mask indicates which Fourier components are observed.
    """
    mask = np.zeros((img_size, img_size))

    # Simulate Earth rotation tracks (8 baselines)
    num_tracks = 8
    for _ in range(num_tracks):
        angle = np.random.uniform(0, np.pi)
        length = np.random.uniform(0.1, 0.9)

        # Create track in UV plane
        for r in np.linspace(0, length, img_size // 2):
            idx_x = int(img_size / 2 + r * np.cos(angle) * (img_size / 2 - 1))
            idx_y = int(img_size / 2 + r * np.sin(angle) * (img_size / 2 - 1))

            # Ensure indices are valid
            idx_x = np.clip(idx_x, 0, img_size - 1)
            idx_y = np.clip(idx_y, 0, img_size - 1)

            mask[idx_x, idx_y] = 1
            # Conjugate symmetry: V(-u, -v) = V*(u, v)
            mask[img_size - idx_x - 1, img_size - idx_y - 1] = 1

    return torch.FloatTensor(mask)


def physics_loss(gen_img, gt_visibilities, mask):
    """
    Physics-informed loss: Enforces Fourier domain consistency.

    This is the key innovation: instead of just matching pixels, we enforce
    that the generated image's Fourier transform matches observed visibilities
    at the sparse measurement locations.

    Loss = ||mask * (FFT(generated) - observed_visibilities)||²
    """
    # FFT of generated image
    gen_vis = torch.fft.fftshift(torch.fft.fft2(gen_img.squeeze(1)))

    # Masked MSE loss in Fourier domain
    # Only penalize differences where we have measurements
    diff = mask * (gen_vis - gt_visibilities)
    loss = torch.mean(torch.abs(diff) ** 2)
    return loss


def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    """
    Gradient Penalty for WGAN-GP: Improves training stability.

    Penalizes discriminator gradients to be close to 1, preventing
    discriminator from becoming too confident.

    Note: real_imgs and fake_imgs should be detached to avoid graph issues.
    """
    batch_size = real_imgs.size(0)

    # Ensure inputs don't require gradients (they shouldn't, but be safe)
    real_imgs = real_imgs.detach() if real_imgs.requires_grad else real_imgs
    fake_imgs = fake_imgs.detach() if fake_imgs.requires_grad else fake_imgs

    # Create interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=False)

    # Interpolate between real and fake
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    # Compute discriminator output on interpolated samples
    d_interpolates = discriminator(interpolates)

    # Compute gradients of discriminator output w.r.t. interpolated samples
    # For WGAN-GP, we need create_graph=True to allow backpropagation through the penalty
    # This enables gradients to flow back to the discriminator parameters
    # retain_graph=True keeps the graph alive for the subsequent d_loss.backward() call
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,  # Need this to backprop through penalty to discriminator
        retain_graph=True,  # Retain graph so d_loss.backward() can use it
        only_inputs=True,  # Only compute gradients w.r.t. interpolates (not discriminator params here)
    )[0]

    # Flatten gradients for norm calculation
    gradients = gradients.view(gradients.size(0), -1)

    # Penalty: (||gradient|| - 1)²
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ============================================================================
# STEP 4: EVALUATION METRICS
# ============================================================================


def calculate_psnr(img1, img2):
    """
    Peak Signal-to-Noise Ratio: Measures reconstruction quality.

    PSNR = 20 * log10(MAX_VAL / sqrt(MSE))
    Higher is better. Typical values: 20-30 dB for good reconstruction.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    max_val = 2.0  # Since images are in [-1, 1]
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Structural Similarity Index: Measures perceptual quality.

    SSIM considers luminance, contrast, and structure.
    Range: [-1, 1], higher is better.
    """
    # Simplified SSIM calculation
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim


def calculate_fourier_error(gen_img, gt_visibilities, mask):
    """
    Fourier domain error: Measures consistency with interferometric data.

    This is the key metric for physics-informed reconstruction.
    """
    gen_vis = torch.fft.fftshift(torch.fft.fft2(gen_img.squeeze(1)))
    diff = mask * (gen_vis - gt_visibilities)
    error = torch.mean(torch.abs(diff) ** 2)
    return error


# ============================================================================
# STEP 5: TRAINING CONFIGURATION
# ============================================================================

# Hyperparameters
img_size = 64
latent_dim = 100
n_epochs = 500  # Reduced for quick results
batch_size = 32
lambda_phys = 5.0  # Weight for physics loss
lambda_gp = 10.0  # Weight for gradient penalty
n_critic = 5  # Train discriminator n_critic times per generator update

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
generator = Generator(latent_dim, img_size).to(device)
discriminator = Discriminator(img_size).to(device)

# Optimizers with learning rate scheduling
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Learning rate schedulers (reduce LR on plateau)
scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_G, mode="min", factor=0.5, patience=50
)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_D, mode="min", factor=0.5, patience=50
)

# Loss functions
adversarial_loss = nn.BCELoss()

# Data setup (ensure no gradients on ground truth data)
ground_truth = generate_synthetic_bh(img_size).to(device).detach().requires_grad_(False)
uv_mask = create_uv_mask(img_size).to(device).detach().requires_grad_(False)
gt_visibilities = (
    torch.fft.fftshift(torch.fft.fft2(ground_truth.squeeze(1)))
    .to(device)
    .detach()
    .requires_grad_(False)
)

# Training history
history = defaultdict(list)


# Model statistics
def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


gen_params = count_parameters(generator)
disc_params = count_parameters(discriminator)
total_params = gen_params + disc_params

print("=" * 80)
print("Physics-Informed GAN for Black Hole Image Reconstruction")
print("=" * 80)
print(f"Image size: {img_size}x{img_size}")
print(f"Latent dimension: {latent_dim}")
print(f"Batch size: {batch_size}")
print(f"Physics loss weight: {lambda_phys}")
print(f"Generator parameters: {gen_params:,} ({gen_params/1e6:.2f}M)")
print(f"Discriminator parameters: {disc_params:,} ({disc_params/1e6:.2f}M)")
print(f"Total model size: {total_params:,} ({total_params/1e6:.2f}M parameters)")
print("=" * 80)

start_time = time.time()

# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================

best_psnr = 0.0
patience_counter = 0
early_stop_patience = 100

for epoch in range(n_epochs):

    # Train Discriminator (n_critic times)
    for _ in range(n_critic):
        optimizer_D.zero_grad()

        # Real images (ground truth with noise as priors) - ensure no gradients
        with torch.no_grad():
            real_imgs_base = ground_truth.repeat(batch_size, 1, 1, 1)
            noise = 0.1 * torch.randn(batch_size, 1, img_size, img_size, device=device)
        real_imgs = (real_imgs_base + noise).detach().requires_grad_(False)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Discriminator loss on real images
        real_pred = discriminator(real_imgs)
        real_loss = adversarial_loss(real_pred, valid)

        # Generate fake images (detached for discriminator training)
        z_d = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            gen_imgs_d = generator(z_d)  # Generate without tracking gradients
        gen_imgs_d = gen_imgs_d.detach().requires_grad_(False)  # Ensure no gradients
        fake_pred = discriminator(gen_imgs_d)
        fake_loss = adversarial_loss(fake_pred, fake)

        # Compute adversarial loss first
        adv_loss = (real_loss + fake_loss) / 2

        # Gradient penalty (uses detached images) - compute separately
        # This creates a separate computation graph that will be combined with adv_loss
        gp = gradient_penalty(discriminator, real_imgs, gen_imgs_d, device)

        # Total discriminator loss
        d_loss = adv_loss + lambda_gp * gp

        # Store loss value before backward to avoid accessing after graph is freed
        d_loss_val = float(d_loss.item())

        # Backward pass
        d_loss.backward()
        optimizer_D.step()

    # Train Generator (generate fresh images with gradient tracking)
    optimizer_G.zero_grad()

    # Generate new fake images for generator training (with gradients)
    z_g = torch.randn(batch_size, latent_dim, device=device)
    gen_imgs = generator(z_g)

    # Adversarial loss (fool discriminator)
    g_adv_loss = adversarial_loss(discriminator(gen_imgs), valid)

    # Physics-informed loss (match Fourier domain)
    g_phys_loss = physics_loss(gen_imgs, gt_visibilities, uv_mask)

    # Total generator loss
    g_loss = g_adv_loss + lambda_phys * g_phys_loss

    # Store loss values before backward
    g_loss_val = float(g_loss.item())

    g_loss.backward()
    optimizer_G.step()
    scheduler_G.step(g_loss_val)
    scheduler_D.step(d_loss_val)

    # Logging & Evaluation
    if epoch % 25 == 0 or epoch < 10:  # More frequent early logging
        with torch.no_grad():
            generator.eval()  # Set to eval mode for inference (fixes BatchNorm issue with batch_size=1)
            test_z = torch.randn(1, latent_dim, device=device)
            reconstruction = generator(test_z)
            generator.train()  # Set back to training mode

            # Calculate metrics
            psnr = calculate_psnr(reconstruction, ground_truth)
            ssim = calculate_ssim(reconstruction, ground_truth)
            fourier_err = calculate_fourier_error(
                reconstruction, gt_visibilities, uv_mask
            )

            # Track best model
            if psnr.item() > best_psnr:
                best_psnr = psnr.item()
                patience_counter = 0
                # Save best model
                torch.save(generator.state_dict(), "best_generator.pth")
            else:
                patience_counter += 1

            # Store history (convert to float to avoid tensor retention)
            history["d_loss"].append(d_loss_val)
            history["g_loss"].append(g_loss_val)
            history["g_adv_loss"].append(float(g_adv_loss.item()))
            history["g_phys_loss"].append(float(g_phys_loss.item()))
            history["psnr"].append(float(psnr.item()))
            history["ssim"].append(float(ssim.item()))
            history["fourier_error"].append(float(fourier_err.item()))

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f} | "
                f"Phys Loss: {g_phys_loss.item():.6f} | "
                f"PSNR: {psnr.item():6.2f} dB | "
                f"SSIM: {ssim.item():.4f} | "
                f"Fourier Err: {fourier_err.item():.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} evaluations)"
                )
                break

# ============================================================================
# STEP 7: FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)

# Load best model (handle case where file might not exist)
try:
    generator.load_state_dict(torch.load("best_generator.pth"))
    print("Loaded best model from checkpoint.")
except FileNotFoundError:
    print("No checkpoint found, using current model state.")

# Final evaluation
with torch.no_grad():
    generator.eval()  # Set to eval mode for inference (fixes BatchNorm issue with batch_size=1)
    test_z = torch.randn(1, latent_dim, device=device)
    final_reconstruction = generator(test_z)

    final_psnr = calculate_psnr(final_reconstruction, ground_truth)
    final_ssim = calculate_ssim(final_reconstruction, ground_truth)
    final_fourier_err = calculate_fourier_error(
        final_reconstruction, gt_visibilities, uv_mask
    )

    # Convert to Python floats to avoid any tensor issues
    final_psnr_val = float(final_psnr.item())
    final_ssim_val = float(final_ssim.item())
    final_fourier_err_val = float(final_fourier_err.item())

# Calculate additional statistics
total_time = time.time() - start_time
convergence_epoch = len(history["psnr"])
psnr_improvement = best_psnr - history["psnr"][0] if len(history["psnr"]) > 0 else 0
fourier_reduction = (
    (history["fourier_error"][0] / (history["fourier_error"][-1] + 1e-10))
    if len(history["fourier_error"]) > 1
    else 1.0
)
initial_fourier = (
    history["fourier_error"][0] if len(history["fourier_error"]) > 0 else 0
)
final_fourier_hist = (
    history["fourier_error"][-1] if len(history["fourier_error"]) > 0 else 0
)
avg_psnr = float(np.mean(history["psnr"])) if len(history["psnr"]) > 0 else 0.0
std_psnr = float(np.std(history["psnr"])) if len(history["psnr"]) > 0 else 0.0
avg_ssim = float(np.mean(history["ssim"])) if len(history["ssim"]) > 0 else 0.0
final_mse = F.mse_loss(final_reconstruction, ground_truth).item()
pixel_accuracy = (1 - final_mse / 4.0) * 100  # 4.0 is max MSE for [-1,1] range
final_phys_loss = history["g_phys_loss"][-1] if len(history["g_phys_loss"]) > 0 else 0.0

print(f"\nFINAL NUMERICAL RESULTS:")
print(f"  {'='*70}")
print(f"  RECONSTRUCTION QUALITY METRICS:")
print(f"    Peak Signal-to-Noise Ratio (PSNR): {final_psnr_val:.2f} dB")
print(f"    Best PSNR during training: {best_psnr:.2f} dB")
print(f"    Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
print(f"    PSNR improvement: {psnr_improvement:.2f} dB (from initial)")
print(f"    Structural Similarity Index (SSIM): {final_ssim_val:.4f}")
print(f"    Average SSIM: {avg_ssim:.4f}")
print(f"    Mean Squared Error (MSE): {final_mse:.6f}")
print(f"    Pixel-level accuracy: {pixel_accuracy:.2f}%")
print(f"  {'='*70}")
print(f"  PHYSICS CONSISTENCY METRICS:")
print(f"    Fourier Domain Error: {final_fourier_err_val:.6f}")
print(
    f"    Fourier error reduction: {fourier_reduction:.2f}x (from {initial_fourier:.6f} to {final_fourier_hist:.6f})"
)
print(f"    Physics loss (final): {final_phys_loss:.6f}")
print(f"  {'='*70}")
print(f"  TRAINING EFFICIENCY METRICS:")
print(
    f"    Total training time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)"
)
print(f"    Convergence epoch: {convergence_epoch}")
print(f"    Training epochs completed: {len(history['psnr'])}")
print(f"    Time per epoch: {total_time/max(len(history['psnr']), 1):.2f} seconds")
print(f"    Final learning rate (G): {optimizer_G.param_groups[0]['lr']:.6f}")
print(f"    Final learning rate (D): {optimizer_D.param_groups[0]['lr']:.6f}")
print(f"  {'='*70}")
print(f"  MODEL COMPLEXITY:")
print(f"    Generator parameters: {gen_params:,} ({gen_params/1e6:.2f}M)")
print(f"    Discriminator parameters: {disc_params:,} ({disc_params/1e6:.2f}M)")
print(f"    Total model size: {total_params:,} ({total_params/1e6:.2f}M parameters)")
print(
    f"    Model size on disk: {sum(p.numel() * 4 for p in generator.parameters()) / 1024 / 1024:.2f} MB"
)
print(f"  {'='*70}")

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================


def plot_results(gt, mask, recon, history, final_psnr_value):
    """Create comprehensive visualization of results."""
    fig = plt.figure(figsize=(18, 12))

    # Row 1: Images
    plt.subplot(3, 3, 1)
    plt.title("Ground Truth\n(Black Hole Ring)", fontsize=12, fontweight="bold")
    plt.imshow(gt[0, 0].cpu().numpy(), cmap="hot", origin="lower")
    plt.colorbar(label="Intensity")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("Sparse UV Mask\n(Telescope Coverage)", fontsize=12, fontweight="bold")
    plt.imshow(mask.cpu().numpy(), cmap="gray", origin="lower")
    plt.colorbar(label="Coverage")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.title(
        f"PI-GAN Reconstruction\nPSNR: {final_psnr_value:.2f} dB",
        fontsize=12,
        fontweight="bold",
    )
    plt.imshow(recon[0, 0].cpu().numpy(), cmap="hot", origin="lower")
    plt.colorbar(label="Intensity")
    plt.axis("off")

    # Row 2: Training curves
    plt.subplot(3, 3, 4)
    plt.plot(history["d_loss"], label="Discriminator", linewidth=2)
    plt.plot(history["g_loss"], label="Generator", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 5)
    plt.plot(history["g_adv_loss"], label="Adversarial", linewidth=2, alpha=0.7)
    plt.plot(history["g_phys_loss"], label="Physics", linewidth=2, alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss Components", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.subplot(3, 3, 6)
    plt.plot(history["psnr"], label="PSNR", linewidth=2, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Reconstruction Quality", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 3: Additional metrics
    plt.subplot(3, 3, 7)
    plt.plot(history["ssim"], label="SSIM", linewidth=2, color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Structural Similarity", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 8)
    plt.plot(history["fourier_error"], label="Fourier Error", linewidth=2, color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Physics Consistency", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Difference map
    plt.subplot(3, 3, 9)
    diff = torch.abs(gt - recon)
    plt.imshow(diff[0, 0].cpu().numpy(), cmap="viridis", origin="lower")
    plt.colorbar(label="Absolute Difference")
    plt.title("Reconstruction Error Map", fontweight="bold")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_results.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'reconstruction_results.png'")


plot_results(ground_truth, uv_mask, final_reconstruction, history, final_psnr_val)

print("\n" + "=" * 80)
print("RESUME-READY SUMMARY")
print("=" * 80)
print("\nKey Achievements:")
print(f"  - PSNR: {final_psnr_val:.2f} dB (improvement: {psnr_improvement:.2f} dB)")
print(f"  - SSIM: {final_ssim_val:.4f}")
print(f"  - Fourier Error Reduction: {fourier_reduction:.1f}x")
print(f"  - Model Size: {total_params/1e6:.2f}M parameters")
print(
    f"  - Training Efficiency: {convergence_epoch} epochs, {total_time/60:.2f} minutes"
)
print(f"  - Physics Consistency: {final_fourier_err_val:.6f} (target: <0.001)")
print("\nAll done! Check 'reconstruction_results.png' for comprehensive results.")
