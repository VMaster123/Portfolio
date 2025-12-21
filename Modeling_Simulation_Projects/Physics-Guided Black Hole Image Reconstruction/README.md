# Physics-Guided Black Hole Image Reconstruction (PI-GAN)

This project implements a Physics-Informed Generative Adversarial Network (PI-GAN) to reconstruct black hole images from sparse interferometric data, similar to the method used by the Event Horizon Telescope (EHT).

## Key Features
- **Deep Convolutional GAN (DCGAN)**: High-resolution image generation using transposed convolutions.
- **Physics-Informed Loss**: Enforces consistency in the Fourier domain using a simulated sparse UV sampling mask.
- **Synthetic Data Generation**: Realistic black hole ring models with relativistic doppler boosting.
- **Quantitative Metrics**: Performance evaluation using Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE).

## Numerical Results
The model tracks reconstruction quality during training. Typical results show:
- **Physics Loss Reduction**: Decreases by several orders of magnitude as the generator learns the Fourier constraints.
- **PSNR Improvement**: Reconstructions typically reach **20-25+ dB** compared to ground truth, starting from ~10 dB.

## How to Run
1. Install dependencies: `pip install torch numpy matplotlib`
2. Run the training script: `python Pi-Gan_Black_Hole.py`
3. View the results in `reconstruction_results.png`.
