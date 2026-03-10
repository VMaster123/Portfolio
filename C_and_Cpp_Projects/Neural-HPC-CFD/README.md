# Neural-HPC-CFD: Physics-Informed GPU-Accelerated Fluid Dynamics

This project is a high-performance framework for CFD simulations that integrate traditional numerical methods with **Hybrid Machine Learning (PINNs + FNOs)**, optimized for HPC environments and **Inverse Problem Theory**.

## 🚀 Key Features

*   **Hybrid Inverse Problem Solver**: Combines FNO (Predictor) and PINN (Validator/Refiner) to reconstruct full fluid fields from sparse sensor data.
*   **Fourier Neural Operators (FNO)**: Resolution-invariant spectral mapping using **cuFFT** for near-instant flow prediction.
*   **Physics-Informed ML (PINNs)**: A custom CUDA MLP implementation that computes Navier-Stokes residuals as a loss constraint.
*   **CUDA-Accelerated Solver**: High-performance grid operations for real-time fluid simulation.
*   **MPI Domain Decomposition**: Designed to scale across multiple GPUs and compute nodes in an HPC cluster.

## 🏗️ The Hybrid Approach
Instead of using one ML model, this project uses two in a complementary pipeline:
1.  **FNO (The Predictor)**: Acts as a "fast guesser." It looks at sparse data points and hallucinated a global field based on frequency patterns.
2.  **PINN (The Validator)**: Acts as the "physics police." It calculates if the FNO's guess follows physics (mass conservation, momentum). If not, it uses gradient descent to nudge the result into physical compliance.

## 🧪 Getting Started

```bash
# Compile with MPI and CUDA (Requires: cuFFT, cuBLAS)
make

# Run on 4 GPUs/Nodes
mpirun -np 4 ./neural_cfd
```

## 🧠 Inverse Problem Theory in Fluid Dynamics
In many real-world scenarios, we only have a few sensors (e.g., in a wind tunnel). The "Inverse Problem" is to work backward from those sparse measurements to reconstruct the entire 3D flow. 

This project solves this by optimizing the loss function:
$$ \mathcal{L}_{total} = \lambda_{data} \mathcal{L}_{sensor} + \lambda_{physics} \mathcal{L}_{pde} $$

## 📁 Project Structure

*   `src/inverse_solver.cu`: Core logic for reconciling data with physics.
*   `src/operator_engine.cu`: Fourier Neural Operator implementation using cuFFT.
*   `src/pinn.cu`: Physics-Informed residual loss calculation.
*   `src/solver.cu`: Traditional CFD stencils.
