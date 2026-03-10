#ifndef PINN_H
#define PINN_H

#include <cuda_runtime.h>

typedef struct {
    int num_layers;
    int *layer_sizes;
    float **weights;
    float **biases;
    float **d_weights; // Device pointers
    float **d_biases;
} PINNConfig;

// Initialize the neural network with specified layer sizes
void pinn_init(PINNConfig *pinn, int *layers, int num_layers);

// Perform a forward pass and return the prediction (e.g., [u, v, p] for given [x, y, t])
__device__ void pinn_forward(PINNConfig *pinn, float *input, float *output);

// Compute the Physics-Informed Residual: L = || Navier-Stokes(NN(x, y, t)) ||^2
// This function calculates derivatives of the NN w.r.t inputs using Autodiff or Finite Differences on GPU
__global__ void pinn_compute_residual_loss(PINNConfig *pinn, float *points, float *residuals, int num_points);

// Clean up memory
void pinn_free(PINNConfig *pinn);

#endif
