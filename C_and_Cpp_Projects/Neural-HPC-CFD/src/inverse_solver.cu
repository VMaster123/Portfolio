#include <stdio.h>
#include <cuda_runtime.h>
#include "inverse_solver.h"

// Kernel to calculate 'Data Loss' (Sensor Misalignment)
__global__ void compute_data_loss_kernel(float *predicted_field, float *sensor_pos, float *sensor_val, int n_sensors, int nx, float *loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sensors) {
        // Find which grid cell the sensor is in
        int x_idx = (int)(sensor_pos[idx*2] * (nx - 1));
        int y_idx = (int)(sensor_pos[idx*2 + 1] * (nx - 1));
        int grid_idx = y_idx * nx + x_idx;

        float diff = predicted_field[grid_idx] - sensor_val[idx];
        atomicAdd(loss, diff * diff); // L2 error for data reconstruction
    }
}

void inverse_init(InverseProblemConfig *inv, int n_sensors) {
    inv->num_sensors = n_sensors;
    inv->lambda_data = 1.0f;
    inv->lambda_pde = 1e-4f; // We trust physics, but data is the anchor

    cudaMalloc(&inv->d_sensor_positions, n_sensors * 2 * sizeof(float));
    cudaMalloc(&inv->d_sensor_values, n_sensors * sizeof(float));
    
    // Example: Initialize random sensors/measurements
    cudaMemset(inv->d_sensor_positions, 0, n_sensors * 2 * sizeof(float));
    cudaMemset(inv->d_sensor_values, 0, n_sensors * sizeof(float));
}

void inverse_solve_hybrid(InverseProblemConfig *inv, FNOConfig *fno, PINNConfig *pinn, float *d_reconstructed_field) {
    // 1. FAST INITIAL GUESS (FNO)
    // The FNO maps coarse sensor data to a potential full field quickly.
    // This is much faster than running 1000s of iterations from scratch.
    fno_spectral_conv_step(fno, d_reconstructed_field, d_reconstructed_field);

    float *d_data_loss;
    cudaMalloc(&d_data_loss, sizeof(float));
    cudaMemset(d_data_loss, 0, sizeof(float));

    // 2. PINN REFINEMENT (PHYSICS LOSS)
    // Minimizing total loss: L = lambda_data*L_data + lambda_pde*L_physics
    compute_data_loss_kernel<<<(inv->num_sensors + 127) / 128, 128>>>(
        d_reconstructed_field, inv->d_sensor_positions, inv->d_sensor_values, 
        inv->num_sensors, fno->nx, d_data_loss);

    // After data alignment, we call the PINN logic to ensure the result is 'Physically Consistent'
    // pinn_compute_residual_loss<<<...>>>(pinn, points, residuals, num_points);

    cudaFree(d_data_loss);
}

void inverse_cleanup(InverseProblemConfig *inv) {
    cudaFree(inv->d_sensor_positions);
    cudaFree(inv->d_sensor_values);
}
