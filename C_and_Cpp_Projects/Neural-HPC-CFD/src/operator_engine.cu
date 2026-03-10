#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "operator_engine.h"

// Kernel to apply spectral weights: Y(omega) = W(omega) * X(omega)
__global__ void apply_spectral_weights_kernel(cufftComplex *field, float *w_r, float *w_i, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int idx = j * nx + i;
        // Complex Multiplication: (A+Bi) * (C+Di) = (AC - BD) + (AD + BC)i
        float a = field[idx].x;
        float b = field[idx].y;
        float c = w_r[idx];
        float d = w_i[idx];

        field[idx].x = (a * c - b * d);
        field[idx].y = (a * d + b * c);
    }
}

void fno_init(FNOConfig *fno, int nx, int ny) {
    fno->nx = nx;
    fno->ny = ny;
    
    // Create plans for 2D Real-to-Complex and Complex-to-Real FFTs
    cufftPlan2d(&fno->plan_forward, ny, nx, CUFFT_R2C);
    cufftPlan2d(&fno->plan_inverse, ny, nx, CUFFT_C2R);

    size_t complex_size = nx * (ny / 2 + 1) * sizeof(cufftComplex);
    cudaMalloc(&fno->d_field_fft, complex_size);
    
    // Initialize random spectral weights for the operator to "learn"
    cudaMalloc(&fno->d_weights_real, complex_size);
    cudaMalloc(&fno->d_weights_imag, complex_size);
    cudaMemset(fno->d_weights_real, 0, complex_size); // Initialize somehow
}

void fno_spectral_conv_step(FNOConfig *fno, float *d_input_field, float *d_output_field) {
    // 1. Transform spatial domain to frequency domain
    cufftExecR2C(fno->plan_forward, (cufftReal*)d_input_field, fno->d_field_fft);

    // 2. High-Frequency Filtering / Weight Multiplication on GPU
    dim3 block(16, 16);
    dim3 grid((fno->nx + 15) / 16, (fno->ny / 2 + 1) / 16);
    apply_spectral_weights_kernel<<<grid, block>>>(fno->d_field_fft, fno->d_weights_real, fno->d_weights_imag, fno->nx, fno->ny / 2 + 1);

    // 3. Inverse FFT back to spatial domain
    cufftExecC2R(fno->plan_inverse, fno->d_field_fft, (cufftReal*)d_output_field);
    
    // 4. Normalize (cuFFT is unnormalized)
    // kernel<<<...>>>(d_output_field, 1.0f / (nx * ny));
}

void fno_cleanup(FNOConfig *fno) {
    cufftDestroy(fno->plan_forward);
    cufftDestroy(fno->plan_inverse);
    cudaFree(fno->d_field_fft);
    cudaFree(fno->d_weights_real);
    cudaFree(fno->d_weights_imag);
}
