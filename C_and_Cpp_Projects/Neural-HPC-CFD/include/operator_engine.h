#ifndef OPERATOR_ENGINE_H
#define OPERATOR_ENGINE_H

#include <cuda_runtime.h>
#include <cufft.h>

typedef struct {
    int nx, ny;
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    cufftComplex *d_field_fft; // Field in frequency domain
    float *d_weights_real;     // Learnable spectral weights
    float *d_weights_imag;
} FNOConfig;

// Initialize FNO with Fourier plans (FFT)
void fno_init(FNOConfig *fno, int nx, int ny);

// Perform a Spectral Convolution: FFT -> Multiply by Weights -> Inverse FFT
// This is the core of an FNO, replacing standard CNN layers
void fno_spectral_conv_step(FNOConfig *fno, float *d_input_field, float *d_output_field);

void fno_cleanup(FNOConfig *fno);

#endif
