#ifndef INVERSE_SOLVER_H
#define INVERSE_SOLVER_H

#include <cuda_runtime.h>
#include "pinn.h"
#include "operator_engine.h"

typedef struct {
    float *d_sensor_positions; // Sparse (x,y) coordinates of sensors
    float *d_sensor_values;    // Recorded measurements (e.g. Pressure)
    int num_sensors;
    float lambda_data;         // Weight for Data Loss
    float lambda_pde;          // Weight for Physics Loss
} InverseProblemConfig;

// Initialize the inverse problem setup
void inverse_init(InverseProblemConfig *inv, int n_sensors);

// Hybrid Solve: Use FNO as a PREDICTOR and PINN as a REFINER
// This solves the Inverse Problem: Reconstructing field from sparse data
void inverse_solve_hybrid(InverseProblemConfig *inv, FNOConfig *fno, PINNConfig *pinn, float *d_reconstructed_field);

void inverse_cleanup(InverseProblemConfig *inv);

#endif
