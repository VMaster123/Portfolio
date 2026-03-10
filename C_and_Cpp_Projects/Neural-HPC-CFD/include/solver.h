#ifndef SOLVER_H
#define SOLVER_H

#include <cuda_runtime.h>
#include <mpi.h>

typedef struct {
    int nx, ny;         // Grid dimensions
    float dx, dy;       // Grid spacing
    float dt;           // Time step
    float visco;        // Kinematic viscosity (nu)
    float *u, *v, *p;   // Velocity and Pressure fields (host)
    float *d_u, *d_v, *d_p; // GPU device pointers
} SolverConfig;

// Initialize the CFD solver
void solver_init(SolverConfig *config, int nx, int ny);

// Run one time step of the simulation using CUDA
void solver_step(SolverConfig *config);

// Synchronize GPU boundaries using MPI
void solver_sync_mpi(SolverConfig *config);

// Free solver resources
void solver_cleanup(SolverConfig *config);

#endif
