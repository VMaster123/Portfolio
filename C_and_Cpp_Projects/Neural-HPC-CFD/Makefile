# Neural-HPC-CFD Makefile
# Optimized for HPC environments with CUDA and MPI

# Compilers
NVCC = nvcc
MPICC = mpicc

# Paths
INC = -Iinclude
LIB = -lmpi -lcublas -lcudart -lcufft

# Flags
NVCCFLAGS = -O3 -arch=sm_80 $(INC) # Target Ampere/HPC GPUs (A100/A30)
CFLAGS = -O3 $(INC)

# Artifacts
TARGET = neural_cfd
SRC = src/main.cu src/solver.cu src/pinn.cu src/operator_engine.cu src/inverse_solver.cu
OBJ = $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) $(LIB) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

# Helper to run on 2 GPUs locally (if MPI is present)
run: all
	mpirun -np 2 ./$(TARGET)
