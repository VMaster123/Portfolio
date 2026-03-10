import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_result(filename, nx=256, ny=256):
    """
    Visualizes raw binary CFD output files from the Neural-CFD project.
    Assumes binary files are float32 flat arrays.
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found. Run the simulation first!")
        return

    # Load data
    data = np.fromfile(filename, dtype=np.float32)
    
    # Reshape (Assuming 2D grid)
    try:
        data = data.reshape((ny, nx))
    except ValueError:
        print(f"Data size {data.size} doesn't match {nx}x{ny}. Truncating...")
        data = data[:nx*ny].reshape((ny, nx))

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='jet', extent=[0, 1, 0, 1], origin='lower')
    plt.colorbar(label='Velocity Magnitude / Pressure')
    plt.title('Neural-HPC-CFD Simulation Field (PINN-Informed)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('cfd_result.png', dpi=300)
    print("Vusalization saved to 'cfd_result.png'.")
    plt.show()

if __name__ == "__main__":
    # In a real run, the C code would dump files like 'output_u.bin'
    visualize_result('output_u.bin')
