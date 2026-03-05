import pandas as pd
import matplotlib.pyplot as plt
import json
import subprocess
import os
import time

def run_validation_sim(duration=30):
    print(f"--- Running Validation Simulation ({duration}s) ---")
    exe_path = os.path.join("build", "Debug", "aerotwin_sim.exe")
    data = []
    
    process = subprocess.Popen([exe_path], stdout=subprocess.PIPE, text=True)
    start_time = time.time()
    
    while time.time() - start_time < duration:
        line = process.stdout.readline()
        if not line: break
        try:
            data.append(json.loads(line.strip()))
        except:
            continue
            
    process.kill()
    return pd.DataFrame(data)

def analyze(df):
    if df.empty:
        print("No data collected for validation.")
        return

    # 1. State Tracking RMSE
    # Target path is defined in C++ as 50m altitude
    target_alt = 50.0
    df['alt_error'] = target_alt - df['alt']
    rmse = (df['alt_error']**2).mean()**0.5
    
    print(f"\nVALIDATION REPORT:")
    print(f"------------------")
    print(f"Altitude Tracking RMSE: {rmse:.4f} m (Target: <0.5m)")
    
    # 2. CL Convergence
    # Final CL should stabilize around the vision-inferred value
    final_cl = df['cl'].iloc[-10:].mean()
    print(f"Steady-State CL Estimate: {final_cl:.4f}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(df['alt'], label='Actual Altitude', color='cyan')
    ax1.axhline(y=50, color='r', linestyle='--', label='Target')
    ax1.set_title("Altitude Tracking Stability")
    ax1.set_ylabel("Meters")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['cl'], label='Estimated CL', color='lime')
    ax2.set_title("Aerodynamic Parameter Convergence (Inverse Engine)")
    ax2.set_ylabel("Coefficient (CL)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("scripts/validation_results.png")
    print(f"Validation plot saved to scripts/validation_results.png")

if __name__ == "__main__":
    if not os.path.exists("scripts"):
        os.makedirs("scripts")
    df = run_validation_sim()
    analyze(df)
