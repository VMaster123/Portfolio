# AeroTwin: Robust ML & Simulation Validation Suite
# This script ensures that the C++ simulation, Python ML interface, and Telemetry systems are functional.

import subprocess
import json
import os
import sys

def test_simulation_output():
    """Checks if the C++ simulation is built and emitting telemetry."""
    exe_path = os.path.join("build", "Debug", "aerotwin_sim.exe")
    if not os.path.exists(exe_path):
        print("❌ Error: C++ Simulation binary NOT found. Build failed.")
        return False
    
    print("--- Validating C++ Simulation Output ---")
    try:
        process = subprocess.Popen([exe_path, "square"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Capture first 5 lines of telemetry
        outputs = []
        for _ in range(5):
            line = process.stdout.readline()
            if line:
                outputs.append(line.strip())
        process.kill()

        if not outputs:
            print("❌ Error: Simulation produced NO output.")
            return False

        # Validate JSON format
        for line in outputs:
            try:
                data = json.loads(line)
                keys = ["t", "px", "py", "pz", "vx", "cl", "conf", "res"]
                for k in keys:
                    if k not in data:
                        print(f"❌ Error: Missing key '{k}' in telemetry.")
                        return False
            except json.JSONDecodeError:
                print(f"❌ Error: Invalid JSON in telemetry: {line[:50]}...")
                return False
        
        print("✅ SUCCESS: C++ Simulation emitting valid JSON telemetry.")
        return True
    except Exception as e:
        print(f"❌ Error: Failed to run simulation: {e}")
        return False

def check_ml_environment():
    """Checks for PyTorch and required dependencies."""
    print("--- checking ML Training Environment ---")
    try:
        import torch
        print(f"✅ PyTorch v{torch.__version__} is available.")
        return True
    except ImportError:
        print("⚠️  Warning: PyTorch (torch) NOT found in this environment.")
        print("   -> Real Training is disabled. Run 'pip install torch' to enable it.")
        return False

def run_all_tests():
    print("==================================================")
    print("🚀 AeroTwin Integrated Portfolio Validation Suite")
    print("==================================================\n")
    
    sim_ok = test_simulation_output()
    ml_ok = check_ml_environment()
    
    print("\n--------------------------------------------------")
    if sim_ok:
        print("✅ OVERALL STATUS: AeroTwin Core is functional.")
    else:
        print("❌ OVERALL STATUS: System issues detected (Check build).")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
