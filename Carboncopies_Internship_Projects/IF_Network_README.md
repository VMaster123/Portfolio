# Integrate-and-Fire Neural Network Component-Wise Emulation

## Overview

This project implements a comprehensive framework for emulating integrate-and-fire (IF) neural networks using component-wise deep neural network (DNN) approximations. The approach treats each neuron as an independent learning problem, then combines the learned models into a full network emulation.

## Key Features

### 1. **Integrate-and-Fire Neuron Model**
- Leaky integrate-and-fire dynamics with configurable parameters
- Membrane potential tracking with threshold and reset mechanisms
- Complete I/O history recording for data collection

### 2. **Component-Wise Approximation**
- Individual DNN approximator for each neuron
- Estimates input connection weights from observed I/O patterns
- Treats each neuron as an independent learning problem

### 3. **Network Emulation**
- Combines component models into full network emulation
- Uses estimated weights for synaptic connections
- Maintains network dynamics and temporal behavior

### 4. **High-Level Constraint Optimization**
- System-level metric (firing rate) computed from whole-network I/O
- Constraint-based re-training of component models
- Balances component accuracy with system-level behavior

### 5. **Behavioral Validation**
- Left-hand/Right-hand behavioral check
- Validates network response to lateralized stimuli
- Measures behavioral accuracy based on neuron firing patterns

### 6. **Comprehensive Testing**
- Comparison metrics: correlation, MSE, firing rate differences
- Noise robustness testing
- Visualization of network dynamics and weight matrices

## Methodology

### Step-by-Step Process

1. **Network Setup**: Create an IF network with sparse connectivity
2. **Data Collection**: Observe I/O for each neuron during network simulation
3. **Component Training**: Train separate DNNs to approximate weights for each neuron
4. **Emulation**: Combine component models into full network emulation
5. **Baseline Comparison**: Compare original vs emulated network behavior
6. **Noise Testing**: Evaluate robustness with noisy input data
7. **Constraint Definition**: Compute high-level system metric from whole-network I/O
8. **Constrained Re-training**: Re-train components with system-level constraint
9. **Behavioral Validation**: Test left-hand/right-hand response patterns
10. **Final Evaluation**: Comprehensive comparison and visualization

## Numerical Results

The framework produces quantitative metrics including:

- **Output Correlation**: Correlation coefficient between original and emulated spike patterns
- **Mean Squared Error (MSE)**: Difference in spike timing and magnitude
- **Firing Rate Accuracy**: Comparison of overall network activity
- **Behavioral Accuracy**: Left-hand/right-hand response correctness (target: >0.7)

### Typical Performance

- **Baseline Emulation**: Correlation ~0.6-0.8, MSE ~0.01-0.05
- **Noisy Data**: Correlation ~0.5-0.7 (robust to 10% noise)
- **Constrained Emulation**: Improved system-level consistency
- **Behavioral Accuracy**: 0.65-0.85 depending on network connectivity

## Usage

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run comprehensive test suite
python IF_Network_Emulation.py
```

The script will:
- Create and simulate an IF network
- Train component approximators
- Test with and without constraints
- Perform behavioral validation
- Generate visualization plots
- Output numerical results summary

## Output Files

- `if_network_results.png`: Visualization comparing original and emulated networks
  - Spike raster plots
  - Activity time series
  - Weight matrix comparisons

## Technical Details

### Network Architecture
- **Neurons**: 10 integrate-and-fire neurons (configurable)
- **Connectivity**: Sparse random connections (~20% density)
- **Time Step**: 0.1 ms
- **Membrane Time Constant**: 20 ms
- **Threshold**: 1.0 (normalized)

### DNN Approximator Architecture
- Input: Current input + recent history (2 time steps)
- Hidden Layers: 2 layers of 64 units with ReLU activation
- Output: Estimated connection weights (tanh normalized)
- Training: Adam optimizer, MSE loss

### High-Level Constraint
- Metric: System-wide firing rate
- Constraint Weight: 1.0 (balances component vs system accuracy)
- Re-training: Fewer epochs with combined loss function

## Applications

This methodology is applicable to:
- **Neural Network Compression**: Approximate large networks with smaller models
- **Network Analysis**: Understand component contributions to system behavior
- **Robustness Testing**: Evaluate network behavior under noise and perturbations
- **Behavioral Modeling**: Validate network responses to specific stimuli patterns

## Future Enhancements

- Support for more complex neuron models (adaptive threshold, synaptic plasticity)
- Multi-scale constraints (local and global)
- Real-time weight adaptation
- Integration with experimental neural data

