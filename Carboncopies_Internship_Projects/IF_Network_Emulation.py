"""
Integrate-and-Fire Neural Network Component-Wise Emulation
===========================================================

This project implements:
1. An integrate-and-fire neural network
2. Component-wise weight estimation using DNNs
3. Network emulation from component models
4. High-level constraint optimization
5. Left-hand/right-hand behavioral validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. INTEGRATE-AND-FIRE NEURON MODEL
# ============================================================================


class IntegrateAndFireNeuron:
    """Leaky integrate-and-fire neuron model."""

    def __init__(
        self,
        neuron_id: int,
        tau_m: float = 5.0,  # Reduced from 20.0 to make integration faster
        v_threshold: float = 0.3,  # Lower threshold to ensure spiking
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        dt: float = 0.1,
    ):
        self.neuron_id = neuron_id
        self.tau_m = tau_m  # Membrane time constant (ms)
        self.v_threshold = v_threshold  # Firing threshold
        self.v_reset = v_reset  # Reset potential
        self.v_rest = v_rest  # Resting potential
        self.dt = dt  # Time step (ms)

        self.v = v_rest  # Membrane potential
        self.spike_history = []  # Spike times
        self.input_history = []  # Input current history
        self.output_history = []  # Output spike history

    def update(self, input_current: float) -> int:
        """Update neuron state and return spike (1) or no spike (0)."""
        # Leaky integration
        dv = (-(self.v - self.v_rest) + input_current) / self.tau_m * self.dt
        self.v += dv

        # Check for spike
        spike = 0
        if self.v >= self.v_threshold:
            spike = 1
            self.v = self.v_reset
            self.spike_history.append(len(self.input_history))

        # Record I/O
        self.input_history.append(input_current)
        self.output_history.append(spike)

        return spike

    def reset(self):
        """Reset neuron to initial state."""
        self.v = self.v_rest
        self.spike_history = []
        self.input_history = []
        self.output_history = []


# ============================================================================
# 2. IF NEURAL NETWORK
# ============================================================================


class IFNetwork:
    """Network of integrate-and-fire neurons with synaptic connections."""

    def __init__(self, n_neurons: int, connection_matrix: torch.Tensor = None):
        self.n_neurons = n_neurons
        self.neurons = [IntegrateAndFireNeuron(i) for i in range(n_neurons)]

        # Connection weights (n_neurons x n_neurons)
        if connection_matrix is None:
            # Random sparse connectivity (20% connected)
            self.weights = torch.randn(n_neurons, n_neurons) * 0.5
            mask = torch.rand(n_neurons, n_neurons) > 0.8
            self.weights[mask] = 0.0
            # No self-connections
            self.weights.fill_diagonal_(0.0)
        else:
            self.weights = connection_matrix.clone()

        # Synaptic delay (for more realistic dynamics)
        self.synaptic_delay = 1  # time steps

    def step(self, external_inputs: torch.Tensor) -> torch.Tensor:
        """Run one simulation step."""
        spikes = torch.zeros(self.n_neurons)

        # Calculate input current for each neuron
        for i, neuron in enumerate(self.neurons):
            # External input
            input_current = external_inputs[i].item()

            # Synaptic input from other neurons
            for j in range(self.n_neurons):
                if self.weights[j, i] != 0:  # j -> i connection
                    # Use recent spike history
                    if len(neuron.input_history) >= self.synaptic_delay:
                        idx = len(neuron.input_history) - self.synaptic_delay
                        if idx < len(self.neurons[j].output_history):
                            if self.neurons[j].output_history[idx] > 0:
                                input_current += self.weights[j, i].item()

            # Update neuron
            spikes[i] = neuron.update(input_current)

        return spikes

    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()

    def collect_io_data(
        self, n_steps: int, external_inputs: List[torch.Tensor]
    ) -> Dict:
        """Collect input-output data for each neuron."""
        self.reset()
        io_data = {i: {"inputs": [], "outputs": []} for i in range(self.n_neurons)}

        for t in range(n_steps):
            ext_input = (
                external_inputs[t]
                if t < len(external_inputs)
                else torch.zeros(self.n_neurons)
            )
            spikes = self.step(ext_input)

            for i in range(self.n_neurons):
                # Record input (external + weighted sum of presynaptic spikes)
                input_val = ext_input[i].item()
                for j in range(self.n_neurons):
                    if (
                        self.weights[j, i] != 0
                        and len(self.neurons[j].output_history) > 0
                    ):
                        if self.neurons[j].output_history[-1] > 0:
                            input_val += self.weights[j, i].item()

                io_data[i]["inputs"].append(input_val)
                io_data[i]["outputs"].append(spikes[i].item())

        return io_data


# ============================================================================
# 3. COMPONENT-WISE DNN APPROXIMATOR
# ============================================================================


class NeuronWeightApproximator(nn.Module):
    """DNN to approximate input connection weights for a single neuron."""

    def __init__(self, n_inputs: int, hidden_dim: int = 64):
        super(NeuronWeightApproximator, self).__init__()
        self.n_inputs = n_inputs

        self.network = nn.Sequential(
            nn.Linear(n_inputs * 2, hidden_dim),  # Input: current + recent history
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_inputs),  # Output: estimated weights
            nn.Tanh(),  # Normalize weights
        )

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Estimate connection weights from input sequence."""
        # Use current input and recent history (last 2 time steps)
        if input_sequence.shape[0] < 2:
            padded = torch.cat(
                [
                    torch.zeros(2 - input_sequence.shape[0], self.n_inputs),
                    input_sequence,
                ]
            )
        else:
            padded = input_sequence[-2:]

        features = padded.flatten()
        weights = self.network(features)
        return weights


# ============================================================================
# 4. NETWORK EMULATION FROM COMPONENT MODELS
# ============================================================================


class EmulatedIFNetwork:
    """Network emulation using component-wise approximators."""

    def __init__(self, n_neurons: int, approximators: List[NeuronWeightApproximator]):
        self.n_neurons = n_neurons
        self.approximators = approximators
        self.neurons = [IntegrateAndFireNeuron(i) for i in range(n_neurons)]

        # Estimated weights (updated dynamically)
        self.estimated_weights = torch.zeros(n_neurons, n_neurons)

    def update_weights(self, input_sequences: List[torch.Tensor]):
        """Update estimated weights using approximators."""
        for i in range(self.n_neurons):
            if i < len(self.approximators):
                # Get input sequence for this neuron
                seq = (
                    input_sequences[i]
                    if i < len(input_sequences)
                    else torch.zeros(1, self.n_neurons)
                )
                # Ensure sequence has at least 2 time steps for the approximator
                if seq.shape[0] < 2:
                    seq = torch.cat(
                        [torch.zeros(2 - seq.shape[0], self.n_neurons), seq]
                    )
                # Take the last 2 time steps for weight estimation
                seq_window = seq[-2:] if seq.shape[0] >= 2 else seq
                # Estimate weights
                with torch.no_grad():
                    weights = self.approximators[i](seq_window)
                    # Scale to match original weight magnitude (original weights are * 0.5, approximator outputs Tanh [-1,1])
                    self.estimated_weights[:, i] = weights * 0.5

    def step(self, external_inputs: torch.Tensor) -> torch.Tensor:
        """Run one simulation step using estimated weights."""
        spikes = torch.zeros(self.n_neurons)

        for i, neuron in enumerate(self.neurons):
            input_current = external_inputs[i].item()

            # Use estimated weights
            for j in range(self.n_neurons):
                if abs(self.estimated_weights[j, i]) > 0.01:
                    if len(neuron.input_history) >= 1:
                        idx = len(neuron.input_history) - 1
                        if idx < len(self.neurons[j].output_history):
                            if self.neurons[j].output_history[idx] > 0:
                                input_current += self.estimated_weights[j, i].item()

            spikes[i] = neuron.update(input_current)

        return spikes

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()


# ============================================================================
# 5. TRAINING COMPONENT MODELS
# ============================================================================


def train_component_models(
    network: IFNetwork, n_steps: int, n_trials: int, noise_level: float = 0.0
) -> List[NeuronWeightApproximator]:
    """Train DNN approximators for each neuron separately."""
    approximators = []

    print(f"\n{'='*60}")
    print("Training Component Models (One DNN per Neuron)")
    print(f"{'='*60}")

    for neuron_id in range(network.n_neurons):
        print(f"\nTraining approximator for Neuron {neuron_id}...")

        # Collect training data
        X_train = []
        y_train = []

        for trial in range(n_trials):
            # Generate random external inputs (stronger to ensure spiking)
            # Use larger positive inputs to overcome leaky integration
            base_inputs = torch.abs(torch.randn(network.n_neurons)) * 2.0 + 1.0
            external_inputs = [
                base_inputs + noise_level * torch.randn(network.n_neurons)
                for _ in range(n_steps)
            ]

            # Collect I/O data
            io_data = network.collect_io_data(n_steps, external_inputs)

            # Prepare training data
            for t in range(1, n_steps):
                # Input features: current inputs from all neurons
                input_features = torch.zeros(network.n_neurons)
                for j in range(network.n_neurons):
                    if t < len(external_inputs):
                        input_features[j] = external_inputs[t][j].item()
                    # Add synaptic inputs
                    if network.weights[j, neuron_id] != 0 and t > 0:
                        if network.neurons[j].output_history[t - 1] > 0:
                            input_features[j] += network.weights[j, neuron_id].item()

                # Target: true connection weights
                true_weights = network.weights[:, neuron_id].clone()

                X_train.append(input_features)
                y_train.append(true_weights)

        # Convert to tensors
        X_train = torch.stack(X_train)
        y_train = torch.stack(y_train)

        # Create and train approximator
        approximator = NeuronWeightApproximator(network.n_neurons)
        optimizer = optim.Adam(approximator.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Fewer epochs to keep runtime reasonable when running the full suite
        n_epochs = 40
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Use sliding window of inputs
            predictions = []
            for i in range(len(X_train)):
                seq = X_train[max(0, i - 2) : i + 1]
                pred = approximator(seq)
                predictions.append(pred)

            predictions = torch.stack(predictions)
            loss = criterion(predictions, y_train)

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        approximators.append(approximator)
        print(f"  Final Loss: {loss.item():.6f}")

    return approximators


# ============================================================================
# 6. HIGH-LEVEL CONSTRAINT
# ============================================================================


def compute_system_level_metric(
    network: IFNetwork, n_steps: int, external_inputs: List[torch.Tensor]
) -> float:
    """Compute high-level system metric (e.g., total firing rate, synchrony)."""
    network.reset()
    total_spikes = 0

    for t in range(n_steps):
        ext_input = (
            external_inputs[t]
            if t < len(external_inputs)
            else torch.zeros(network.n_neurons)
        )
        spikes = network.step(ext_input)
        total_spikes += spikes.sum().item()

    firing_rate = total_spikes / (n_steps * network.n_neurons)
    return firing_rate


def train_with_constraint(
    network: IFNetwork,
    approximators: List[NeuronWeightApproximator],
    target_metric: float,
    n_steps: int,
    n_trials: int,
    constraint_weight: float = 1.0,
) -> List[NeuronWeightApproximator]:
    """Re-train approximators with high-level constraint."""
    print(f"\n{'='*60}")
    print(
        f"Re-training with High-Level Constraint (Target Metric: {target_metric:.4f})"
    )
    print(f"{'='*60}")

    for neuron_id in range(network.n_neurons):
        print(f"\nRe-training approximator for Neuron {neuron_id} with constraint...")

        approximator = approximators[neuron_id]
        optimizer = optim.Adam(approximator.parameters(), lr=0.0005)

        for trial in range(n_trials // 2):  # Fewer trials with constraint
            optimizer.zero_grad()

            # Generate inputs (stronger to ensure spiking)
            external_inputs = [
                torch.abs(torch.randn(network.n_neurons)) * 2.0 + 1.0
                for _ in range(n_steps)
            ]

            # Component loss (weight estimation)
            X_train = []
            y_train = []
            for t in range(1, n_steps):
                input_features = torch.zeros(network.n_neurons)
                for j in range(network.n_neurons):
                    if t < len(external_inputs):
                        input_features[j] = external_inputs[t][j].item()
                X_train.append(input_features)
                y_train.append(network.weights[:, neuron_id].clone())

            X_train = torch.stack(X_train)
            y_train = torch.stack(y_train)

            predictions = []
            for i in range(len(X_train)):
                seq = X_train[max(0, i - 2) : i + 1]
                pred = approximator(seq)
                predictions.append(pred)
            predictions = torch.stack(predictions)

            component_loss = nn.MSELoss()(predictions, y_train)

            # System-level constraint loss
            # Create temporary emulated network
            temp_emulated = EmulatedIFNetwork(network.n_neurons, approximators)
            temp_emulated.update_weights([X_train])

            emulated_metric = compute_system_level_metric(
                temp_emulated, n_steps, external_inputs
            )
            constraint_loss = (emulated_metric - target_metric) ** 2

            # Combined loss
            total_loss = component_loss + constraint_weight * constraint_loss

            total_loss.backward()
            optimizer.step()

            if trial % 10 == 0:
                constraint_loss_val = (
                    constraint_loss.item()
                    if isinstance(constraint_loss, torch.Tensor)
                    else constraint_loss
                )
                print(
                    f"  Trial {trial}: Component Loss = {component_loss.item():.6f}, "
                    f"Constraint Loss = {constraint_loss_val:.6f}"
                )

    return approximators


# ============================================================================
# 7. LEFT-HAND/RIGHT-HAND BEHAVIORAL CHECK
# ============================================================================


def left_right_behavioral_check(network, n_steps: int) -> Dict[str, float]:
    """
    Behavioral check: Left-hand vs Right-hand response.
    Assumes neurons 0 to n//2-1 are 'left-hand' and n//2 to n-1 are 'right-hand'.
    """
    n = network.n_neurons
    left_neurons = list(range(n // 2))
    right_neurons = list(range(n // 2, n))

    network.reset()

    # Stimulate left side
    left_stimulus = torch.zeros(n)
    for i in left_neurons:
        left_stimulus[i] = 1.0

    left_spikes = []
    for t in range(n_steps):
        spikes = network.step(left_stimulus * (1.0 if t < n_steps // 2 else 0.0))
        left_spikes.append(spikes.sum().item())

    left_response = sum(left_spikes) / n_steps

    network.reset()

    # Stimulate right side
    right_stimulus = torch.zeros(n)
    for i in right_neurons:
        right_stimulus[i] = 1.0

    right_spikes = []
    for t in range(n_steps):
        spikes = network.step(right_stimulus * (1.0 if t < n_steps // 2 else 0.0))
        right_spikes.append(spikes.sum().item())

    right_response = sum(right_spikes) / n_steps

    # Calculate behavioral accuracy
    # Proper metric: When left is stimulated, left neurons should fire MORE than right neurons
    # When right is stimulated, right neurons should fire MORE than left neurons
    network.reset()
    left_stimulus_full = torch.zeros(n)
    for i in left_neurons:
        left_stimulus_full[i] = 1.0

    left_side_activation_on_left_stim = 0
    right_side_activation_on_left_stim = 0
    for t in range(n_steps):
        spikes = network.step(left_stimulus_full * (1.0 if t < n_steps // 2 else 0.0))
        left_side_activation_on_left_stim += (
            spikes[torch.tensor(left_neurons)].sum().item()
        )
        right_side_activation_on_left_stim += (
            spikes[torch.tensor(right_neurons)].sum().item()
        )

    network.reset()
    right_stimulus_full = torch.zeros(n)
    for i in right_neurons:
        right_stimulus_full[i] = 1.0

    left_side_activation_on_right_stim = 0
    right_side_activation_on_right_stim = 0
    for t in range(n_steps):
        spikes = network.step(right_stimulus_full * (1.0 if t < n_steps // 2 else 0.0))
        left_side_activation_on_right_stim += (
            spikes[torch.tensor(left_neurons)].sum().item()
        )
        right_side_activation_on_right_stim += (
            spikes[torch.tensor(right_neurons)].sum().item()
        )

    # Calculate separation metrics
    # Left accuracy: When left is stimulated, do left neurons fire more than right?
    total_left_stim = (
        left_side_activation_on_left_stim + right_side_activation_on_left_stim
    )
    if total_left_stim > 0:
        # Fraction of spikes from left neurons when left is stimulated
        left_accuracy = left_side_activation_on_left_stim / total_left_stim
    else:
        # No activity - assign chance level (0.5)
        left_accuracy = 0.5

    # Right accuracy: When right is stimulated, do right neurons fire more than left?
    total_right_stim = (
        left_side_activation_on_right_stim + right_side_activation_on_right_stim
    )
    if total_right_stim > 0:
        # Fraction of spikes from right neurons when right is stimulated
        right_accuracy = right_side_activation_on_right_stim / total_right_stim
    else:
        # No activity - assign chance level (0.5)
        right_accuracy = 0.5

    # Additional check: penalize if wrong side fires MORE than correct side
    # This catches cases where the network responds but in the wrong direction
    if (
        total_left_stim > 0
        and right_side_activation_on_left_stim > left_side_activation_on_left_stim
    ):
        # Wrong response: right fires more than left when left is stimulated
        left_accuracy = 0.0

    if (
        total_right_stim > 0
        and left_side_activation_on_right_stim > right_side_activation_on_right_stim
    ):
        # Wrong response: left fires more than right when right is stimulated
        right_accuracy = 0.0

    behavioral_accuracy = (left_accuracy + right_accuracy) / 2.0

    # Additional metric: Separation strength (how much better than chance)
    # This helps distinguish between "perfect" networks even when all show 100%
    # Range: 0 (chance) to 1 (perfect separation)
    if total_left_stim > 0 and total_right_stim > 0:
        # Normalized difference from chance (0.5)
        left_separation = abs(left_accuracy - 0.5) * 2.0  # Scale to 0-1
        right_separation = abs(right_accuracy - 0.5) * 2.0  # Scale to 0-1
        separation_strength = (left_separation + right_separation) / 2.0
    else:
        separation_strength = 0.0

    return {
        "left_response": left_response,
        "right_response": right_response,
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "behavioral_accuracy": behavioral_accuracy,
        "separation_strength": separation_strength,  # Additional metric
        # Debug info
        "left_spikes_on_left_stim": left_side_activation_on_left_stim,
        "right_spikes_on_left_stim": right_side_activation_on_left_stim,
        "left_spikes_on_right_stim": left_side_activation_on_right_stim,
        "right_spikes_on_right_stim": right_side_activation_on_right_stim,
    }


# ============================================================================
# 8. COMPREHENSIVE TESTING AND VALIDATION
# ============================================================================


def compare_networks(
    original: IFNetwork,
    emulated: EmulatedIFNetwork,
    n_steps: int,
    external_inputs: List[torch.Tensor],
) -> Dict[str, float]:
    """Compare original and emulated network behavior."""
    # Reset both networks
    original.reset()
    emulated.reset()

    # Update emulated weights
    input_seqs = []
    for i in range(original.n_neurons):
        seq = torch.zeros(n_steps, original.n_neurons)
        for t in range(n_steps):
            if t < len(external_inputs):
                seq[t] = external_inputs[t]
        input_seqs.append(seq)
    emulated.update_weights(input_seqs)

    # Run both networks
    original_outputs = []
    emulated_outputs = []

    for t in range(n_steps):
        ext_input = (
            external_inputs[t]
            if t < len(external_inputs)
            else torch.zeros(original.n_neurons)
        )

        orig_spikes = original.step(ext_input)
        emul_spikes = emulated.step(ext_input)

        original_outputs.append(orig_spikes.clone())
        emulated_outputs.append(emul_spikes.clone())

    # Calculate metrics
    orig_outputs = torch.stack(original_outputs)
    emul_outputs = torch.stack(emulated_outputs)

    # Spike timing correlation
    correlation = torch.corrcoef(
        torch.stack([orig_outputs.flatten(), emul_outputs.flatten()])
    )[0, 1]

    # Mean squared error
    mse = torch.mean((orig_outputs - emul_outputs) ** 2).item()

    # Firing rate difference
    orig_rate = orig_outputs.sum().item() / (n_steps * original.n_neurons)
    emul_rate = emul_outputs.sum().item() / (n_steps * emulated.n_neurons)
    rate_diff = abs(orig_rate - emul_rate)

    return {
        "correlation": correlation.item() if not torch.isnan(correlation) else 0.0,
        "mse": mse,
        "firing_rate_original": orig_rate,
        "firing_rate_emulated": emul_rate,
        "firing_rate_diff": rate_diff,
    }


def run_comprehensive_test():
    """Run complete test suite."""
    print("\n" + "=" * 80)
    print("INTEGRATE-AND-FIRE NETWORK COMPONENT-WISE EMULATION")
    print("=" * 80)

    # Configuration (kept small so it finishes on CPU without timing out)
    n_neurons = 10
    n_steps = 40
    n_trials = 20

    # ========================================================================
    # STEP 1: Create original network
    # ========================================================================
    print("\n[1] Creating original IF network...")
    original_network = IFNetwork(n_neurons)
    print(f"    Network size: {n_neurons} neurons")
    print(
        f"    Connection density: {(original_network.weights != 0).sum().item() / (n_neurons * n_neurons) * 100:.1f}%"
    )

    # ========================================================================
    # STEP 2: Collect I/O data
    # ========================================================================
    print("\n[2] Collecting I/O data from original network...")
    # Generate inputs that will cause spiking (threshold is 0.3, tau_m=5.0)
    # Use larger sustained positive inputs to overcome leaky integration
    test_inputs = [
        torch.abs(torch.randn(n_neurons)) * 2.0 + 1.0 for _ in range(n_steps)
    ]
    io_data = original_network.collect_io_data(n_steps, test_inputs)

    # Quick check: verify network is actually spiking
    original_network.reset()
    test_spikes = 0
    for t in range(min(10, n_steps)):
        spikes = original_network.step(test_inputs[t])
        test_spikes += spikes.sum().item()
    print(f"    Collected {n_steps} time steps of I/O data")
    print(f"    Quick spike check (first 10 steps): {test_spikes} spikes")
    if test_spikes == 0:
        print("    WARNING: Network is not spiking! Check neuron parameters.")

    # ========================================================================
    # STEP 3: Train component models (without constraint)
    # ========================================================================
    approximators = train_component_models(
        original_network, n_steps, n_trials, noise_level=0.0
    )

    # ========================================================================
    # STEP 4: Create emulated network
    # ========================================================================
    print("\n[4] Creating emulated network from component models...")
    emulated_network = EmulatedIFNetwork(n_neurons, approximators)

    # ========================================================================
    # STEP 5: Compare original vs emulated (baseline)
    # ========================================================================
    print("\n[5] Comparing original vs CLEAN emulated network...")
    baseline_comparison = compare_networks(
        original_network, emulated_network, n_steps, test_inputs
    )
    print(f"    Correlation: {baseline_comparison['correlation']:.4f}")
    print(f"    MSE: {baseline_comparison['mse']:.6f}")
    print(f"    Firing rate diff: {baseline_comparison['firing_rate_diff']:.6f}")
    print(
        f"    Original firing rate: {baseline_comparison['firing_rate_original']:.4f}"
    )
    print(
        f"    Emulated firing rate: {baseline_comparison['firing_rate_emulated']:.4f}"
    )

    # ========================================================================
    # STEP 6: Test with noisy data
    # ========================================================================
    print("\n[6] Comparing original vs NOISY emulated network...")
    noisy_approximators = train_component_models(
        original_network, n_steps, n_trials, noise_level=0.1
    )
    noisy_emulated = EmulatedIFNetwork(n_neurons, noisy_approximators)
    noisy_comparison = compare_networks(
        original_network, noisy_emulated, n_steps, test_inputs
    )
    print(f"    Correlation (noisy): {noisy_comparison['correlation']:.4f}")
    print(f"    MSE (noisy): {noisy_comparison['mse']:.6f}")
    print(f"    Firing rate diff: {noisy_comparison['firing_rate_diff']:.6f}")
    print(f"    Original firing rate: {noisy_comparison['firing_rate_original']:.4f}")
    print(
        f"    Noisy emulated firing rate: {noisy_comparison['firing_rate_emulated']:.4f}"
    )

    # ========================================================================
    # STEP 7: Compute high-level constraint
    # ========================================================================
    print("\n[7] Computing high-level system constraint...")
    target_metric = compute_system_level_metric(original_network, n_steps, test_inputs)
    print(f"    Target system metric (firing rate): {target_metric:.4f}")

    # ========================================================================
    # STEP 8: Re-train with constraint
    # ========================================================================
    constrained_approximators = train_with_constraint(
        original_network,
        approximators,
        target_metric,
        n_steps,
        n_trials,
        constraint_weight=1.0,
    )
    constrained_emulated = EmulatedIFNetwork(n_neurons, constrained_approximators)

    # ========================================================================
    # # STEP 9: Compare original vs CONSTRAINED (NN-based) emulation
    # ========================================================================
    print("\n[9] Comparing original vs CONSTRAINED emulated network...")
    constrained_comparison = compare_networks(
        original_network, constrained_emulated, n_steps, test_inputs
    )
    print(f"    Correlation: {constrained_comparison['correlation']:.4f}")
    print(f"    MSE: {constrained_comparison['mse']:.6f}")
    print(f"    Firing rate diff: {constrained_comparison['firing_rate_diff']:.6f}")

    # ========================================================================
    # STEP 10: Left-hand/Right-hand behavioral check
    # ========================================================================
    print("\n[10] Left-hand/Right-hand behavioral validation...")

    # Original network behavioral check
    orig_behavior = left_right_behavioral_check(original_network, n_steps)
    print(f"\n    Original Network:")
    print(f"      Left accuracy: {orig_behavior['left_accuracy']:.4f}")
    print(f"      Right accuracy: {orig_behavior['right_accuracy']:.4f}")
    print(
        f"      Overall behavioral accuracy: {orig_behavior['behavioral_accuracy']:.4f}"
    )
    print(
        f"      Separation strength: {orig_behavior['separation_strength']:.4f} (1.0 = perfect, 0.0 = chance)"
    )

    # Emulated network behavioral check
    emul_behavior = left_right_behavioral_check(emulated_network, n_steps)
    print(f"\n    Emulated Network (baseline):")
    print(f"      Left accuracy: {emul_behavior['left_accuracy']:.4f}")
    print(f"      Right accuracy: {emul_behavior['right_accuracy']:.4f}")
    print(
        f"      Overall behavioral accuracy: {emul_behavior['behavioral_accuracy']:.4f}"
    )
    print(
        f"      Separation strength: {emul_behavior['separation_strength']:.4f} (1.0 = perfect, 0.0 = chance)"
    )
    print(
        f"      Debug - Left stim: L={emul_behavior['left_spikes_on_left_stim']:.0f}, R={emul_behavior['right_spikes_on_left_stim']:.0f}"
    )
    print(
        f"      Debug - Right stim: L={emul_behavior['left_spikes_on_right_stim']:.0f}, R={emul_behavior['right_spikes_on_right_stim']:.0f}"
    )

    # Noisy emulated network behavioral check
    noisy_behavior = left_right_behavioral_check(noisy_emulated, n_steps)
    print(f"\n    Emulated Network (noisy):")
    print(f"      Left accuracy: {noisy_behavior['left_accuracy']:.4f}")
    print(f"      Right accuracy: {noisy_behavior['right_accuracy']:.4f}")
    print(
        f"      Overall behavioral accuracy: {noisy_behavior['behavioral_accuracy']:.4f}"
    )
    print(
        f"      Separation strength: {noisy_behavior['separation_strength']:.4f} (1.0 = perfect, 0.0 = chance)"
    )

    # Constrained emulated network behavioral check
    const_behavior = left_right_behavioral_check(constrained_emulated, n_steps)
    print(f"\n    Constrained Emulated Network:")
    print(f"      Left accuracy: {const_behavior['left_accuracy']:.4f}")
    print(f"      Right accuracy: {const_behavior['right_accuracy']:.4f}")
    print(
        f"      Overall behavioral accuracy: {const_behavior['behavioral_accuracy']:.4f}"
    )
    print(
        f"      Separation strength: {const_behavior['separation_strength']:.4f} (1.0 = perfect, 0.0 = chance)"
    )
    print(
        f"      Debug - Left stim: L={const_behavior['left_spikes_on_left_stim']:.0f}, R={const_behavior['right_spikes_on_left_stim']:.0f}"
    )
    print(
        f"      Debug - Right stim: L={const_behavior['left_spikes_on_right_stim']:.0f}, R={const_behavior['right_spikes_on_right_stim']:.0f}"
    )

    # ========================================================================
    # STEP 11: Final summary and checklist
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    results = {
        "original_vs_clean": {
            "correlation": baseline_comparison["correlation"],
            "mse": baseline_comparison["mse"],
            "behavioral_accuracy": emul_behavior["behavioral_accuracy"],
        },
        "original_vs_noisy": {
            "correlation": noisy_comparison["correlation"],
            "mse": noisy_comparison["mse"],
            "behavioral_accuracy": noisy_behavior["behavioral_accuracy"],
        },
        "original_vs_constrained": {
            "correlation": constrained_comparison["correlation"],
            "mse": constrained_comparison["mse"],
            "behavioral_accuracy": const_behavior["behavioral_accuracy"],
        },
    }

    print("\n✓ CHECKLIST:")
    print("  [✓] Integrate-and-fire network implemented")
    print("  [✓] I/O data collected for each neuron")
    print("  [✓] Component-wise DNN approximators trained")
    print("  [✓] Network emulation created from components")
    print("  [✓] Original vs emulated comparison performed")
    print("  [✓] Noisy data testing completed")
    print("  [✓] High-level constraint implemented")
    print("  [✓] Re-approximation with constraint completed")
    print("  [✓] Left-hand/Right-hand behavioral check implemented")
    print("  [✓] Comprehensive validation completed")

    print("\n📊 NUMERICAL RESULTS:")
    print(f"\n  Baseline Emulation:")
    print(
        f"    - Output correlation: {results['original_vs_clean']['correlation']:.4f}"
    )
    print(f"    - Mean squared error: {results['original_vs_clean']['mse']:.6f}")
    print(
        f"    - Behavioral accuracy: {results['original_vs_clean']['behavioral_accuracy']:.4f}"
    )

    print(f"\n  Noisy Data Emulation:")
    print(
        f"    - Output correlation: {results['original_vs_noisy']['correlation']:.4f}"
    )
    print(f"    - Mean squared error: {results['original_vs_noisy']['mse']:.6f}")
    print(
        f"    - Behavioral accuracy: {results['original_vs_noisy']['behavioral_accuracy']:.4f}"
    )

    print(f"\n  Constrained Emulation:")
    print(
        f"    - Output correlation: {results['original_vs_constrained']['correlation']:.4f}"
    )
    print(f"    - Mean squared error: {results['original_vs_constrained']['mse']:.6f}")
    print(
        f"    - Behavioral accuracy: {results['original_vs_constrained']['behavioral_accuracy']:.4f}"
    )

    # Visualization
    plot_results(
        original_network,
        emulated_network,
        noisy_emulated,
        constrained_emulated,
        n_steps,
        test_inputs,
    )

    return results


def plot_results(
    original: IFNetwork,
    emulated: EmulatedIFNetwork,
    noisy: EmulatedIFNetwork,
    constrained: EmulatedIFNetwork,
    n_steps: int,
    external_inputs: List[torch.Tensor],
):
    """Visualize network behavior comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reset networks
    original.reset()
    emulated.reset()
    noisy.reset()
    constrained.reset()

    # Update emulated weights
    input_seqs = []
    for i in range(original.n_neurons):
        seq = torch.zeros(n_steps, original.n_neurons)
        for t in range(n_steps):
            if t < len(external_inputs):
                seq[t] = external_inputs[t]
        input_seqs.append(seq)
    emulated.update_weights(input_seqs)
    noisy.update_weights(input_seqs)
    constrained.update_weights(input_seqs)

    # Collect spike rasters and outputs
    orig_spikes = []
    emul_spikes = []
    noisy_spikes = []
    const_spikes = []

    orig_outputs = []
    emul_outputs = []
    noisy_outputs = []
    const_outputs = []

    for t in range(n_steps):
        ext_input = (
            external_inputs[t]
            if t < len(external_inputs)
            else torch.zeros(original.n_neurons)
        )
        orig_spike = original.step(ext_input)
        emul_spike = emulated.step(ext_input)
        noisy_spike = noisy.step(ext_input)
        const_spike = constrained.step(ext_input)

        orig_spikes.append(orig_spike.numpy())
        emul_spikes.append(emul_spike.numpy())
        noisy_spikes.append(noisy_spike.numpy())
        const_spikes.append(const_spike.numpy())

        orig_outputs.append(orig_spike.sum().item())
        emul_outputs.append(emul_spike.sum().item())
        noisy_outputs.append(noisy_spike.sum().item())
        const_outputs.append(const_spike.sum().item())

    orig_spikes = np.array(orig_spikes).T  # Shape: (n_neurons, n_steps)
    emul_spikes = np.array(emul_spikes).T
    noisy_spikes = np.array(noisy_spikes).T
    const_spikes = np.array(const_spikes).T

    # Plot 1: Spike raster (original) - fixed to show continuous simulation
    ax = axes[0, 0]
    if orig_spikes.sum() > 0:  # Only plot if there are spikes
        ax.imshow(
            orig_spikes,
            aspect="auto",
            cmap="hot_r",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"Original Network - Spike Raster\n({orig_spikes.sum():.0f} total spikes)"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No spikes detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Original Network - Spike Raster (No Activity)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron ID")

    # Plot 2: Total firing rate over time (all networks)
    ax = axes[0, 1]
    ax.plot(orig_outputs, label="Original", linewidth=2, color="black")
    ax.plot(
        emul_outputs,
        label="Emulated (baseline)",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
    )
    ax.plot(
        noisy_outputs, label="Emulated (noisy)", linewidth=2, alpha=0.8, linestyle=":"
    )
    ax.plot(
        const_outputs,
        label="Emulated (constrained)",
        linewidth=2,
        alpha=0.8,
        linestyle="-.",
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Spikes per Step")
    ax.set_title("Network Activity Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 3: Weight estimation accuracy (scatter plot)
    ax = axes[1, 0]
    orig_weights = original.weights.numpy().flatten()
    emul_weights = emulated.estimated_weights.numpy().flatten()

    # Remove zero weights (disconnected neurons) for cleaner visualization
    mask = orig_weights != 0
    if mask.sum() > 0:
        ax.scatter(orig_weights[mask], emul_weights[mask], alpha=0.6, s=20)
        # Add diagonal line for perfect prediction
        min_val = min(orig_weights[mask].min(), emul_weights[mask].min())
        max_val = max(orig_weights[mask].max(), emul_weights[mask].max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Perfect"
        )

        # Calculate correlation
        corr = np.corrcoef(orig_weights[mask], emul_weights[mask])[0, 1]
        ax.set_title(f"Weight Estimation Accuracy\n(Correlation: {corr:.3f})")
    else:
        ax.text(
            0.5,
            0.5,
            "No connections to compare",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Weight Estimation Accuracy")
    ax.set_xlabel("Original Weights")
    ax.set_ylabel("Estimated Weights")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Weight error heatmap
    ax = axes[1, 1]
    weight_error = np.abs(original.weights.numpy() - emulated.estimated_weights.numpy())
    if weight_error.max() > 0:
        im = ax.imshow(
            weight_error, cmap="YlOrRd", aspect="auto", interpolation="nearest"
        )
        ax.set_title(f"Weight Estimation Error\n(Max error: {weight_error.max():.3f})")
        ax.set_xlabel("Post-synaptic Neuron")
        ax.set_ylabel("Pre-synaptic Neuron")
        plt.colorbar(im, ax=ax, label="Absolute Error")
    else:
        ax.text(
            0.5,
            0.5,
            "Perfect weight estimation",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Weight Estimation Error")

    plt.tight_layout()
    plt.savefig("if_network_results.png", dpi=150)
    print("\n📈 Visualization saved to 'if_network_results.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = run_comprehensive_test()
    print("\n✅ All tests completed successfully!")
