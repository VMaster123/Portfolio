import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from scipy.special import expit


class IFNeuron:
    def __init__(
        self,
        weights=None,
        gain=0.1,
        neuron_idx=None,
        dt=1,
        tau=20.0,
        tau_syn=5.0,
        V_rest=-65.0,
        V_reset=-70.0,
        V_th=-35.0,
        R=10.0,
        refractory_period=10.0,
        ou_theta=0.5,
        ou_mu=0.2,
        ou_sigma=2,
        **kwargs
    ):
        self.weights = np.array(weights) if weights is not None else None
        self.gain = gain
        self.neuron_idx = neuron_idx
        self.dt = dt
        self.tau = tau
        self.tau_syn = tau_syn
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.V = V_rest
        self.refractory_timer = 0.0
        self.refractory_period = refractory_period
        self.spiked = False

        # Precompute constants for voltage update
        self.a = -1 / tau
        self.exp_ah = np.exp(self.a * dt)
        self.phi_1 = (1 - self.exp_ah) / self.a

        # OU noise initialization
        self.ou_theta = ou_theta
        self.ou_mu = ou_mu
        self.ou_sigma = ou_sigma
        self.noise_ou = 0.3

        # Synaptic current initialization
        self.I_syn = 0.0
        self.exp_syn = np.exp(-dt / tau_syn)

    def step(self, inputs):
        if self.refractory_timer > 0:
            self.refractory_timer -= self.dt
            self.V = self.V_reset
            self.spiked = False
            return False, self.V

        # Update OU noise
        dW = np.random.normal(0, np.sqrt(self.dt))
        self.noise_ou += (
            self.ou_theta * (self.ou_mu - self.noise_ou) * self.dt + self.ou_sigma * dW
        )

        # Update synaptic current with weighted sum of spikes and decay
        if self.weights is not None:
            weighted_spikes = np.dot(self.weights, inputs)  # sum_j w_ij * spike_j(t)
        else:
            weighted_spikes = 0.0

        self.I_syn = self.I_syn * self.exp_syn + weighted_spikes

        # Total input current with gain and OU noise
        I_total = self.gain * self.I_syn + self.noise_ou

        # Voltage update (exact LIF solution)
        b_t = (self.R * I_total + self.V_rest) / self.tau
        self.V = self.exp_ah * self.V + self.phi_1 * b_t

        # Spike threshold with noise
        random_threshold = self.V_th + np.random.normal(0, 3.0)

        if self.V >= random_threshold:
            self.spiked = True
            self.V = self.V_reset
            self.refractory_timer = self.refractory_period
            return True, 30
        else:
            self.spiked = False
            return False, self.V


def simulate_network_with_synaptic_current(
    n_neurons=3, t_max=1000, dt=1.0, noise_std=0.2, tau_syn=20.0
):
    np.random.seed(42)

    weights = np.array(
        [
            [0.0, 0.5, 0.2],
            [0.3, 0.0, 0.4],
            [0.6, 0.1, 0.0],
        ]
    )

    neurons = [
        IFNeuron(weights=None, dt=dt, gain=0.1, ou_sigma=2.0, neuron_idx=i)
        for i in range(n_neurons)
    ]

    spike_history = np.zeros((t_max, n_neurons))
    syn_current = np.zeros(n_neurons)  # synaptic current vector, per neuron

    stim_duration = 50
    ext_stim = np.zeros((t_max, n_neurons))
    ext_stim[:stim_duration] = np.random.binomial(1, 0.3, (stim_duration, n_neurons))

    exp_decay = np.exp(-dt / tau_syn)

    for t in range(t_max):
        # previous spikes
        prev_spikes = spike_history[t - 1] if t > 0 else np.zeros(n_neurons)

        # update synaptic currents with exponential filtering of spikes
        syn_current = exp_decay * syn_current + prev_spikes

        # weighted input currents for each neuron
        input_current = weights @ syn_current

        # add external stimulus current (e.g. scaled to 1.0)
        input_current += ext_stim[t] * 1.0

        # add noise
        input_current += np.random.normal(0, noise_std, n_neurons)

        for i, neuron in enumerate(neurons):
            # Instead of passing binary spike inputs, pass the continuous input current
            spike, voltage = neuron.step(input_current[i])
            spike_history[t, i] = spike

    return spike_history, ext_stim, weights


def estimate_weights_per_neuron(spike_history, input_history, alpha=1.0):
    n_timesteps, n_neurons = spike_history.shape
    estimated_weights = np.zeros((n_neurons, n_neurons))

    for neuron_idx in range(n_neurons):
        X = input_history  # Use all neuron inputs as features (shape T x N)
        y = spike_history[:, neuron_idx]  # binary spikes for neuron i (shape T,)

        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(X, y)

        estimated_weights[neuron_idx] = reg.coef_

    return estimated_weights


def simulate_emulator(
    estimated_weights, t_max=1000, dt=1.0, stim_duration=50, tau_syn=10
):
    n_neurons = estimated_weights.shape[0]
    neurons = []

    for i in range(n_neurons):
        neuron = IFNeuron(
            weights=estimated_weights[i],
            dt=dt,
            V_th=-35.0,
            R=10.0,
            ou_sigma=2,
        )
        neurons.append(neuron)

    spike_history = np.zeros((t_max, n_neurons))
    ext_stim = np.zeros((t_max, n_neurons))
    ext_stim[:stim_duration] = np.random.binomial(1, 0.05, (stim_duration, n_neurons))

    syn_currents = np.zeros((n_neurons, n_neurons))
    alpha = np.exp(-dt / tau_syn)

    for t in range(t_max):
        prev_spikes = spike_history[t - 1] if t > 0 else np.zeros(n_neurons)
        syn_currents = alpha * syn_currents + (1 - alpha) * prev_spikes[np.newaxis, :]

        # Input vector is filtered synaptic currents plus external stim
        noisy_inputs = np.clip(syn_currents + ext_stim[t][np.newaxis, :], 0, 1)

        for i, neuron in enumerate(neurons):
            input_vec = noisy_inputs[i].copy()
            spike, _ = neuron.step(input_vec)
            spike_history[t, i] = spike

    return spike_history


def compare_spike_trains(original, emulator):
    """
    Compare spike trains between the original and emulator networks.
    Returns per-neuron similarity score (fraction of timesteps with identical spikes).
    """
    n_neurons = original.shape[1]
    scores = []
    for i in range(n_neurons):
        match = np.sum(original[:, i] == emulator[:, i])
        score = match / original.shape[0]
        scores.append(score)
    return scores


def system_constrained_weight_estimation(
    spike_history, input_history, weight_bounds=(-0.5, 0.5), total_weight_limit=1.0
):
    T, N = spike_history.shape
    num_weights = N * N

    def loss(w_system):
        w_matrix = w_system.reshape(N, N)
        total_loss = 0.0
        for i in range(N):
            X = input_history  # (T, N)
            y = spike_history[:, i]  # (T,)

            y_hat_raw = X @ w_matrix[i]  # linear prediction
            y_hat = expit(y_hat_raw)  # sigmoid to get pseudo-probabilities [0,1]

            total_loss += np.mean((y_hat - y) ** 2)

        return total_loss / N

    def total_abs_constraint(w_system):
        return total_weight_limit - np.sum(np.abs(w_system))

    bounds = [weight_bounds] * num_weights
    constraints = {"type": "ineq", "fun": total_abs_constraint}
    w0 = np.zeros(num_weights)

    result = minimize(loss, w0, bounds=bounds, constraints=constraints, method="SLSQP")

    if not result.success:
        print("[Warning] Optimization did not converge:", result.message)

    return result.x.reshape(N, N)


# --- Main flow ---

t_max = 200
noise_std = 0.05
tau_syn = 10  # synaptic time constant in ms

spikes, inputs, true_weights = simulate_network_with_synaptic_current(
    t_max=t_max, noise_std=noise_std, tau_syn=tau_syn
)
print("True weights:\n", true_weights)

est_weights = estimate_weights_per_neuron(spikes, inputs, alpha=1.0)
print("Estimated weights (ridge):\n", est_weights)

emulator_spikes = simulate_emulator(est_weights, t_max=t_max, tau_syn=tau_syn)
scores = compare_spike_trains(spikes, emulator_spikes)
print("Spike train matching scores per neuron (no constraint):", scores)

constrained_weights = system_constrained_weight_estimation(
    spikes, inputs, weight_bounds=(-0.5, 0.5), total_weight_limit=1.0
)
print("Constrained estimated weights:\n", constrained_weights)

emulator_spikes_constrained = simulate_emulator(
    constrained_weights, t_max=t_max, tau_syn=tau_syn
)
scores_constrained = compare_spike_trains(spikes, emulator_spikes_constrained)
print("Spike train matching scores per neuron (with constraint):", scores_constrained)

# --- Plot spike trains for neuron 0 ---
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.title("Original spikes neuron 0")
plt.plot(spikes[:, 0], label="Original", color="black")

plt.subplot(3, 1, 2)
plt.title("Emulator spikes neuron 0 (unconstrained)")
plt.plot(emulator_spikes[:, 0], label="Emulator (Ridge)", color="orange")

plt.subplot(3, 1, 3)
plt.title("Emulator spikes neuron 0 (constrained)")
plt.plot(
    emulator_spikes_constrained[:, 0], label="Emulator (Constrained)", color="green"
)

plt.tight_layout()
plt.show()
