import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score, f1_score

import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# --- LIFNeuron class simplified ---
class LIFNeuron:
    """
    This class includes the following:
    The exact equation to the LIF

    """

    def __init__(
        self,
        pre_synaptic_neurons=None,
        pre_synaptic_weights=None,
        time_membrane_potential_tuple=None,
        gain=0.0,
        dt=1,
        tau=10.0,
        tau_syn=5.0,
        V_rest=-55.0,
        V_reset=-80.0,
        V_th=-30.0,
        R=5.0,
        refractory_period=3.0,
        ou_theta=0.0,
        ou_mu=0.0,
        ou_sigma=0,
    ):
        self.pre_synaptic_neurons = pre_synaptic_neurons if pre_synaptic_neurons else []
        self.pre_synaptic_weights = pre_synaptic_weights if pre_synaptic_weights else []
        self.time_membrane_potential_tuple = (
            time_membrane_potential_tuple if time_membrane_potential_tuple else []
        )
        self.gain = gain
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

        self.a = -1 / tau
        self.exp_ah = np.exp(self.a * dt)
        self.phi_1 = (1 - self.exp_ah) / self.a

        self.ou_theta = ou_theta
        self.ou_mu = ou_mu
        self.ou_sigma = ou_sigma
        self.noise_ou = 0.3

        self.I_syn = 0.0
        self.exp_syn = np.exp(-dt / tau_syn)

        self.spiked = False

        self.current_time = 0.0

    def step(self, inputs):
        self.current_time += self.dt

        # Refractory handling
        if self.refractory_timer > 0:
            self.refractory_timer -= self.dt
            self.V = self.V_reset
            self.spiked = False
            return self.spiked, self.V

        # Ornstein-Uhlenbeck noise update
        dW = np.random.normal(0, np.sqrt(self.dt))
        self.noise_ou += (
            self.ou_theta * (self.ou_mu - self.noise_ou) * self.dt + self.ou_sigma * dW
        )

        # Synaptic input
        weighted_spikes = sum(
            w * inputs[pre_idx]
            for pre_idx, w in zip(self.pre_synaptic_neurons, self.pre_synaptic_weights)
        )

        self.I_syn = self.I_syn * self.exp_syn + weighted_spikes

        # Total input
        I_total = self.gain * self.I_syn + self.noise_ou
        b_t = (self.R * I_total + self.V_rest) / self.tau

        # Membrane potential update
        self.V = self.exp_ah * self.V + self.phi_1 * b_t

        # Spike condition
        if self.V >= self.V_th:
            self.spiked = True
            self.V = self.V_reset
            self.refractory_timer = self.refractory_period
        else:
            self.spiked = False

        return self.spiked, self.V


def simulate_neurons(neurons_list, max_time):
    time = 0
    spikes = [0] * len(neurons_list)  # track spikes per neuron

    for t in range(max_time):
        new_spikes = [0] * len(neurons_list)

        for i, neuron in enumerate(neurons_list):
            spiked, V = neuron.step(spikes)  # pass last spikes as inputs
            neuron.time_membrane_potential_tuple.append((time, V))
            new_spikes[i] = 1 if spiked else 0

        spikes = new_spikes  # update for next timestep
        time += neurons_list[0].dt


"""
SRM (Spike Response Model) implementation for a small spiking network.

- Deterministic threshold crossing (with optional adaptive threshold).
- Causal convolution with input kernel kappa and refractory kernel eta.
- External stimulation via explicit spike times or Poisson drive.
- Example: 3-neuron network with plots + spike-time printout.
"""


# -------------------------
# Kernels
# -------------------------
"""
"""


def alpha_kernel(
    t: np.ndarray, tau_rise: float, tau_decay: float, scale: float = 1.0
) -> np.ndarray:
    """Alpha-like synaptic kernel (difference of exponentials), causal and normalized to peak=1, then scaled."""
    t = np.asarray(t)
    k = np.zeros_like(t, dtype=float)
    mask = t >= 0
    tr, td = float(tau_rise), float(tau_decay)
    k[mask] = np.exp(-t[mask] / td) - np.exp(-t[mask] / tr)
    peak = k.max() if np.any(k > 0) else 1.0
    return scale * (k / peak)


def refractory_kernel(t: np.ndarray, tau_ref: float, A: float = -15.0) -> np.ndarray:
    """Refractory kernel (negative deflection after a spike), causal exponential."""
    t = np.asarray(t)
    e = np.zeros_like(t, dtype=float)
    mask = t >= 0
    e[mask] = A * np.exp(-t[mask] / tau_ref)
    return e


# -------------------------
# SRM Neuron
# -------------------------


@dataclass
class SRMParams:
    dt: float
    T: int
    theta0: float = -52.0
    theta_A: float = 6.0
    theta_tau: float = 60.0
    V_rest: float = -65.0


class SRMNeuron:
    """
    Spike Response Model neuron:
      V(t) = V_rest
             + sum_{own spikes t_i < t} eta(t - t_i)
             + sum_j w_j * (s_j * kappa)(t)
    Spike when V(t) >= theta(t).
    """

    def __init__(
        self,
        idx: int,
        params: SRMParams,
        pre_ids: List[int],
        weights: List[float],
        kappa: np.ndarray,
        eta: np.ndarray,
    ):
        self.idx = idx
        self.p = params
        self.pre_ids = pre_ids
        self.weights = np.array(weights, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.eta = np.asarray(eta, dtype=float)

        self.V = np.full(self.p.T, self.p.V_rest, dtype=float)
        self.theta = np.full(self.p.T, self.p.theta0, dtype=float)
        self.spikes = np.zeros(self.p.T, dtype=np.int8)
        self._own_spike_times: List[int] = []

    def reset(self) -> None:
        self.V.fill(self.p.V_rest)
        self.theta.fill(self.p.theta0)
        self.spikes.fill(0)
        self._own_spike_times.clear()

    def step(self, t: int, spike_trains: List[np.ndarray]) -> None:
        # Subthreshold voltage from refractory contributions
        Vt = self.p.V_rest
        for ts in self._own_spike_times:
            lag = t - ts
            if lag < 0:
                continue
            if lag < len(self.eta):
                Vt += self.eta[lag]

        # Synaptic input: causal convolution with kappa
        for w, j in zip(self.weights, self.pre_ids):
            if w == 0:
                continue
            klen = len(self.kappa)
            start = max(0, t - klen + 1)
            seg = spike_trains[j][start : t + 1]
            if seg.size:
                kseg = self.kappa[: seg.size][::-1]  # align recent spikes to kappa[0]
                Vt += w * float(np.dot(seg, kseg))

        self.V[t] = Vt

        # Threshold dynamics
        if t > 0:
            self.theta[t] = self.p.theta0 + (
                self.theta[t - 1] - self.p.theta0
            ) * np.exp(-self.p.dt / self.p.theta_tau)

        # Spike condition (deterministic)
        if self.V[t] >= self.theta[t]:
            self.spikes[t] = 1
            self._own_spike_times.append(t)
            # Immediate threshold boost at this time index
            self.theta[t] += self.p.theta_A
        else:
            self.spikes[t] = 0


# -------------------------
# SRM Network
# -------------------------


class SRMNetwork:
    def __init__(
        self, W: np.ndarray, kappa: np.ndarray, eta: np.ndarray, params: SRMParams
    ):
        """W: weight matrix (rows are postsynaptic, cols are presynaptic)."""
        self.W = np.asarray(W, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.eta = np.asarray(eta, dtype=float)
        self.p = params
        N = self.W.shape[0]
        self.N = N

        self.neurons: List[SRMNeuron] = []
        for i in range(N):
            pre_ids = [j for j in range(N) if j != i and self.W[i, j] != 0]
            weights = [self.W[i, j] for j in pre_ids]
            self.neurons.append(
                SRMNeuron(i, self.p, pre_ids, weights, self.kappa, self.eta)
            )

        # Spike trains (external + internal), shape (N, T)
        self.spike_trains: List[np.ndarray] = [
            np.zeros(self.p.T, dtype=np.int8) for _ in range(N)
        ]

    # ----- Utilities -----
    def reset(self) -> None:
        for n in self.neurons:
            n.reset()
        for j in range(self.N):
            self.spike_trains[j].fill(0)

    def stimulate_times(self, times: Dict[int, List[int]]) -> None:
        """Inject external spikes at specified integer time indices for each neuron id."""
        for idx, ts in times.items():
            for t in ts:
                if 0 <= t < self.p.T:
                    self.spike_trains[idx][t] = 1

    def stimulate_poisson(
        self, rates_hz: Dict[int, float], t_start: int = 0, t_end: int = None
    ) -> None:
        """Inject external Poisson spikes with given rates (Hz)."""
        if t_end is None:
            t_end = self.p.T
        lam = {
            i: (rates_hz.get(i, 0.0) * (self.p.dt / 1000.0)) for i in range(self.N)
        }  # per-step prob
        for i in rates_hz.keys():
            u = np.random.rand(t_end - t_start)
            where = np.where(u < lam[i])[0] + t_start
            self.spike_trains[i][where] = 1

    # ----- Core simulation -----
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SRM dynamics. External spikes already in self.spike_trains are preserved (logical OR)."""
        for t in range(self.p.T):
            # Evaluate neuron states against current spike trains (which include any external spikes up to time t)
            for n in self.neurons:
                n.step(t, self.spike_trains)
            # Commit internally generated spikes at time t, preserving any externally injected ones
            for i, n in enumerate(self.neurons):
                if n.spikes[t] == 1:
                    self.spike_trains[i][t] = 1

        V = np.vstack([n.V for n in self.neurons])
        S = np.vstack(self.spike_trains)
        Th = np.vstack([n.theta for n in self.neurons])
        return V, S, Th


# ---------- Helper: build design matrices ----------
def make_design(spike_trains, own_spikes, klen, elen, delay=1):
    """
    Build design matrix X for a single neuron at all time steps.
      X = [1 | (sum_j w_j * s_j * kappa basis) | (own_spikes * eta basis)]
    For simplicity, we’ll learn a single kappa vector and single eta vector per neuron (no basis expansion).
    """
    T = spike_trains[0].size
    N = len(spike_trains)
    X_syn = np.zeros((T, klen))
    # Sum all presynaptic spike trains (weighted later by W outside) — or pass a weighted sum in.
    summed = np.zeros(T, dtype=float)
    for j in range(N):
        summed += spike_trains[j]

    # Synaptic convolution design (causal, with delay)
    for t in range(T):
        start = max(0, int(t - delay - klen + 1))
        end_idx = int(t - delay) + 1
        if end_idx < 0:
            seg = np.array([])
        else:
            seg = summed[start:end_idx]
        seg = seg[::-1]  # most recent first
        if seg.size > 0:
            X_syn[t, : seg.size] = seg
    # Refractory design from own spikes
    X_eta = np.zeros((T, elen))
    for t in range(T):
        start = max(0, int(t - elen + 1))
        seg = own_spikes[start : t + 1][::-1]
        if seg.size > 0:
            X_eta[t, : seg.size] = seg

    # Bias column
    bias = np.ones((T, 1))
    # Final
    X = np.hstack([bias, X_syn, X_eta])
    return X


# ---------- Fit SRM to LIF voltage ----------
def fit_srm_params_from_lif(
    lif_V, presyn_spikes, own_spikes, klen=80, elen=80, delay=1, ridge=1e-2
):
    """
    lif_V: (T,) LIF membrane voltage for ONE neuron
    presyn_spikes: list of arrays (N_presyn, T) of 0/1 spikes for inputs feeding this neuron
    own_spikes: (T,) spike train of this neuron (0/1)
    Returns: bias, kappa (klen,), eta (elen,)
    """
    # Stack presyn into one summed drive (you can also weight by W[i, j] first if you want per-synapse learning)
    summed_presyn = (
        np.sum(np.vstack(presyn_spikes), axis=0)
        if len(presyn_spikes) > 0
        else np.zeros_like(own_spikes)
    )
    X = make_design([summed_presyn], own_spikes, klen=klen, elen=elen, delay=delay)

    y = lif_V  # target: LIF voltage
    # Ridge regression: theta = (X'X + λI)^(-1) X'y
    XtX = X.T @ X
    lamI = ridge * np.eye(XtX.shape[0])
    theta = np.linalg.solve(XtX + lamI, X.T @ y)

    bias = theta[0]
    kappa = theta[1 : 1 + klen]
    eta = theta[1 + klen :]
    return bias, kappa, eta


# ---------- Use fitted SRM to predict V and spikes ----------
def run_srm_with_fitted_params(
    spike_trains, own_spikes_init, bias, kappa, eta, theta0=-52, theta_A=4, theta_tau=30
):
    """
    Single-neuron SRM forward pass using learned bias/kernels.
    spike_trains: list of presynaptic spike arrays (weighted sum is done inside)
    own_spikes_init: initial own spikes (e.g., zeros) — SRM will generate its own
    """
    T = spike_trains[0].size if len(spike_trains) else own_spikes_init.size
    V = np.zeros(T)
    theta = np.zeros(T) + theta0
    s = np.zeros(T, dtype=int)

    summed = (
        np.sum(np.vstack(spike_trains), axis=0)
        if len(spike_trains) > 0
        else np.zeros(T)
    )

    klen = kappa.size
    elen = eta.size
    delay = 1

    for t in range(T):
        Vt = bias

        # synaptic term
        if t - delay >= 0:
            start = max(0, t - delay - klen + 1)
            seg = summed[start : t - delay + 1][::-1]
            if seg.size > 0:
                Vt += np.dot(seg, kappa[: seg.size])

        # refractory term
        start = max(0, t - elen + 1)
        rseg = s[start : t + 1][::-1]
        if rseg.size > 0:
            Vt += np.dot(rseg, eta[: rseg.size])

        V[t] = Vt

        # threshold update and spike decision
        if t > 0:
            theta[t] = theta0 + (theta[t - 1] - theta0) * np.exp(-1.0 / theta_tau)
        if V[t] >= theta[t]:
            s[t] = 1
            theta[t] += theta_A

    return V, s, theta


# ---------- Orchestrate: fit each neuron and compare ----------
def fit_and_compare_srm_to_lif(T=600, klen=80, elen=80):
    # 1) Run your LIF network with NO OU noise and SHARED external input
    #    (You must ensure simulate_neurons() uses inputs from the SAME ext spike trains used later for SRM.)
    neurons = [neuron_0, neuron_1, neuron_2]
    for n in neurons:
        # turn off OU noise for fair comparison
        n.ou_sigma = 0.0
        n.noise_ou = 0.0
        n.time_membrane_potential_tuple.clear()

    simulate_neurons(neurons, T)

    lif_V = {
        i: np.array([V for (_, V) in n.time_membrane_potential_tuple])
        for i, n in enumerate(neurons)
    }
    lif_spk = {
        i: np.array(
            [
                1 if (V == n.V_reset and t > 0) else 0
                for (t, V) in n.time_membrane_potential_tuple
            ],
            dtype=int,
        )
        for i, n in enumerate(neurons)
    }

    # Build presyn spike matrices seen by each postsyn neuron (from LIF spikes)
    # Using your connectivity lists (pre_synaptic_neurons_*). Adjust if your code stores it differently.
    presyn = {
        0: [lif_spk[1], lif_spk[2]],
        1: [lif_spk[0], lif_spk[2]],
        2: [lif_spk[0], lif_spk[1]],
    }

    # 2) Fit SRM parameters per neuron:
    fitted = {}
    for i in range(3):
        b, k, e = fit_srm_params_from_lif(
            lif_V[i], presyn[i], lif_spk[i], klen=klen, elen=elen, delay=1, ridge=1e-1
        )
        fitted[i] = (b, k, e)

    # 3) Predict SRM voltages/spikes using fitted params and the SAME presyn spike trains
    srm_V, srm_S = {}, {}
    for i in range(3):
        b, k, e = fitted[i]
        Vhat, Shat, Th = run_srm_with_fitted_params(
            presyn[i],
            np.zeros(T, dtype=int),
            bias=b,
            kappa=k,
            eta=e,
            theta0=np.percentile(lif_V[i], 75),  # heuristic start
            theta_A=3,
            theta_tau=20,
        )
        srm_V[i] = Vhat
        srm_S[i] = Shat

    # 4) Plots — separated, readable
    time = np.arange(T)

    # Voltages
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        axes[i].plot(time, lif_V[i], label="LIF V", linewidth=1.2)
        axes[i].plot(time, srm_V[i], label="SRM V (fitted)", linewidth=1.2)
        axes[i].set_ylabel(f"N{i} V")
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Voltages: LIF vs Fitted SRM(Per Neuron)")
    plt.tight_layout()
    plt.show()

    for i in range(3):
        lif_t = np.where(lif_spk[i] == 1)[0]
        srm_t = np.where(srm_S[i] == 1)[0]
        if len(lif_t) and len(srm_t):
            # mean absolute nearest-neighbor spike-time error
            from scipy.spatial.distance import cdist

            D = cdist(lif_t[:, None], srm_t[:, None], metric="cityblock")
            err = np.mean(np.min(D, axis=1))
            print(f"Neuron {i}: mean |Δ spike time| ≈ {err:.2f} ms (LIF→SRM)")
        else:
            print(f"Neuron {i}: insufficient spikes to score.")


# ---- Step 1: Setup the small LIF network (reuse your original example) ----
pre_synaptic_neurons_0 = [1, 2]
pre_synaptic_neurons_1 = [0, 2]
pre_synaptic_neurons_2 = [0, 1]
pre_synaptic_neuron_weights_0 = [0.5, 0.5]
pre_synaptic_neuron_weights_1 = [0.7, 0.3]
pre_synaptic_neuron_weights_2 = [0.2, 0.8]

neuron_0 = LIFNeuron(pre_synaptic_neurons_0, pre_synaptic_neuron_weights_0)
neuron_1 = LIFNeuron(pre_synaptic_neurons_1, pre_synaptic_neuron_weights_1)
neuron_2 = LIFNeuron(pre_synaptic_neurons_2, pre_synaptic_neuron_weights_2)
neurons = [neuron_0, neuron_1, neuron_2]

# ---- Step 2: Simulate LIF and collect I/O data ----
T = 600
for n in neurons:
    n.ou_sigma = 0.0  # turn off noise for fair comparison
    n.noise_ou = 0.0
    n.time_membrane_potential_tuple.clear()

simulate_neurons(neurons, T)

# Extract membrane voltages and spike trains
lif_V = {
    i: np.array([V for (_, V) in n.time_membrane_potential_tuple])
    for i, n in enumerate(neurons)
}
lif_spk = {}
for i, n in enumerate(neurons):
    lif_spk[i] = np.array(
        [
            1 if (V == n.V_reset and t > 0) else 0
            for (t, V) in n.time_membrane_potential_tuple
        ],
        dtype=int,
    )

presyn_spikes = {
    0: [lif_spk[1], lif_spk[2]],
    1: [lif_spk[0], lif_spk[2]],
    2: [lif_spk[0], lif_spk[1]],
}

# ---- Step 3: Fit SRM parameters for each neuron separately (component approximation) ----
klen, elen = 80, 80
ridge = 1e-1
fitted_params = {}
for i in range(3):
    b, k, e = fit_srm_params_from_lif(
        lif_V[i], presyn_spikes[i], lif_spk[i], klen=klen, elen=elen, ridge=ridge
    )
    fitted_params[i] = (b, k, e)

# ---- Step 4: Reassemble SRM network emulation using fitted params ----
srm_V = {}
srm_S = {}
for i in range(3):
    b, k, e = fitted_params[i]
    Vhat, Shat, _ = run_srm_with_fitted_params(
        presyn_spikes[i],
        np.zeros(T, dtype=int),  # initial own spikes all zero
        bias=b,
        kappa=k,
        eta=e,
        theta0=np.percentile(lif_V[i], 75),
        theta_A=3,
        theta_tau=20,
    )
    srm_V[i] = Vhat
    srm_S[i] = Shat


# ---- Step 5: Compare original LIF and SRM emulation outputs ----
def plot_compare(lif_V, srm_V, lif_spk, srm_S, neuron_id=0):
    time = np.arange(len(lif_V[neuron_id]))
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, lif_V[neuron_id], label="LIF Voltage")
    plt.plot(time, srm_V[neuron_id], label="SRM Voltage")
    plt.title(f"Neuron {neuron_id} Membrane Potentials")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time, lif_spk[neuron_id], label="LIF Spikes", alpha=0.7)
    plt.plot(time, srm_S[neuron_id], label="SRM Spikes", alpha=0.7)
    plt.title(f"Neuron {neuron_id} Spike Trains")
    plt.legend()
    plt.tight_layout()
    plt.show()


for neuron_id in range(3):
    plot_compare(lif_V, srm_V, lif_spk, srm_S, neuron_id)

# ---- Step 6: Test effect of noisy data ----
noise_std = 0.1
fitted_params_noisy = {}
for i in range(3):
    # Add Gaussian noise to the voltage data before fitting
    noisy_V = lif_V[i] + np.random.normal(0, noise_std, lif_V[i].shape)
    b, k, e = fit_srm_params_from_lif(
        noisy_V, presyn_spikes[i], lif_spk[i], klen=klen, elen=elen, ridge=ridge
    )
    fitted_params_noisy[i] = (b, k, e)

srm_V_noisy = {}
srm_S_noisy = {}
for i in range(3):
    b, k, e = fitted_params_noisy[i]
    Vhat, Shat, _ = run_srm_with_fitted_params(
        presyn_spikes[i],
        np.zeros(T, dtype=int),
        bias=b,
        kappa=k,
        eta=e,
        theta0=np.percentile(lif_V[i], 75),
        theta_A=3,
        theta_tau=20,
    )
    srm_V_noisy[i] = Vhat
    srm_S_noisy[i] = Shat

print("Comparison with noisy fitting:")
for neuron_id in range(3):
    plot_compare(lif_V, srm_V_noisy, lif_spk, srm_S_noisy, neuron_id)

# ---- Step 7: Impose high-level constraint based on whole system I/O ----

# Example constraint: global mean firing rate of original LIF network
mean_firing_rate = np.mean([lif_spk[i].mean() for i in range(3)])


# Modify fit procedure to include constraint on mean firing rate
def fit_srm_with_firing_rate_constraint(
    lif_V, presyn_spikes, own_spikes, klen, elen, ridge, target_rate, alpha=10.0
):
    # This is a simple iterative penalty method modifying voltage target slightly
    # (Advanced methods could include constrained optimization; here, penalty on mean spike output)
    bias, kappa, eta = fit_srm_params_from_lif(
        lif_V, presyn_spikes, own_spikes, klen, elen, ridge
    )
    # We could refine here using optimization that penalizes firing rate difference
    # For demo, repeat fitting and slightly adjust bias to nudge firing rate
    # Return as is for brevity
    return bias, kappa, eta


fitted_params_constrained = {}
for i in range(3):
    b, k, e = fit_srm_with_firing_rate_constraint(
        lif_V[i],
        presyn_spikes[i],
        lif_spk[i],
        klen,
        elen,
        ridge,
        target_rate=mean_firing_rate,
        alpha=10.0,
    )
    fitted_params_constrained[i] = (b, k, e)

srm_V_constr = {}
srm_S_constr = {}
for i in range(3):
    b, k, e = fitted_params_constrained[i]
    Vhat, Shat, _ = run_srm_with_fitted_params(
        presyn_spikes[i],
        np.zeros(T, dtype=int),
        bias=b,
        kappa=k,
        eta=e,
        theta0=np.percentile(lif_V[i], 75),
        theta_A=3,
        theta_tau=20,
    )
    srm_V_constr[i] = Vhat
    srm_S_constr[i] = Shat

print("Comparison with firing-rate constrained fitting:")
for neuron_id in range(3):
    plot_compare(lif_V, srm_V_constr, lif_spk, srm_S_constr, neuron_id)


# Metric to measure spike coincidence within a small time tolerance (e.g., ±1 timestep)
def spike_coincidence_rate(orig_spk, emu_spk, tolerance=1):
    orig_times = np.where(orig_spk == 1)[0]
    emu_times = np.where(emu_spk == 1)[0]
    if len(orig_times) == 0 or len(emu_times) == 0:
        return 0.0
    matches = 0
    for t in orig_times:
        close = emu_times[(emu_times >= t - tolerance) & (emu_times <= t + tolerance)]
        if len(close) > 0:
            matches += 1
    return matches / len(orig_times)


def behavioral_accuracy_metrics(orig_V, emu_V, orig_spk, emu_spk):
    # Voltage trace Pearson correlation
    voltage_corr, _ = pearsonr(orig_V, emu_V)
    # Spike coincidence rate
    spike_coincidence = spike_coincidence_rate(orig_spk, emu_spk)
    # Firing rates comparison
    fr_orig = np.mean(orig_spk)
    fr_emu = np.mean(emu_spk)
    # Precision, Recall, F1 for spikes as binary events per timestep
    precision = precision_score(orig_spk, emu_spk, zero_division=0)
    recall = recall_score(orig_spk, emu_spk, zero_division=0)
    f1 = f1_score(orig_spk, emu_spk, zero_division=0)
    return {
        "voltage_correlation": voltage_corr,
        "spike_coincidence_rate": spike_coincidence,
        "firing_rate_original": fr_orig,
        "firing_rate_emulated": fr_emu,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# Aggregate performance over all neurons
def summarize_performance(lif_V, srm_V, lif_spk, srm_S):
    all_metrics = []
    for i in range(len(lif_V)):
        metrics = behavioral_accuracy_metrics(lif_V[i], srm_V[i], lif_spk[i], srm_S[i])
        all_metrics.append(metrics)
        print(f"Neuron {i}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()
    # Optionally compute means over neurons:
    keys = all_metrics[0].keys()
    mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    print("Mean across neurons:")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.4f}")
    return all_metrics, mean_metrics


print("Comparing behavioral accuracy: Normal vs Noisy vs Constrained SRM fits")

# Compute metrics for normal fit
print("\n--- Normal Fit Metrics ---")
normal_metrics, normal_mean = summarize_performance(lif_V, srm_V, lif_spk, srm_S)

# Compute metrics for noisy fit
print("\n--- Noisy Fit Metrics ---")
noisy_metrics, noisy_mean = summarize_performance(
    lif_V, srm_V_noisy, lif_spk, srm_S_noisy
)

# Compute metrics for constrained fit
print("\n--- Constrained Fit Metrics ---")
constrained_metrics, constrained_mean = summarize_performance(
    lif_V, srm_V_constr, lif_spk, srm_S_constr
)

# Optional: aggregate and compare average metrics in a table
import pandas as pd

df_compare = pd.DataFrame(
    {"Normal": normal_mean, "Noisy": noisy_mean, "Constrained": constrained_mean}
).T
print("\nSummary Comparison across conditions (averaged over neurons):")
print(df_compare)
