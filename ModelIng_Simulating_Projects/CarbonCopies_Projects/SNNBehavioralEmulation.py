import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import random
import math

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# --- LIFNeuron class simplified ---
class IFNeuron:
    """
    This class includes the following:
    The exact equation to the LIF

    """

    def __init__(
        self,
        pre_synaptic_neurons=None,
        pre_synaptic_weights=None,
        gain=0.0,
        dt=1,
        tau=10.0,
        tau_syn=5.0,
        V_rest=-65.0,
        V_reset=-70.0,
        V_th=-30.0,
        R=5.0,
        refractory_period=3.0,
        ou_theta=0.5,
        ou_mu=0.2,
        ou_sigma=2,
    ):
        self.pre_synaptic_neurons = pre_synaptic_neurons if pre_synaptic_neurons else []
        self.pre_synaptic_weights = pre_synaptic_weights if pre_synaptic_weights else []
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


# ---------------------------
# Surrogate gradient spike fn
# ---------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # fast sigmoid derivative as surrogate: 1 / (1 + |x|)^2
        sg = 1.0 / (1.0 + x.abs()) ** 2
        return grad_output * sg


spike_fn = SpikeFn.apply


# Recurrent LIF cell
class LIFRecurrentCell(nn.Module):
    def __init__(
        self, in_features, hidden_size, tau_mem=20.0, dt=1.0, v_th=1.0, v_reset=0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.dt = dt
        self.decay = math.exp(-dt / tau_mem)
        self.v_th = v_th
        self.v_reset = v_reset

        # Input weights and recurrent weights
        self.w_in = nn.Linear(in_features, hidden_size, bias=False)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        # optional learnable thresholds
        self.learn_th = False
        if self.learn_th:
            self.v_th_param = nn.Parameter(torch.tensor(v_th))

        self._reset_state()

    def _reset_state(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ):
        self.v = torch.zeros(batch_size, self.hidden_size, device=device)
        self.s = torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x_t):
        """
        x_t: [B, in_features]
        Returns s_t (spikes) and updates internal v, s
        """
        i_t = self.w_in(x_t) + self.w_rec(self.s)  # syn current
        v_t = self.decay * self.v + i_t - self.s * self.v_th  # reset on spike
        th = self.v_th if not self.learn_th else self.v_th_param
        s_t = spike_fn(v_t - th)
        # hard reset
        self.v = torch.where(s_t > 0, torch.full_like(v_t, self.v_reset), v_t)
        self.s = s_t
        return s_t


# ---------------------------
# Spiking SNN over a window
# ---------------------------
class SpikingSNN(nn.Module):
    """
    Unrolls a recurrent LIF layer over time window T.
    Produces spike *probabilities* per time step via a readout on hidden spikes.
    """

    def __init__(
        self, input_size, hidden_size=128, tau_mem=20.0, v_th=1.0, readout_bias=True
    ):
        super().__init__()
        self.cell = LIFRecurrentCell(
            input_size, hidden_size, tau_mem=tau_mem, v_th=v_th
        )
        # readout on hidden spikes (or hidden membrane) -> prob
        self.readout = nn.Linear(hidden_size, 1, bias=readout_bias)

    def forward(self, x):
        """
        x: [B, T, input_size]
        returns probs: [B, T] in (0,1)
        """
        B, T, _ = x.shape
        device = x.device
        self.cell._reset_state(batch_size=B, device=device)

        probs = []
        for t in range(T):
            s_t = self.cell(x[:, t, :])  # [B,H] spikes
            p_t = torch.sigmoid(self.readout(s_t))  # [B,1]
            probs.append(p_t)

        probs = torch.cat(probs, dim=1)  # [B,T]
        return probs


# ========================
# Define Populations
# ========================
POP_A = [0, 1]  # Left twitch neurons
POP_B = [2]  # Right twitch neurons
TWITCH_THRESHOLD = 0.15  # Minimum firing rate to count as twitch

# ========================
# New Loss Components
# ========================


# --- temporal kernel (van Rossum–style exponential smoothing) NEW STUFF  ---
def exp_filter(signal_bt, tau=5, K=25):
    # signal_bt: [B,T] in [0,1]
    t = torch.arange(K, device=signal_bt.device, dtype=signal_bt.dtype)
    kernel = torch.exp(-t / tau)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    x = signal_bt.unsqueeze(1)  # [B,1,T]
    y = F.conv1d(x, kernel, padding=K - 1)[:, :, : signal_bt.shape[1]]
    return y.squeeze(1)  # [B,T]


def timing_loss_vr(preds_bt, target_bt, tau=7, K=31):
    fp = exp_filter(preds_bt, tau=tau, K=K)
    ft = exp_filter(target_bt, tau=tau, K=K)
    return F.mse_loss(fp, ft)


def time_corr_loss(preds_bt, target_bt, eps=1e-6):
    B, T = preds_bt.shape
    p = preds_bt - preds_bt.mean(dim=0, keepdim=True)
    t = target_bt - target_bt.mean(dim=0, keepdim=True)
    p_std = p.std(dim=0, unbiased=False) + eps
    t_std = t.std(dim=0, unbiased=False) + eps
    corr_t = (p * t).mean(dim=0) / (p_std * t_std)
    corr = corr_t.clamp(-1, 1).mean()
    return 1.0 - corr


def population_proxy_loss(preds_bt, target_bt):
    # safe per-neuron rate matching proxy (window already provided)
    return F.mse_loss(preds_bt.mean(dim=1), target_bt.mean(dim=1))


def behavioral_success(spikes_nt, popA_idx, popB_idx, window=20, thr=0.20):
    """
    spikes_nt: [N, T] {0,1}
    Return fraction of windows where exactly one population exceeds thr and 'wins'.
    """
    N, T = spikes_nt.shape
    wins, total = 0, 0
    for s in range(0, T, window):
        e = min(T, s + window)
        a_rate = spikes_nt[popA_idx, s:e].mean()
        b_rate = spikes_nt[popB_idx, s:e].mean()
        if (a_rate > thr and b_rate <= thr) or (b_rate > thr and a_rate <= thr):
            wins += 1
        total += 1
    return wins / total if total > 0 else 0.0


# END OF NEW STUFF
# ========================
# Modify train_rnn_estimators
# ========================
def train_snn_estimators(
    spike_matrix,
    neurons,
    window_size=10,
    epochs=12,
    batch_size=64,
    constrained=True,
    # loss weights
    w_bce=1.0,
    w_timing=0.9,
    w_corr=0.5,
    w_rate=0.2,
    w_pop=0.8,
    # timing kernel
    tau=7,
    K=31,
    # SNN hyperparams
    hidden_size=64,
    tau_mem=20.0,
    v_th=1.2,
):
    """
    Trains one SpikingSNN per neuron using ONLY its pre-synaptic inputs.
    Targets are spike trains over each window (shape [B, W]).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    num_neurons = len(neurons)
    timesteps = spike_matrix.shape[1]

    for neuron_idx in range(num_neurons):
        neuron = neurons[neuron_idx]
        pre_ids = neuron.pre_synaptic_neurons
        if len(pre_ids) == 0:
            models.append(None)
            continue

        # Build dataset of sliding windows
        X, y = [], []
        for t in range(window_size, timesteps):
            win_in = np.stack(
                [spike_matrix[i, t - window_size : t] for i in pre_ids], axis=-1
            )  # [W, in]
            X.append(win_in)
            y.append(spike_matrix[neuron_idx, t - window_size : t])  # [W]
        X = torch.tensor(np.array(X), dtype=torch.float32)  # [S,W,in]
        y = torch.tensor(np.array(y), dtype=torch.float32)  # [S,W]

        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = SpikingSNN(
            input_size=len(pre_ids),
            hidden_size=hidden_size,
            tau_mem=tau_mem,
            v_th=v_th,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)  # xb:[B,W,IN], yb:[B,W]
                preds = model(xb)  # [B,W] probs

                bce = F.binary_cross_entropy(preds, yb)
                if constrained:
                    t_loss = timing_loss_vr(preds, yb, tau=tau, K=K)
                    c_loss = time_corr_loss(preds, yb)
                    rate_loss = F.mse_loss(preds.mean(dim=1), yb.mean(dim=1))
                    pop_loss = population_proxy_loss(preds, yb)

                    loss = (
                        w_bce * bce
                        + w_timing * t_loss
                        + w_corr * c_loss
                        + w_rate * rate_loss
                        + w_pop * pop_loss
                    )
                else:
                    loss = w_bce * bce

                opt.zero_grad()
                loss.backward()
                opt.step()

        # quick sanity print
        model.eval()
        with torch.no_grad():
            preds = model(X.to(device)).cpu().numpy()
            bin_preds = (preds > 0.5).astype(np.int32)
            print(
                f"Accuracy for neuron {neuron_idx} ({'constrained' if constrained else 'unconstrained'}):",
                accuracy_score(y.numpy().ravel(), bin_preds.ravel()),
            )

        models.append(model)

    return models


# --- Simulate network ---
neurons = [
    IFNeuron(pre_synaptic_neurons=[1, 2], pre_synaptic_weights=[1.0, 1.0]),
    IFNeuron(pre_synaptic_neurons=[0, 2], pre_synaptic_weights=[0.7, 0.4]),
    IFNeuron(pre_synaptic_neurons=[0, 1], pre_synaptic_weights=[0.5, 0.5]),
]

timesteps = 2000
voltages = np.zeros((len(neurons), timesteps))
external_drive = np.zeros((timesteps, len(neurons)))
external_drive[20:25, 0] = 1.0  # Pulse input neuron 0

inputs = np.zeros(len(neurons))
spike_matrix = np.zeros((len(neurons), timesteps))

for t in range(timesteps):
    new_spikes = []
    for i, neuron in enumerate(neurons):
        spiked, voltage = neuron.step(inputs)
        voltages[i, t] = voltage
        spike_matrix[i, t] = 1 if spiked else 0
        new_spikes.append(1 if spiked else 0)
    inputs = np.array(new_spikes) + external_drive[t]

# --- 1) Estimate weights per neuron via logistic regression ---


class SNNEmulatedNetwork:
    def __init__(self, models, neurons, window_size=10):
        self.models = models
        self.neurons = neurons
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, seed_spikes, timesteps):
        """
        seed_spikes: [N,T] with first window prefilled.
        Returns emulated binary spikes [N,T].
        """
        N = len(self.neurons)
        emu = np.zeros((N, timesteps), dtype=np.float32)
        emu[:, : self.window_size] = seed_spikes[:, : self.window_size]

        for t in range(self.window_size, timesteps):
            for n, model in enumerate(self.models):
                if model is None:
                    continue
                pre = self.neurons[n].pre_synaptic_neurons
                xw = np.stack(
                    [emu[i, t - self.window_size : t] for i in pre], axis=-1
                )  # [W,in]
                xb = torch.tensor(
                    xw, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # [1,W,in]
                model.eval()
                with torch.no_grad():
                    probs = model(xb)  # [1,W]
                    p = probs[0, -1].item()  # last step prob
                emu[n, t] = 1.0 if np.random.rand() < p else 0.0

        return emu


window_size = 5
noise_std = 0.2  # tweak noise level here


# Clean SNN (no constraints if you want a baseline)
snn_models_clean = train_snn_estimators(
    spike_matrix, neurons, window_size=20, epochs=12, batch_size=64, constrained=False
)

# Noisy SNN baseline (optional): add jitter to inputs offline if you want
snn_models_noisy = train_snn_estimators(
    spike_matrix, neurons, window_size=20, epochs=12, batch_size=64, constrained=False
)

# Constrained SNN (timing + behavior oriented)
snn_models_constrained = train_snn_estimators(
    spike_matrix,
    neurons,
    window_size=20,
    epochs=16,
    batch_size=64,
    constrained=True,
)


# Emulate network using RNNs
emu_clean = SNNEmulatedNetwork(snn_models_clean, neurons, window_size=40).run(
    spike_matrix, timesteps
)
emu_noisy = SNNEmulatedNetwork(snn_models_noisy, neurons, window_size=40).run(
    spike_matrix, timesteps
)
emu_constrained = SNNEmulatedNetwork(
    snn_models_constrained, neurons, window_size=40
).run(spike_matrix, timesteps)


# ============================================================
# EVALUATION: Compare models at multiple levels
# ============================================================


# ---------------------- helpers ----------------------
def smooth_rate(spikes, win=20):
    """spikes: [N,T] -> rates: [N,T] using causal moving average"""
    N, T = spikes.shape
    k = np.ones(win, dtype=np.float32) / float(win)
    rates = np.stack([np.convolve(spikes[i], k, mode="same") for i in range(N)], axis=0)
    return rates


def lagged_xcorr(x, y, max_lag=50):
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for L in lags:
        if L < 0:
            corr = np.mean(x[-L:] * y[: len(y) + L])
        elif L > 0:
            corr = np.mean(x[: len(x) - L] * y[L:])
        else:
            corr = np.mean(x * y)
        corrs.append(corr)
    corrs = np.array(corrs)
    idx = np.argmax(corrs)
    return float(corrs[idx]), int(lags[idx]), lags, corrs


def population_rates(spikes, idxs, win=20):
    """Mean rate over a population."""
    rates = smooth_rate(spikes[idxs], win=win)  # [|idxs|,T]
    return rates.mean(axis=0)  # [T]


def population_margin(spikes, popA, popB, win=20):
    """A-B margin over time (positive => A dominates)."""
    rA = population_rates(spikes, popA, win)
    rB = population_rates(spikes, popB, win)
    return rA - rB, rA, rB


def behavioral_success_timeline(spikes, popA, popB, win=20, thr=0.20):
    """Returns boolean arrays for A-win, B-win, and success mask."""
    rA = population_rates(spikes, popA, win)
    rB = population_rates(spikes, popB, win)
    A_win = (rA >= thr) & (rB < thr)
    B_win = (rB >= thr) & (rA < thr)
    success = A_win | B_win
    return success, A_win, B_win, rA, rB


def raster(ax, spikes, t_max=None, title=None):
    """Quick raster plot: black dots where spikes occur."""
    N, T = spikes.shape
    t_max = T if t_max is None else min(t_max, T)
    ys, xs = np.where(spikes[:, :t_max] > 0)
    ax.scatter(xs, ys, s=2)
    ax.set_ylim(-1, N)
    ax.set_xlim(0, t_max)
    ax.set_ylabel("Neuron")
    if title:
        ax.set_title(title)


# ---------------------- main diagnostic block ----------------------
def diagnostic_evaluation(
    name,
    spikes_true,
    spikes_pred,
    popA,
    popB,
    rate_win=20,
    xcorr_lag=50,
    thr=0.20,
    t_view=400,
):
    """
    Prints summary + plots:
      1) Per-neuron best lag & peak corr (distribution)
      2) Population margin time-series & histogram
      3) Behavior success timeline
      4) Rates & rasters (first few neurons)
    """
    print(f"\n========== {name.upper()} ==========")
    N, T = spikes_true.shape

    # --- 1) Lagged xcorr per neuron on rates ---
    rates_true = smooth_rate(spikes_true, win=rate_win)
    rates_pred = smooth_rate(spikes_pred, win=rate_win)
    peak_corrs, best_lags = [], []
    for i in range(N):
        c, L, _, _ = lagged_xcorr(rates_pred[i], rates_true[i], max_lag=xcorr_lag)
        peak_corrs.append(c)
        best_lags.append(L)
    peak_corrs = np.array(peak_corrs)
    best_lags = np.array(best_lags)
    print(
        f"Lagged xcorr: mean peak corr = {peak_corrs.mean():.3f}  (median={np.median(peak_corrs):.3f})"
    )
    print(
        f"Best lags (samples): mean = {best_lags.mean():.1f}, median = {np.median(best_lags):.0f}, "
        f"fraction |lag|<=5 = {(np.abs(best_lags)<=5).mean():.2f}"
    )

    # --- 2) Population margin ---
    margin_true, rA_true, rB_true = population_margin(
        spikes_true, popA, popB, win=rate_win
    )
    margin_pred, rA_pred, rB_pred = population_margin(
        spikes_pred, popA, popB, win=rate_win
    )

    # --- 3) Behavioral success ---
    succ_pred, A_win_pred, B_win_pred, rA_p, rB_p = behavioral_success_timeline(
        spikes_pred, popA, popB, win=rate_win, thr=thr
    )
    succ_rate = succ_pred.mean()
    print(f"Behavioral Success Rate: {succ_rate*100:.2f}%  (thr={thr}, win={rate_win})")

    # --------------- Plots ---------------
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    # (A) Lagged xcorr summaries
    axA = fig.add_subplot(gs[0, 0])
    axA.hist(peak_corrs, bins=20)
    axA.set_title("Per-neuron peak lagged xcorr (rates)")
    axA.set_xlabel("corr")
    axA.set_ylabel("#neurons")

    axB = fig.add_subplot(gs[0, 1])
    axB.hist(best_lags, bins=range(-xcorr_lag, xcorr_lag + 2, 2))
    axB.set_title("Best lags (samples)")
    axB.set_xlabel("lag")
    axB.set_ylabel("#neurons")

    # (B) Population margin time series
    axC = fig.add_subplot(gs[1, 0])
    t = np.arange(T)
    axC.plot(t[:t_view], margin_true[:t_view], label="True A−B")
    axC.plot(t[:t_view], margin_pred[:t_view], label="Pred A−B")
    axC.axhline(0, linestyle="--")
    axC.set_title("Population margin (A−B) over time")
    axC.set_xlabel("time")
    axC.set_ylabel("margin")
    axC.legend(loc="upper right")

    # Margin histogram (separability)
    axD = fig.add_subplot(gs[1, 1])
    axD.hist(margin_true, bins=30, alpha=0.6, label="True")
    axD.hist(margin_pred, bins=30, alpha=0.6, label="Pred")
    axD.set_title("Margin histogram (A−B)")
    axD.set_xlabel("margin")
    axD.set_ylabel("count")
    axD.legend()

    # (C) Behavior success & rates overlay
    axE = fig.add_subplot(gs[2, 0])
    axE.plot(t[:t_view], rA_true[:t_view], label="True A rate")
    axE.plot(t[:t_view], rB_true[:t_view], label="True B rate")
    axE.plot(t[:t_view], rA_pred[:t_view], label="Pred A rate", linestyle="--")
    axE.plot(t[:t_view], rB_pred[:t_view], label="Pred B rate", linestyle="--")
    axE.axhline(thr, linestyle=":", label="thr")
    # success shading
    ylo, yhi = axE.get_ylim()
    succ_mask = succ_pred[:t_view].astype(float)
    succ_mask = (succ_mask - succ_mask.min()) / (
        succ_mask.max() - succ_mask.min() + 1e-8
    )
    axE.fill_between(
        t[:t_view],
        ylo,
        ylo + (yhi - ylo) * succ_mask * 0.15,
        alpha=0.2,
        step="pre",
        label="success mask",
    )
    axE.set_title("Population rates + success mask")
    axE.set_xlabel("time")
    axE.set_ylabel("rate")
    axE.legend(loc="upper right")

    # Raster for quick visual (pred only, first 30 neurons)
    axF = fig.add_subplot(gs[2, 1])
    raster(
        axF,
        spikes_pred[: min(30, spikes_pred.shape[0])],
        t_view,
        title="Predicted spikes (raster)",
    )
    plt.tight_layout()
    plt.show()

    return {
        "peak_corrs": peak_corrs,
        "best_lags": best_lags,
        "succ_rate": succ_rate,
        "margin_true": margin_true,
        "margin_pred": margin_pred,
    }


# ---------------------- one-call wrapper ----------------------
def evaluate_all_models(
    spike_matrix,
    emu_clean,
    emu_noisy,
    emu_constrained,
    popA=None,
    popB=None,
    rate_win=20,
    xcorr_lag=50,
    thr=0.20,
    t_view=400,
):
    N = spike_matrix.shape[0]
    if popA is None or popB is None:
        # default split in half
        half = N // 2
        popA = list(range(half))
        popB = list(range(half, N))

    print(
        f"PopA size: {len(popA)} | PopB size: {len(popB)} | thr={thr}, rate_win={rate_win}"
    )
    out_clean = diagnostic_evaluation(
        "CLEAN", spike_matrix, emu_clean, popA, popB, rate_win, xcorr_lag, thr, t_view
    )
    out_noisy = diagnostic_evaluation(
        "NOISY", spike_matrix, emu_noisy, popA, popB, rate_win, xcorr_lag, thr, t_view
    )
    out_cons = diagnostic_evaluation(
        "CONSTRAINED",
        spike_matrix,
        emu_constrained,
        popA,
        popB,
        rate_win,
        xcorr_lag,
        thr,
        t_view,
    )
    return {"clean": out_clean, "noisy": out_noisy, "constrained": out_cons}


# Define populations (keep your two-pop default or specify explicitly)
half = spike_matrix.shape[0] // 2
popA = list(range(half))
popB = list(range(half, spike_matrix.shape[0]))

# Run the rich diagnostics
diag = evaluate_all_models(
    spike_matrix,
    emu_clean,
    emu_noisy,
    emu_constrained,
    popA=popA,
    popB=popB,
    rate_win=20,  # smoothing window used throughout
    xcorr_lag=50,  # search ±50 samples for best lag
    thr=0.20,  # twitch threshold (tune if needed)
    t_view=500,  # number of samples to show in time-domain plots
)
