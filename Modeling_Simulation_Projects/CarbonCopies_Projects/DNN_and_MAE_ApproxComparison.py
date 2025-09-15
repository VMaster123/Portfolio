import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import time


# This is a very simple example where ISI data is sampled from Gama distribution.
# It uses MLE to find optimal paramters k and theta for Gamma Distribution given data

# Step 1: Generate synthetic ISI data from Gamma distribution
N_SAMPLES = 500
N_SPIKES = 400
SHAPE_RANGE = (1.0, 5.0)  # gamma shape parameter (k)
SCALE_RANGE = (0.5, 3.0)  # gamma scale parameter (theta)


def generate_gamma_isis(shape, scale, n_spikes=N_SPIKES):
    return gamma.rvs(a=shape, scale=scale, size=n_spikes)


X = []
y = []
for _ in range(N_SAMPLES):
    shape = np.random.uniform(*SHAPE_RANGE)
    scale = np.random.uniform(*SCALE_RANGE)
    isis = generate_gamma_isis(shape, scale)
    X.append(isis)
    y.append([shape, scale])

X = np.array(X)
y = np.array(y)

# -----------------------------
# Step 2: MLE function to fit gamma parameters from ISI data
# -----------------------------


def gamma_nll(params, data):
    shape, scale = params
    if shape <= 0 or scale <= 0:
        return np.inf
    return -np.sum(gamma.logpdf(data, a=shape, scale=scale))


def fit_gamma_isi(isi_data):
    init_params = [2.0, 1.0]
    bounds = [(1e-3, None), (1e-3, None)]
    res = minimize(gamma_nll, init_params, args=(isi_data,), bounds=bounds)
    return res.x


# -----------------------------
# Step 3: Split data and normalize for DNN
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_mean = X_train.mean()
X_std = X_train.std()

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# -----------------------------
# Step 4: Train DNN to predict shape and scale from ISI sequence
# -----------------------------

dnn = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
)
dnn.fit(X_train_norm, y_train)
preds_dnn = dnn.predict(X_test_norm)

# -----------------------------
# Step 5: Method 1 - Use MLE fitting for each ISI sequence in test set
# -----------------------------

preds_mle = []
for isi in X_test:
    shape_hat, scale_hat = fit_gamma_isi(isi)
    preds_mle.append([shape_hat, scale_hat])

preds_mle = np.array(preds_mle)

# -----------------------------
# Step 6: Evaluation
# -----------------------------

mae_shape_dnn = mean_absolute_error(y_test[:, 0], preds_dnn[:, 0])
mae_scale_dnn = mean_absolute_error(y_test[:, 1], preds_dnn[:, 1])
mae_shape_mle = mean_absolute_error(y_test[:, 0], preds_mle[:, 0])
mae_scale_mle = mean_absolute_error(y_test[:, 1], preds_mle[:, 1])

print(f"MAE for shape (DNN): {mae_shape_dnn:.4f}")
print(f"MAE for scale (DNN): {mae_scale_dnn:.4f}")
print(f"MAE for shape (MLE): {mae_shape_mle:.4f}")
print(f"MAE for scale (MLE): {mae_scale_mle:.4f}")


# Measure time for DNN training + prediction
start_dnn = time.time()
dnn.fit(X_train_norm, y_train)
preds_dnn = dnn.predict(X_test_norm)
end_dnn = time.time()
print(f"DNN training + prediction time: {end_dnn - start_dnn:.4f} seconds")

# Measure time for MLE fitting on test set
start_mle = time.time()
preds_mle = []
for isi in X_test:
    shape_hat, scale_hat = fit_gamma_isi(isi)
    preds_mle.append([shape_hat, scale_hat])
preds_mle = np.array(preds_mle)
end_mle = time.time()
print(f"MLE fitting time on test set: {end_mle - start_mle:.4f} seconds")

# -----------------------------
# Step 7: Visualization
# -----------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Gamma Shape Parameter Estimates")
plt.plot(y_test[:, 0], "k--", label="True Shape")
plt.plot(preds_dnn[:, 0], "b", label="DNN Approx")
plt.plot(preds_mle[:, 0], "r", label="MLE (True Model)")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Gamma Scale Parameter Estimates")
plt.plot(y_test[:, 1], "k--", label="True Scale")
plt.plot(preds_dnn[:, 1], "b", label="DNN Approx")
plt.plot(preds_mle[:, 1], "r", label="MLE (True Model)")
plt.legend()

plt.tight_layout()
plt.show()
