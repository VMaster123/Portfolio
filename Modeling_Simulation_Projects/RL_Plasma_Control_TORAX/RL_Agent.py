import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gymtorax
import gymnasium as gym
import gymtorax.action_handler as ah
import gymtorax.observation_handler as oh
from gymtorax.envs.base_env import BaseEnv
from gymtorax import rewards as reward
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

# Extract states that are meant to be observable to the RL agent
state_space_csv = "Gym_TORAX_IterHybrid-v0_env - State Space.csv"
df = pd.read_csv(state_space_csv)
observable_df = df[df["OBSERVABLE?"] == "Yes"][["NAME", "TYPE"]].dropna()

# Profile (vector) variables
OBS_PROFILES = (
    observable_df[observable_df["TYPE"] == "vector"]["NAME"]
    .dropna()
    .tolist()
)

# Scalar variables
OBS_SCALARS = (
    observable_df[observable_df["TYPE"] != "vector"]["NAME"]
    .dropna()
    .tolist()
)

print("OBS_PROFILES =", OBS_PROFILES)
print("OBS_SCALARS =", OBS_SCALARS)

"""Config for ITER hybrid scenario based parameters with nonlinear solver.

ITER hybrid scenario based (roughly) on van Mulders Nucl. Fusion 2021.
With Newton-Raphson solver and adaptive timestep (backtracking)
"""

_NBI_W_TO_MA = 1 / 16e6  # rough estimate of NBI heating power to current drive
W_to_Ne_ratio = 0

# No NBI during rampup. Rampup all NBI power between 99-100 seconds
nbi_times = np.array([0, 99, 100])
nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA

# Gaussian prescription of "NBI" deposition profiles and fractional deposition
r_nbi = 0.25
w_nbi = 0.25
el_heat_fraction = 0.66

# No ECCD power for this config (but kept here for future flexibility)
eccd_power = {0: 0, 99: 0, 100: 20.0e6}


CONFIG = {
    "plasma_composition": {
        "main_ion": {"D": 0.5, "T": 0.5},  # (bundled isotope average)
        "impurity": {"Ne": 1 - W_to_Ne_ratio, "W": W_to_Ne_ratio},
        "Z_eff": {0.0: {0.0: 2.0, 1.0: 2.0}},  # sets impurity densities
    },
    "profile_conditions": {
        "Ip": {0: 3e6, 100: 12.5e6},  # total plasma current in MA
        "T_i": {0.0: {0.0: 6.0, 1.0: 0.2}},  # T_i initial condition
        "T_i_right_bc": 0.2,  # T_i boundary condition
        "T_e": {0.0: {0.0: 6.0, 1.0: 0.2}},  # T_e initial condition
        "T_e_right_bc": 0.2,  # T_e boundary condition
        "n_e_right_bc_is_fGW": True,
        "n_e_right_bc": {0: 0.35, 100: 0.35},  # n_e boundary condition
        # set initial condition density according to Greenwald fraction.
        "nbar": 0.85,  # line average density for initial condition
        "n_e": {0: {0.0: 1.3, 1.0: 1.0}},  # Initial electron density profile
        "normalize_n_e_to_nbar": True,  # normalize initial n_e to nbar
        "n_e_nbar_is_fGW": True,  # nbar is in units for greenwald fraction
        "initial_psi_from_j": True,  # initial psi from current formula
        "initial_j_is_total_current": True,  # only ohmic current on init
        "current_profile_nu": 2,  # exponent in initial current formula
    },
    "numerics": {
        "t_final": 150,  # length of simulation time in seconds
        "fixed_dt": 1,  # fixed timestep
        "evolve_ion_heat": True,  # solve ion heat equation
        "evolve_electron_heat": True,  # solve electron heat equation
        "evolve_current": True,  # solve current equation
        "evolve_density": True,  # solve density equation
    },
    "geometry": {
        "geometry_type": "chease",
        "geometry_file": "ITER_hybrid_citrin_equil_cheasedata.mat2cols",
        "Ip_from_parameters": True,
        "R_major": 6.2,  # major radius (R) in meters
        "a_minor": 2.0,  # minor radius (a) in meters
        "B_0": 5.3,  # Toroidal magnetic field on axis [T]
    },
    "sources": {
        # Current sources (for psi equation)
        "ecrh": {  # ECRH/ECCD (with Lin-Liu)
            "gaussian_width": 0.05,
            "gaussian_location": 0.35,
            "P_total": eccd_power,
        },
        "generic_heat": {  # Proxy for NBI heat source
            "gaussian_location": r_nbi,  # Gaussian location in normalized coordinates
            "gaussian_width": w_nbi,  # Gaussian width in normalized coordinates
            "P_total": (nbi_times, nbi_powers),  # Total heating power
            # electron heating fraction r
            "electron_heat_fraction": el_heat_fraction,
        },
        "generic_current": {  # Proxy for NBI current source
            "use_absolute_current": True,  # I_generic is total external current
            "gaussian_width": w_nbi,
            "gaussian_location": r_nbi,
            "I_generic": (nbi_times, nbi_cd),
        },
        "fusion": {},  # fusion power
        "ei_exchange": {},  # equipartition
        "ohmic": {},  # ohmic power
        "cyclotron_radiation": {},  # cyclotron radiation
        "impurity_radiation": {  # impurity radiation + bremsstrahlung
            "model_name": "mavrin_fit",
            "radiation_multiplier": 0.0,
        },
    },
    "neoclassical": {
        "bootstrap_current": {
            "bootstrap_multiplier": 1.0,
        },
    },
    "pedestal": {
        "model_name": "set_T_ped_n_ped",
        # use internal boundary condition model (for H-mode and L-mode)
        "set_pedestal": True,
        "T_i_ped": {0: 0.5, 100: 0.5, 105: 3.0},
        "T_e_ped": {0: 0.5, 100: 0.5, 105: 3.0},
        "n_e_ped_is_fGW": True,
        "n_e_ped": 0.85,  # pedestal top n_e in units of fGW
        "rho_norm_ped_top": 0.95,  # set ped top location in normalized radius
    },
    "transport": {
        "model_name": "qlknn",  # Using QLKNN_7_11 default
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        "apply_inner_patch": True,
        "D_e_inner": 0.15,
        "V_e_inner": 0.0,
        "chi_i_inner": 0.3,
        "chi_e_inner": 0.3,
        "rho_inner": 0.1,  # radius below which patch transport is applied
        # set outer core transport coefficients (L-mode near edge region)
        "apply_outer_patch": True,
        "D_e_outer": 0.1,
        "V_e_outer": 0.0,
        "chi_i_outer": 2.0,
        "chi_e_outer": 2.0,
        "rho_outer": 0.95,  # radius above which patch transport is applied
        # allowed chi and diffusivity bounds
        "chi_min": 0.05,  # minimum chi
        "chi_max": 100,  # maximum chi (can be helpful for stability)
        "D_e_min": 0.05,  # minimum electron diffusivity
        "D_e_max": 50,  # maximum electron diffusivity
        "V_e_min": -10,  # minimum electron convection
        "V_e_max": 10,  # minimum electron convection
        "smoothing_width": 0.1,
        "DV_effective": True,
        "include_ITG": True,  # to toggle ITG modes on or off
        "include_TEM": True,  # to toggle TEM modes on or off
        "include_ETG": True,  # to toggle ETG modes on or off
        "avoid_big_negative_s": False,
    },
    "solver": {
        "solver_type": "linear",  # linear solver with picard iteration
        "use_predictor_corrector": True,  # for linear solver
        "n_corrector_steps": 10,  # for linear solver
        "chi_pereverzev": 30,
        "D_pereverzev": 15,
        "use_pereverzev": True,
        #        'log_iterations': False,
    },
    "time_step_calculator": {
        "calculator_type": "fixed",
    },
}


class ReducedObservation(oh.Observation):
    def __init__(self, profiles=None, scalars=None, custom_bounds_file=None):
        variables = {
            "profiles": profiles if profiles is not None else [],
            "scalars": scalars if scalars is not None else [],
        }

        super().__init__(
            variables=variables,
            custom_bounds_filename=custom_bounds_file
        )


class IterHybridEnvPartialObservability(BaseEnv):

    def __init__(self, render_mode=None, **kwargs):

        # Set environment-specific defaults
        kwargs.setdefault("log_level", "warning")
        kwargs.setdefault("plot_config", "default")

        super().__init__(render_mode=render_mode, **kwargs)

    def _define_action_space(self):
        actions = [
            ah.IpAction(
                max=[15e6],  # 15 MA max plasma current
                ramp_rate=[0.2e6],
            ),  # 0.2 MA/s ramp rate limit
            ah.NbiAction(
                max=[33e6, 1.0, 1.0],  # 33 MW max NBI power
            ),
            ah.EcrhAction(
                max=[20e6, 1.0, 1.0],  # 20 MW max ECRH power
            ),
        ]

        return actions

    def _define_observation_space(self):
        # return AllObservation(custom_bounds_file="gymtorax/envs/iter_hybrid.json
        return ReducedObservation(
            profiles=OBS_PROFILES,
            scalars=OBS_SCALARS,
            custom_bounds_file=None
        )

    def _get_torax_config(self):
        return {
            "config": CONFIG,
            "discretization": "fixed",
            "ratio_a_sim": 1,
        }

    def _compute_reward(self, state, next_state, action):
        weight_list = [1, 1, 1, 1, 1]

        # Rewards are only provided for fusion gain when the plasma is in H-mode
        def _is_H_mode():
            if (
                next_state["profiles"]["T_e"][0] > 10
                and next_state["profiles"]["T_i"][0] > 10
            ):
                return True
            else:
                return False

        # Fusion gain Q reward: r_fusion = Q / 10
        def _r_fusion_gain():
            fusion_gain = (
                reward.get_fusion_gain(next_state) / 10
            )  # Normalize with ITER target
            if _is_H_mode():
                return fusion_gain
            else:
                return 0

        # Confinement quality H98 reward: reward H98 value up to 1 (higher H98 value = better confinement)
        def _r_h98():
            h98 = reward.get_h98(next_state)
            if _is_H_mode():
                if h98 <= 1:
                    return h98
                else:
                    return 1
            else:
                return 0

        # Minimum safety factor reward: reward q_min value up to 1 (q_min < 1 leads to disruptions)
        def _r_q_min():
            q_min = reward.get_q_min(next_state)
            if q_min <= 1:
                return q_min
            elif q_min > 1:
                return 1

        # Edge safety factor reward: reward q_95 value / 3 value up to 1
        def _r_q_95():
            q_95 = reward.get_q95(next_state)
            if q_95 / 3 <= 1:
                return q_95 / 3
            else:
                return 1

        # Greenwald fraction reward: reward fgw value < 0.9 with 1, 0.9 - 1 with a small reward, > 1 with 0
        def _r_greenwald():
            fgw = float(next_state["scalars"]["fgw_n_e_line_avg"][0])  # Need to custom define this reward
        
            if fgw <= 0.9:
                return 1
            elif fgw <= 1:
                return 1 - (fgw - 0.9) / 0.1
            else:
                return 0

        # Calculate individual reward components
        r_fusion_gain = weight_list[0] * _r_fusion_gain() / 50
        r_h98 = weight_list[1] * _r_h98() / 50
        r_q_min = weight_list[2] * _r_q_min() / 150
        r_q_95 = weight_list[3] * _r_q_95() / 150
        r_greenwald = weight_list[4] * _r_greenwald() / 150

        total_reward = r_fusion_gain + r_h98 + r_q_min + r_q_95 + r_greenwald

        return total_reward


class FlattenToraxObservation(gym.ObservationWrapper):
    """
    Convert TORAX observation dict:
    
        {
            "profiles": {var: np.array(shape=(n_points,))},
            "scalars":  {var: np.array([value])}
        }

    into one flat 1D vector Box for use with Stable-Baselines3.
    """

    def __init__(self, env):
        super().__init__(env)

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict), \
            "Expected Dict observation space from TORAX."

        # Store key order so flattening is consistent across time steps
        self.profile_keys = list(obs_space["profiles"].spaces.keys())
        self.scalar_keys = list(obs_space["scalars"].spaces.keys())

        # Compute total dimensionality
        dim = 0

        # Profiles (vectors)
        for key in self.profile_keys:
            box = obs_space["profiles"].spaces[key]
            dim += int(np.prod(box.shape))

        # Scalars (1D size=1)
        for key in self.scalar_keys:
            box = obs_space["scalars"].spaces[key]
            dim += int(np.prod(box.shape))

        # Define new flattened observation space
        low = -np.inf * np.ones(dim, dtype=np.float32)
        high = np.inf * np.ones(dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        """Flatten TORAX nested dict observation into a 1D vector."""
        parts = []

        # Profiles
        for key in self.profile_keys:
            parts.append(np.asarray(obs["profiles"][key]).ravel())

        # Scalars
        for key in self.scalar_keys:
            parts.append(np.asarray(obs["scalars"][key]).ravel())

        flat = np.concatenate(parts).astype(np.float32)
        return flat

class FlattenToraxAction(gym.ActionWrapper):
    """
    Convert TORAX action Dict {
        'Ip': Box(1,)
        'NBI': Box(3,)
        'ECRH': Box(3,)
    }
    into a single 1D Box for SB3, and unflatten on env.step().
    """

    def __init__(self, env):
        super().__init__(env)

        act_space = env.action_space
        assert isinstance(act_space, spaces.Dict), \
            "Expected Dict action space from TORAX"

        self.keys = list(act_space.spaces.keys())

        # Compute flattened dimension
        dims = []
        for k in self.keys:
            dims.append(int(np.prod(act_space[k].shape)))

        self.key_dims = dims
        total_dim = sum(dims)

        # Create the new flattened Box action space
        lows = []
        highs = []
        for k in self.keys:
            box = act_space[k]
            lows.append(box.low.flatten())
            highs.append(box.high.flatten())

        lows = np.concatenate(lows).astype(np.float32)
        highs = np.concatenate(highs).astype(np.float32)

        self.action_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

    def action(self, flat_action):
        """
        Convert flat action vector back into TORAX dict for env.step().
        """
        out = {}
        idx = 0
        for k, dim in zip(self.keys, self.key_dims):
            out[k] = flat_action[idx: idx + dim]
            idx += dim
        return out


class ActionRescaleWrapper(gym.ActionWrapper):
    """
    Maps actions from [-1, 1] to the TORAX physical action ranges.
    """

    def __init__(self, env):
        super().__init__(env)

        self.low = env.action_space.low
        self.high = env.action_space.high

        # New action space presented to the agent
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self.low.shape, dtype=np.float32
        )

    def action(self, normalized_action):
        # Convert from [-1,1] to [low, high]
        return self.low + (normalized_action + 1.0) * 0.5 * (self.high - self.low)