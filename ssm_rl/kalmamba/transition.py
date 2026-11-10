"""KalMamba transition model (Becker et al., 2024, Section 4.1 and Figure 2).

A linear Gaussian SSM in a latent space. A Mamba backbone reads [w_t, a_{t-1}] and emits m_t, from which
a small network computes the diagonal dynamics A_t, b_t, Sigma_t of p(z_{t+1} | z_t, a_t). Filtering and
smoothing use the time-parallel Kalman scans in parallel_kalman.

Time indexing follows the replay buffer: actions[:, t] is the action that led to observation t
(a_{t-1} in the paper), so the transition t -> t+1 uses actions[:, t+1].
"""
from typing import Optional

import torch

import ssm_rl.common.activation as va
import ssm_rl.common.bayesian_modules as mcd
import ssm_rl.common.modules as mod
from ssm_rl.kalmamba import parallel_kalman as pk

nn = torch.nn


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block, as in mamba_ssm's MixerModel."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        from mamba_ssm import Mamba  # needs CUDA, so tests of other parts can run without it
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))

    def init_cache(self, batch_size: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state, ssm_state = self.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype)
        return conv_state.to(device), ssm_state.to(device)

    def step(self, x: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """x: (B, 1, D). Mamba.step updates the caches in place and returns them."""
        out, conv_state, ssm_state = self.mixer.step(self.norm(x), conv_state, ssm_state)
        return x + out, conv_state, ssm_state


class LinearGaussianTM(nn.Module):
    """Parts shared by KalMamba and the VRKN: the latent observation model q(w_t | z_t), the dynamics head
    (VRKNTM._build_transition_model in pbecker93/vrkn) and the initial state. The subclasses differ in the
    vector the dynamics are linearized around: m_t from the Mamba backbone for KalMamba, the filtered mean
    for the VRKN."""

    def __init__(self,
                 obs_sizes: list[int],
                 action_dim: int,
                 lod: int,
                 mc_drop_prob: float,
                 activation: str,
                 hidden_size: int,
                 init_ev: float,
                 min_ev: float,
                 max_ev: float,
                 init_trans_var: float,
                 min_trans_var: float,
                 max_trans_var: float,
                 init_obs_var: float,
                 min_obs_var: float,
                 initial_state_var: float,
                 min_initial_state_var: float,
                 learn_initial_state_var: bool):
        super().__init__()
        assert len(obs_sizes) == 1, "Only a single observation is implemented"
        self._obs_sizes = obs_sizes
        self._action_dim = action_dim
        self._lod = lod

        self._latent_obs_net = mod.DiagonalGaussianParameterLayer(in_features=obs_sizes[0],
                                                                  distribution_dim=lod,
                                                                  init_var=init_obs_var,
                                                                  min_var=min_obs_var)

        self._dyn_pre = nn.Sequential(nn.Linear(lod + action_dim, hidden_size),
                                      va.get_activation(activation, shape=(hidden_size,)),
                                      mcd.MCDropout(p=mc_drop_prob))
        self._dyn_cell = nn.GRUCell(input_size=hidden_size, hidden_size=lod)
        self._dyn_post = nn.Sequential(mcd.MCDropout(p=mc_drop_prob),
                                       nn.Linear(lod, hidden_size),
                                       va.get_activation(activation, shape=(hidden_size,)),
                                       mcd.MCDropout(p=mc_drop_prob),
                                       nn.Linear(hidden_size, 3 * lod))
        self._tm_activation = va.ScaledShiftedSigmoidActivation(init_val=init_ev, min_val=min_ev, max_val=max_ev)
        self._tv_activation = va.ScaledShiftedSigmoidActivation(init_val=init_trans_var,
                                                                min_val=min_trans_var,
                                                                max_val=max_trans_var)

        self.register_buffer("_initial_mean", torch.zeros(1, lod))
        self._initial_var = mod.ConstantUncertaintyLayer(dim=lod, init_var=initial_state_var,
                                                         min_var=min_initial_state_var,
                                                         learn=learn_initial_state_var)

    # Interface used by the CoRAL SSM wrapper, objectives and policies

    @property
    def obs_sizes(self) -> list[int]:
        return self._obs_sizes

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def feature_size(self) -> int:
        return self._lod

    @property
    def latent_distribution(self) -> str:
        return "gaussian"

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["sample"]

    def get_deterministic_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["mean"]

    # Model parts

    def _dynamics(self, lin_point: torch.Tensor, action: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A, Sigma, b of p(z_{t+1} | z_t, a_t), linearized around lin_point; works on any leading shape."""
        lead = lin_point.shape[:-1]
        h, action = lin_point.reshape(-1, lin_point.shape[-1]), action.reshape(-1, action.shape[-1])
        raw = self._dyn_post(self._dyn_cell(self._dyn_pre(torch.cat([h, action], dim=-1)), h))
        tm_raw, tv_raw, offset = torch.chunk(raw.reshape(*lead, raw.shape[-1]), chunks=3, dim=-1)
        return self._tm_activation(tm_raw), self._tv_activation(tv_raw), offset

    def _initial(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._initial_mean.repeat(batch_size, 1), self._initial_var(batch_size)

    @staticmethod
    def _sample(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return mean + var.sqrt() * torch.randn_like(mean)

    def _smoothing_outputs(self, smoothed: dict[str, torch.Tensor],
                           tm: torch.Tensor, offset: torch.Tensor, tv: torch.Tensor) -> dict[str, torch.Tensor]:
        sample = self._sample(smoothed["mean"], smoothed["var"])
        cond = pk.smoothed_conditionals(smoothed["mean"], smoothed["var"], sample, smoothed["gain"],
                                        tm, offset, tv)
        return {"mean": smoothed["mean"], "var": smoothed["var"], "sample": sample, **cond}

    def _outputs(self, f: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        return {"post": {"mean": f["post_mean"], "var": f["post_var"],
                         "sample": self._sample(f["post_mean"], f["post_var"])},
                "prior": {"mean": f["prior_mean"], "var": f["prior_var"]}}


class KalMambaTM(LinearGaussianTM):

    def __init__(self,
                 pre_num_layers: int,
                 pre_layer_size: int,
                 num_blocks: int,
                 d_model: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 act_window: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self._act_window = act_window
        lod, action_dim = self._lod, self._action_dim
        # Backbone: [w_t, a_{t-1}] -> m_t. With num_blocks = 0 this is the paper's "No Mamba" ablation.
        pre_layers, pre_out = mcd.build_layers_with_mc_dropout(in_features=lod + action_dim,
                                                               drop_prob=kwargs["mc_drop_prob"],
                                                               layer_sizes=[pre_layer_size] * pre_num_layers,
                                                               activation=kwargs["activation"])
        self._pre_mamba = nn.Sequential(*pre_layers, nn.Linear(pre_out, d_model))
        self._blocks = nn.ModuleList(MambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_blocks))
        self._post_mamba = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, lod))

    def _backbone(self, latent_obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        x = self._pre_mamba(torch.cat([latent_obs, prev_action], dim=-1))
        for block in self._blocks:
            x = block(x)
        return self._post_mamba(x)

    # Training: time-parallel filtering and smoothing

    def forward_pass(self,
                     embedded_obs: list[torch.Tensor],
                     actions: torch.Tensor,
                     obs_valid: Optional[list[torch.Tensor]] = None,
                     smooth: bool = True) -> dict[str, dict[str, torch.Tensor]]:
        assert obs_valid is None, "Missing observations are not implemented for KalMamba"
        obs, obs_var = self._latent_obs_net(embedded_obs[0])
        m = self._backbone(obs, actions)
        tm, tv, offset = self._dynamics(m[:, :-1], actions[:, 1:])
        init_mean, init_var = self._initial(obs.shape[0])

        f = pk.parallel_filter(init_mean, init_var, tm, offset, tv, obs, obs_var)
        out = self._outputs(f) | {"m": m}
        if smooth:
            smoothed = pk.parallel_smoother(f["post_mean"], f["post_var"], f["prior_mean"], f["prior_var"], tm)
            out["smoothed"] = self._smoothing_outputs(smoothed, tm, offset, tv)
        return out

    # Acting: one filter step per environment step

    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        mean, var = self._initial(batch_size)
        state = {"mean": mean, "var": var, "sample": mean, "m": torch.zeros_like(mean),
                 "initial": torch.ones(batch_size, 1, dtype=torch.bool, device=mean.device)}
        if self._act_window > 0:
            # Encoder outputs and actions of the last act_window steps, filled in get_next_posterior.
            state["emb_hist"] = mean.new_zeros(batch_size, 0, self._obs_sizes[0])
            state["act_hist"] = mean.new_zeros(batch_size, 0, self._action_dim)
            return state
        for i, block in enumerate(self._blocks):
            state[f"conv{i}"], state[f"ssm{i}"] = block.init_cache(batch_size, mean.device, mean.dtype)
        return state

    def get_next_posterior(self,
                           latent_obs: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict[str, torch.Tensor],
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        """latent_obs: encoder output for o_t (B, E); action: a_{t-1}; post_state: belief at t-1."""
        assert obs_valid is None, "Missing observations are not implemented for KalMamba"
        if self._act_window > 0:
            return self._windowed_posterior(latent_obs[0], action, post_state)
        obs, obs_var = self._latent_obs_net(latent_obs[0])

        if "initial" in post_state:  # step 0 has no transition, as in forward_pass
            prior_mean, prior_var = post_state["mean"], post_state["var"]
        else:
            tm, tv, offset = self._dynamics(post_state["m"], action)
            prior_mean, prior_var = tm * post_state["mean"] + offset, tm.square() * post_state["var"] + tv
        mean, var = pk.update_step(prior_mean, prior_var, obs, obs_var)

        x = self._pre_mamba(torch.cat([obs, action], dim=-1)).unsqueeze(1)
        new_state = {}
        for i, block in enumerate(self._blocks):
            x, new_state[f"conv{i}"], new_state[f"ssm{i}"] = block.step(x, post_state[f"conv{i}"],
                                                                         post_state[f"ssm{i}"])
        m = self._post_mamba(x.squeeze(1))
        return {"mean": mean, "var": var, "sample": self._sample(mean, var), "m": m, **new_state}

    def _windowed_posterior(self, emb: torch.Tensor, action: torch.Tensor,
                            post_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter the last act_window steps with forward_pass, exactly as a training window would, and return
        the belief at the newest step."""
        emb_hist = torch.cat([post_state["emb_hist"], emb.unsqueeze(1)], dim=1)[:, -self._act_window:]
        act_hist = torch.cat([post_state["act_hist"], action.unsqueeze(1)], dim=1)[:, -self._act_window:]
        out = self.forward_pass(embedded_obs=[emb_hist], actions=act_hist, smooth=False)
        mean, var = out["post"]["mean"][:, -1], out["post"]["var"][:, -1]
        return {"mean": mean, "var": var, "sample": self._sample(mean, var), "m": out["m"][:, -1],
                "emb_hist": emb_hist, "act_hist": act_hist}
