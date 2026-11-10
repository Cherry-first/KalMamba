"""Variational Recurrent Kalman Network (Becker & Neumann, 2022), the VRKN+SAC baseline of the KalMamba paper.

Port of VRKNTM from pbecker93/vrkn onto the shared LinearGaussianTM. The dynamics are linearized around the
filtered mean of the previous step, so filtering is an autoregressive loop over time; the RTS backward pass
is a loop as well, as in VRKNTM.extended_backward_pass. This sequential cost is what the KalMamba paper's
runtime comparison measures.
"""
from typing import Optional

import torch

from ssm_rl.kalmamba import parallel_kalman as pk
from ssm_rl.kalmamba.transition import LinearGaussianTM


class VRKNTM(LinearGaussianTM):

    def forward_pass(self,
                     embedded_obs: list[torch.Tensor],
                     actions: torch.Tensor,
                     obs_valid: Optional[list[torch.Tensor]] = None,
                     smooth: bool = True) -> dict[str, dict[str, torch.Tensor]]:
        assert obs_valid is None, "Missing observations are not implemented"
        obs, obs_var = self._latent_obs_net(embedded_obs[0])
        T = obs.shape[1]

        prior_mean, prior_var = self._initial(obs.shape[0])
        post_m, post_v, prior_m, prior_v, tms, tvs, offsets = [], [], [], [], [], [], []
        for t in range(T):
            if t > 0:
                tm, tv, offset = self._dynamics(post_m[-1], actions[:, t])
                prior_mean, prior_var = tm * post_m[-1] + offset, tm.square() * post_v[-1] + tv
                tms.append(tm), tvs.append(tv), offsets.append(offset)
            mean, var = pk.update_step(prior_mean, prior_var, obs[:, t], obs_var[:, t])
            prior_m.append(prior_mean), prior_v.append(prior_var), post_m.append(mean), post_v.append(var)

        stack = lambda xs: torch.stack(xs, dim=1)
        f = {"post_mean": stack(post_m), "post_var": stack(post_v),
             "prior_mean": stack(prior_m), "prior_var": stack(prior_v)}
        out = self._outputs(f)
        if smooth:
            tm, tv, offset = (stack(x) if x else obs.new_zeros(obs.shape[0], 0, obs.shape[2])
                              for x in (tms, tvs, offsets))
            smoothed = self._rts_backward(f, tm)
            out["smoothed"] = self._smoothing_outputs(smoothed, tm, offset, tv)
        return out

    @staticmethod
    def _rts_backward(f: dict[str, torch.Tensor], tm: torch.Tensor) -> dict[str, torch.Tensor]:
        """VRKNTM._rts_backward applied from the last step to the first."""
        T = f["post_mean"].shape[1]
        means, variances = [f["post_mean"][:, -1]], [f["post_var"][:, -1]]
        gains = [torch.zeros_like(variances[0])]
        for t in range(T - 2, -1, -1):
            c = f["post_var"][:, t] * tm[:, t] / f["prior_var"][:, t + 1]
            means.append(f["post_mean"][:, t] + c * (means[-1] - f["prior_mean"][:, t + 1]))
            variances.append(f["post_var"][:, t] + c.square() * (variances[-1] - f["prior_var"][:, t + 1]))
            gains.append(c)
        stack = lambda xs: torch.stack(xs[::-1], dim=1)
        return {"mean": stack(means), "var": stack(variances), "gain": stack(gains)}

    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        mean, var = self._initial(batch_size)
        return {"mean": mean, "var": var, "sample": mean,
                "initial": torch.ones(batch_size, 1, dtype=torch.bool, device=mean.device)}

    def get_next_posterior(self,
                           latent_obs: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict[str, torch.Tensor],
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        assert obs_valid is None, "Missing observations are not implemented"
        obs, obs_var = self._latent_obs_net(latent_obs[0])
        if "initial" in post_state:
            prior_mean, prior_var = post_state["mean"], post_state["var"]
        else:
            tm, tv, offset = self._dynamics(post_state["mean"], action)
            prior_mean, prior_var = tm * post_state["mean"] + offset, tm.square() * post_state["var"] + tv
        mean, var = pk.update_step(prior_mean, prior_var, obs, obs_var)
        return {"mean": mean, "var": var, "sample": self._sample(mean, var)}
