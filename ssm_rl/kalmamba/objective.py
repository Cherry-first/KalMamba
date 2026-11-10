"""Training objective for SSMs with smoothing inference (KalMamba, and the VRKN baseline).

The ELBO follows Becker & Neumann (2022) and VRKN's AEVIObjective: reconstruction and reward terms on
samples of the smoothed belief, and for t >= 1 the KL between the smoothed conditional
q(z_t | z_{t-1}, w_{>=t}) and the dynamics p(z_t | z_{t-1}, a_{t-1}), both at a sample of z_{t-1}.
KalMamba adds the Mahalanobis regularizer of Equation 2 that pulls m_t towards the filtered mean.
"""
import torch

from ssm_rl.model_learning.abstract_objective import AbstractModelObjective
from ssm_rl.model_learning.kl_objective import AbstractKLObjective

nn = torch.nn


def _std(var: torch.Tensor) -> torch.Tensor:
    return var.clamp_min(1e-10).sqrt()


class SmoothedKLObjective(AbstractKLObjective):
    """KL term of the smoothing ELBO, with KL balancing and free nats from CoRAL's GaussianKLLoss."""

    def __init__(self, initial_scale_factor: float, *args, **kwargs):
        super().__init__("gaussian", *args, **kwargs)
        self._initial_scale_factor = initial_scale_factor

    def forward(self,
                smoothed_or_post_states: dict[str, torch.Tensor],
                prior_states: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        s = smoothed_or_post_states
        dyn_kl, log_dict = self._kl(lhs_mean=s["cond_dyn_mean"], lhs_std=_std(s["cond_dyn_var"]),
                                    rhs_mean=s["dyn_mean"], rhs_std=_std(s["dyn_var"]))
        # The step-0 term compares the smoothed belief with p(z_0). The VRKN weights it with 0 by default.
        initial_kl = self._kl.kl(lhs_mean=s["mean"][:, :1], lhs_std=_std(s["var"][:, :1]),
                                 rhs_mean=prior_states["mean"][:, :1], rhs_std=_std(prior_states["var"][:, :1]))
        log_dict["initial_kl"] = initial_kl.mean().detach().cpu().numpy()
        T = s["mean"].shape[1]
        kl = (self._initial_scale_factor * initial_kl.mean() + (T - 1) * dyn_kl) / T
        return kl, log_dict


class SmoothingSSMObjective(AbstractModelObjective):

    def __init__(self,
                 regularization_scale: float,
                 regularization_reduction: str,
                 regularization_detach_filter: bool,
                 *args, **kwargs):
        """regularization_reduction: "sum" over latent dimensions as in Equation 2, or "mean".
        regularization_detach_filter: pull only m_t towards the filtered belief (True), or both ways."""
        super().__init__(*args, **kwargs)
        assert regularization_reduction in ["sum", "mean"]
        self._regularization_scale = regularization_scale
        self._regularization_reduction = regularization_reduction
        self._regularization_detach_filter = regularization_detach_filter

    def compute_states(self,
                       observations: list[torch.Tensor],
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       terminated: torch.Tensor,
                       truncated: torch.Tensor,
                       info: dict,
                       return_prior_states: bool = False,
                       smoothed_states_if_avail: bool = False):
        obs_valid = info.get("obs_valid", None)
        embedded_obs = self.model.encode(observations=observations, obs_valid=obs_valid)
        out = self.model.transition_model.forward_pass(embedded_obs=embedded_obs, actions=actions,
                                                       obs_valid=obs_valid, smooth=smoothed_states_if_avail)
        states = out["smoothed"] if smoothed_states_if_avail else out["post"]
        return (states, out["prior"]) if return_prior_states else states

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  rewards: torch.Tensor,
                                  terminated: torch.Tensor,
                                  truncated: torch.Tensor,
                                  info: dict,
                                  smoothed_states_if_avail: bool = False) \
            -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        assert not terminated.any(), NotImplementedError
        obs_valid = info.get("obs_valid", None)
        loss_masks = info.get("loss_mask", None)

        embedded_obs = self.model.encode(observations=observations, obs_valid=obs_valid)
        out = self.model.transition_model.forward_pass(embedded_obs=embedded_obs, actions=actions,
                                                       obs_valid=obs_valid)
        smoothed, prior, post = out["smoothed"], out["prior"], out["post"]

        obs_losses, reward_loss, kl, inv_dyn_loss, log_dict = \
            self._compute_losses(observations=observations,
                                 embedded_obs=embedded_obs,
                                 actions=actions,
                                 rewards=rewards,
                                 smoothed_or_post_features=self.model.transition_model.get_features(smoothed),
                                 prior_features=prior["mean"],
                                 smoothed_or_post_states=smoothed,
                                 prior_states=prior,
                                 obs_valid=obs_valid,
                                 loss_masks=loss_masks)
        loss = sum(obs_losses) + reward_loss + kl + inv_dyn_loss

        if "m" in out:
            # Equation 2, averaged over batch and time like the other terms.
            target_mean, target_var = post["mean"], post["var"]
            if self._regularization_detach_filter:
                target_mean, target_var = target_mean.detach(), target_var.detach()
            sq_err = (out["m"] - target_mean).square()
            mahalanobis = (sq_err / target_var).sum(-1).mean()
            log_dict["mahalanobis"] = mahalanobis.detach().cpu().numpy()
            log_dict["m_sq_err"] = sq_err.mean().detach().cpu().numpy()
            log_dict["filter_var"] = post["var"].mean().detach().cpu().numpy()
            log_dict["elbo_loss"] = loss.detach().cpu().numpy()
            if self._regularization_reduction == "mean":
                mahalanobis = mahalanobis / out["m"].shape[-1]
            if self._regularization_scale > 0:
                loss = loss + self._regularization_scale * mahalanobis

        state_return = smoothed if smoothed_states_if_avail else post
        return loss, log_dict, state_return
