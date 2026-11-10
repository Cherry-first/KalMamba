"""Train/act mismatch diagnostic for linear Gaussian SSMs (KalMamba, VRKN).

The model is trained on 32-step windows, so the filtered means SAC learns from come from windows that
start at the prior p(z_0). When acting, the model filters a whole episode (about 500 steps) in step mode.
For KalMamba the Mamba backbone also runs far past the training length there. This compares, on one
full collected episode and in MAP mode:
  - the filtered mean of the full episode (what the policy sees when acting) with the filtered mean of
    training-length windows over the same steps (what SAC was trained on), and
  - for KalMamba, m_t with the filtered mean, by position in the episode.
Errors are squared differences divided by the full-episode filtered variance, averaged over latent
dimensions, so 1.0 means one posterior standard deviation.
"""
import numpy as np
import torch

from ssm_rl.common.bayesian_modules import MAPEstimate

EPISODE_BINS = [(0, 32), (32, 128), (128, 256), (256, 10 ** 6)]
WINDOW_BINS = [(0, 8), (8, 16), (16, 32)]


def _binned(values: torch.Tensor, bins, prefix: str) -> dict:
    """values: (T,); mean over the time steps in each [lo, hi) bin."""
    out = {}
    for lo, hi in bins:
        if lo < values.shape[0]:
            out[f"{prefix}_{lo}_{min(hi, values.shape[0])}"] = values[lo:hi].mean().item()
    return out


@torch.no_grad()
def train_act_mismatch(model, observations: list[torch.Tensor], actions: torch.Tensor, window: int,
                       device: torch.device) -> dict:
    """observations: list of (T, D) tensors and actions (T, A) of one stored episode."""
    tm = model.transition_model
    obs = [o.unsqueeze(0).to(device) for o in observations]
    act = actions.unsqueeze(0).to(device)
    with MAPEstimate(model):
        emb = model.encode(observations=obs)
        full = tm.forward_pass(embedded_obs=emb, actions=act, smooth=False)
        mean, var = full["post"]["mean"][0], full["post"]["var"][0]  # (T, lod)
        log = {"mean_sq": mean.square().mean().item(), "filter_var": var.mean().item()}

        # Step mode as used when acting; should equal the full parallel pass.
        state = tm.get_initial(1)
        step_means = []
        for t in range(act.shape[1]):
            state = tm.get_next_posterior(latent_obs=[e[:, t] for e in emb], action=act[:, t], post_state=state)
            step_means.append(state["mean"])
        log["step_vs_parallel_maxabs"] = (torch.cat(step_means) - mean).abs().max().item()

        if "m" in full:
            m_err = ((full["m"][0] - mean).square() / var).mean(-1)
            log.update(_binned(m_err, EPISODE_BINS, "m_err_t"))

        # Training-length windows: compare with the full-episode belief at the same steps.
        T = act.shape[1]
        gaps, starts = [], list(range(0, T - window + 1, window))
        for s in starts:
            w = tm.forward_pass(embedded_obs=[e[:, s:s + window] for e in emb], actions=act[:, s:s + window],
                                smooth=False)
            gaps.append(((w["post"]["mean"][0] - mean[s:s + window]).square() / var[s:s + window]).mean(-1))
        gaps = torch.stack(gaps)  # (num_windows, window)
        log.update(_binned(gaps[1:].mean(0), WINDOW_BINS, "win_gap_pos"))  # skip the window at the episode start
        by_start = gaps[:, window // 2:].mean(-1)  # second half of each window
        for lo, hi in EPISODE_BINS:
            sel = [i for i, s in enumerate(starts) if lo <= s < hi]
            if sel:
                log[f"win_gap_start_{lo}_{min(hi, T)}"] = by_start[sel].mean().item()
    return {k: float(np.round(v, 5)) for k, v in log.items()}
