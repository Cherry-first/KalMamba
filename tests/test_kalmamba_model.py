"""KalMamba and VRKN model checks. The Mamba kernels need a GPU, so these run in the container on a GPU node."""
import os
import sys

import pytest
import torch

from ssm_rl.common.bayesian_modules import MAPEstimate
from ssm_rl.kalmamba.kalmamba_factory import (KalMambaTransitionFactory, LinearGaussianSSMFactory,
                                              SmoothingSSMObjectiveFactory, VRKNTransitionFactory)
from ssm_rl.model_learning.objective_factories import ReconstructionObjectiveFactory
from ssm_rl.ssm_interface.encoder_factory import EncoderFactory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba kernels need CUDA")

OBS_DIM, ACT_DIM, B, T = 24, 6, 4, 32
# Checkout of pbecker93/vrkn, mounted into the container to compare the VRKN port with the original.
VRKN_REF = os.environ.get("VRKN_REF", "/opt/vrkn_ref")


def _build(num_blocks=2, transition_factory=KalMambaTransitionFactory, act_window=0):
    model_factory = LinearGaussianSSMFactory(encoder_factories=[EncoderFactory()],
                                             transition_factory=transition_factory)
    config = model_factory.get_default_config(finalize_adding=False)
    if transition_factory is KalMambaTransitionFactory:
        config.transition.num_blocks = num_blocks
        config.transition.act_window = act_window
    model = model_factory.build(config=config, input_sizes=[OBS_DIM], input_types=["vector"],
                                bypass_mask=[False], action_dim=ACT_DIM, with_obs_valid=False).cuda()
    obj_factory = SmoothingSSMObjectiveFactory(observation_loss_factories=[ReconstructionObjectiveFactory()])
    objective = obj_factory.build(model=model, target_sizes=[OBS_DIM], target_types=["vector"],
                                  bypass_mask=[False], config=obj_factory.get_default_config(), device="cuda")
    return model, objective


def _batch():
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(B, T, OBS_DIM, generator=g).cuda()
    actions = torch.rand(B, T, ACT_DIM, generator=g).cuda() * 2 - 1
    rewards = torch.rand(B, T, 1, generator=g).cuda()
    flags = torch.zeros(B, T, 1, dtype=torch.bool).cuda()
    return [obs], actions, rewards, flags, flags, {}


@pytest.mark.parametrize("num_blocks", [2, 0])
def test_loss_is_finite_and_trains_all_parts(num_blocks):
    model, objective = _build(num_blocks)
    loss, log_dict, states = objective.compute_losses_and_states(*_batch())
    assert torch.isfinite(loss)
    assert states["mean"].shape == (B, T, 230)
    assert {"kl", "initial_kl", "mahalanobis"} <= set(log_dict)
    loss.backward()
    no_grad = [n for n, p in objective.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == [], no_grad


def test_smoothed_beliefs_are_valid():
    model, _ = _build()
    with torch.no_grad():
        emb = model.encode([_batch()[0][0]])
        out = model.transition_model.forward_pass(emb, _batch()[1])
    s = out["smoothed"]
    assert (s["var"] > 0).all() and (s["cond_dyn_var"] > 0).all()
    # Smoothing conditions on more data, so it never increases the variance.
    assert (s["var"] <= out["post"]["var"] * (1 + 1e-4)).all()
    # The last smoothed belief is the last filtered belief.
    torch.testing.assert_close(s["mean"][:, -1], out["post"]["mean"][:, -1])


@pytest.mark.parametrize("num_blocks", [2, 0])
def test_acting_step_by_step_matches_parallel_training_pass(num_blocks):
    model, _ = _build(num_blocks)
    obs, actions = _batch()[0][0], _batch()[1]
    with torch.no_grad(), MAPEstimate(model):
        emb = model.encode([obs])
        out = model.transition_model.forward_pass(emb, actions, smooth=False)
        state = model.get_initial_state(batch_size=B)
        means, ms = [], []
        for t in range(T):
            state = model.transition_model.get_next_posterior(latent_obs=[emb[0][:, t]], action=actions[:, t],
                                                              post_state=state)
            means.append(state["mean"])
            ms.append(state["m"])
    torch.testing.assert_close(torch.stack(ms, 1), out["m"], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(torch.stack(means, 1), out["post"]["mean"], rtol=1e-3, atol=1e-3)


def test_windowed_acting_matches_training_windows():
    window = 8
    model, _ = _build(act_window=window)
    obs, actions = _batch()[0][0], _batch()[1]
    with torch.no_grad(), MAPEstimate(model):
        emb = model.encode([obs])
        state = model.get_initial_state(batch_size=B)
        for t in range(T):
            state = model.transition_model.get_next_posterior(latent_obs=[emb[0][:, t]], action=actions[:, t],
                                                              post_state=state)
            s = max(0, t - window + 1)
            ref = model.transition_model.forward_pass([emb[0][:, s:t + 1]], actions[:, s:t + 1], smooth=False)
            torch.testing.assert_close(state["mean"], ref["post"]["mean"][:, -1])
            torch.testing.assert_close(state["var"], ref["post"]["var"][:, -1])
            assert state["emb_hist"].shape[1] == min(t + 1, window)


@pytest.mark.parametrize("act_window", [0, 32])
def test_policy_interface_runs_one_episode_step(act_window):
    model, _ = _build(act_window=act_window)
    with torch.inference_mode():
        state = model.get_initial_state(batch_size=1)
        for _ in range(3):
            state = model.get_next_posterior(observation=[torch.randn(1, OBS_DIM).cuda()],
                                             action=torch.zeros(1, ACT_DIM).cuda(), post_state=state)
    assert model.get_deterministic_features(state).shape == (1, 230)


def test_vrkn_loss_is_finite_and_trains_all_parts():
    _, objective = _build(transition_factory=VRKNTransitionFactory)
    loss, log_dict, _ = objective.compute_losses_and_states(*_batch())
    assert torch.isfinite(loss) and "mahalanobis" not in log_dict
    loss.backward()
    assert [n for n, p in objective.named_parameters() if p.requires_grad and p.grad is None] == []


@pytest.mark.skipif(not os.path.isdir(VRKN_REF), reason="pbecker93/vrkn checkout not available")
def test_vrkn_port_matches_original_vrkn():
    sys.path.insert(0, VRKN_REF)
    from ssm_mbrl.common.bayesian_modules import MAPEstimate as RefMAPEstimate
    from ssm_mbrl.vrkn.transition.vrkn_tm import VRKNTM as RefVRKNTM

    model, _ = _build(transition_factory=VRKNTransitionFactory)
    port = model.transition_model
    c = VRKNTransitionFactory.get_default_config()
    ref = RefVRKNTM(obs_sizes=[300], action_dim=ACT_DIM, lod=c.lod, mc_drop_prob=c.mc_drop_prob,
                    init_obs_var=c.init_obs_var, min_obs_var=c.min_obs_var, max_obs_var=5.0, obs_var_sigmoid=False,
                    obs_output_normalization="none", hidden_size=c.hidden_size, activation=c.activation,
                    init_ev=c.init_ev, min_ev=c.min_ev, max_ev=c.max_ev, init_trans_var=c.init_trans_var,
                    min_trans_var=c.min_trans_var, max_trans_var=c.max_trans_var, trans_var_sigmoid=True,
                    learn_initial_state_var=True, initial_state_var=c.initial_state_var,
                    min_initial_state_var=c.min_initial_state_var).cuda()
    pairs = [(ref._tm_pre_layers[0], port._dyn_pre[0]), (ref._fake_gru, port._dyn_cell),
             (ref._tm_hidden_net[1], port._dyn_post[1]), (ref._tm_hidden_net[4], port._dyn_post[4]),
             (ref._latent_obs_nets[0]._mean_net, port._latent_obs_net._mean_net),
             (ref._latent_obs_nets[0]._std_net[0], port._latent_obs_net._std_net[0])]
    with torch.no_grad():
        for r, p in pairs:
            for name, param in r.named_parameters():
                param.copy_(dict(p.named_parameters())[name])
        ref.initial_state_model._initial_variance_model._raw_transition_var.copy_(port._initial_var._raw_var)

    obs, actions = _batch()[0][0], _batch()[1]
    with torch.no_grad(), MAPEstimate(model), RefMAPEstimate(ref):
        emb = model.encode([obs])
        out = port.forward_pass(emb, actions)
        posts, priors, tms = ref.forward_pass(embedded_obs=emb, actions=actions)
        smoothed = ref.extended_backward_pass(post_states=posts, prior_states=priors, transition_models=tms,
                                              full_smoothing=True)
    stack = lambda states, key: torch.stack([s[key] for s in states], dim=1)
    for ours, (states, key) in [(out["post"]["mean"], (posts, "mean")), (out["post"]["var"], (posts, "var")),
                                (out["prior"]["mean"], (priors, "mean")), (out["prior"]["var"], (priors, "var")),
                                (out["smoothed"]["mean"], (smoothed, "mean")),
                                (out["smoothed"]["var"], (smoothed, "var"))]:
        torch.testing.assert_close(ours, stack(states, key), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out["smoothed"]["cond_dyn_var"], stack(smoothed[1:], "cond_dyn_var"),
                               rtol=1e-4, atol=1e-5)
