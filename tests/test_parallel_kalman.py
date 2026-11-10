"""Check the parallel Kalman filter/smoother against a sequential VRKN-style reference."""
import pytest
import torch

from ssm_rl.kalmamba import parallel_kalman as pk

LENGTHS = [1, 2, 3, 7, 32, 33, 100]


def _random_ssm(batch, length, dim, dtype=torch.float64, requires_grad=False):
    g = torch.Generator().manual_seed(length)

    def r(*shape, low=-1.0, high=1.0):
        x = low + (high - low) * torch.rand(*shape, generator=g, dtype=dtype)
        return x.requires_grad_(requires_grad)

    return dict(init_mean=r(batch, dim), init_var=r(batch, dim, low=0.1, high=2.0),
                tm=r(batch, length - 1, dim, low=0.4, high=0.99), offset=r(batch, length - 1, dim),
                trans_var=r(batch, length - 1, dim, low=0.01, high=1.0),
                obs=r(batch, length, dim, low=-3, high=3), obs_var=r(batch, length, dim, low=0.01, high=3.0))


def _sequential(p):
    """Same equations and step-0 convention as VRKNTM._forward_pass / extended_backward_pass."""
    T = p["obs"].shape[1]
    post_m, post_v, prior_m, prior_v = [], [], [], []
    m, v = p["init_mean"], p["init_var"]
    for t in range(T):
        if t > 0:
            m = p["tm"][:, t - 1] * post_m[-1] + p["offset"][:, t - 1]
            v = p["tm"][:, t - 1].square() * post_v[-1] + p["trans_var"][:, t - 1]
        prior_m.append(m), prior_v.append(v)
        k = v / (v + p["obs_var"][:, t])
        m = (1 - k) * m + k * p["obs"][:, t]
        v = (1 - k).square() * v + k.square() * p["obs_var"][:, t]  # Joseph form, as in the VRKN
        post_m.append(m), post_v.append(v)

    sm, sv, gains = [None] * T, [None] * T, [torch.zeros_like(post_m[-1])] * T
    sm[-1], sv[-1] = post_m[-1], post_v[-1]
    for t in range(T - 2, -1, -1):
        c = post_v[t] * p["tm"][:, t] / prior_v[t + 1]
        sm[t] = post_m[t] + c * (sm[t + 1] - prior_m[t + 1])
        sv[t] = post_v[t] + c.square() * (sv[t + 1] - prior_v[t + 1])
        gains[t] = c
    stack = lambda xs: torch.stack(xs, dim=1)
    return dict(post_mean=stack(post_m), post_var=stack(post_v), prior_mean=stack(prior_m),
                prior_var=stack(prior_v), mean=stack(sm), var=stack(sv), gain=stack(gains))


def _parallel(p):
    f = pk.parallel_filter(**p)
    s = pk.parallel_smoother(f["post_mean"], f["post_var"], f["prior_mean"], f["prior_var"], p["tm"])
    return {**f, **s}


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("reverse", [False, True])
def test_associative_scan_matches_cumsum(length, reverse):
    x = torch.randn(3, length, 5, dtype=torch.float64)
    (out,) = pk.associative_scan(lambda a, b: (a[0] + b[0],), (x,), dim=1, reverse=reverse)
    expected = x.flip(1).cumsum(1).flip(1) if reverse else x.cumsum(1)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("length", LENGTHS)
def test_filter_and_smoother_match_sequential(length):
    p = _random_ssm(batch=4, length=length, dim=6)
    par, seq = _parallel(p), _sequential(p)
    for key in seq:
        torch.testing.assert_close(par[key], seq[key], rtol=1e-9, atol=1e-9, msg=key)


def test_gradients_match_sequential():
    p = _random_ssm(batch=2, length=33, dim=3, requires_grad=True)
    weights = torch.randn(2, 33, 3, dtype=torch.float64)

    def grads(fn):
        out = fn(p)
        loss = sum((out[k] * weights).sum() for k in ("post_mean", "post_var", "mean", "var"))
        return torch.autograd.grad(loss, list(p.values()))

    for gp, gs in zip(grads(_parallel), grads(_sequential)):
        torch.testing.assert_close(gp, gs, rtol=1e-8, atol=1e-10)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_float32_long_sequence_is_accurate(device):
    p64 = _random_ssm(batch=8, length=1024, dim=16)
    p32 = {k: v.float().to(device) for k, v in p64.items()}
    par, seq = _parallel(p32), _sequential(p64)
    for key in ("post_mean", "post_var", "mean", "var"):
        torch.testing.assert_close(par[key].double().cpu(), seq[key], rtol=1e-3, atol=1e-4, msg=key)


def test_filter_step_matches_parallel_filter():
    p = _random_ssm(batch=4, length=10, dim=5)
    f = pk.parallel_filter(**p)
    m, v = pk.update_step(p["init_mean"], p["init_var"], p["obs"][:, 0], p["obs_var"][:, 0])
    for t in range(1, 10):
        m, v = pk.filter_step(m, v, p["tm"][:, t - 1], p["offset"][:, t - 1], p["trans_var"][:, t - 1],
                              p["obs"][:, t], p["obs_var"][:, t])
    torch.testing.assert_close(m, f["post_mean"][:, -1])
    torch.testing.assert_close(v, f["post_var"][:, -1])


def test_smoothed_conditionals_match_vrkn():
    p = _random_ssm(batch=3, length=12, dim=4)
    s = _parallel(p)
    sample = s["mean"] + s["var"].sqrt() * torch.randn_like(s["mean"])
    cond = pk.smoothed_conditionals(s["mean"], s["var"], sample, s["gain"], p["tm"], p["offset"], p["trans_var"])
    for t in range(1, 12):  # VRKNTM.extended_rts_backward for the step t-1 -> t
        c = s["gain"][:, t - 1]
        mat = s["var"][:, t] * c / s["var"][:, t - 1]
        torch.testing.assert_close(cond["cond_dyn_mean"][:, t - 1],
                                   s["mean"][:, t] + mat * (sample[:, t - 1] - s["mean"][:, t - 1]))
        torch.testing.assert_close(cond["cond_dyn_var"][:, t - 1], s["var"][:, t] - mat * c * s["var"][:, t])
        torch.testing.assert_close(cond["dyn_mean"][:, t - 1],
                                   p["tm"][:, t - 1] * sample[:, t - 1] + p["offset"][:, t - 1])
    assert (cond["cond_dyn_var"] > 0).all()
