"""Time-parallel Kalman filtering and smoothing for diagonal linear Gaussian SSMs.

Implements the associative formulation of Saerkkae & Garcia-Fernandez,
"Temporal Parallelization of Bayesian Smoothers" (2020), for the latent model used by
KalMamba and the VRKN:

    z_0 ~ N(m_0, P_0)
    z_t = A_t z_{t-1} + b_t + eps_t,  eps_t ~ N(0, Q_t)     t = 1 .. T-1
    w_t = z_t + delta_t,              delta_t ~ N(0, R_t)   t = 0 .. T-1

All matrices are diagonal, so every operation is element-wise. Tensors are (batch, time, dim),
and the transition tensors A, b, Q hold the T-1 transitions into steps 1 .. T-1.
The step-0 convention (the prior of z_0 is updated with w_0, no transition) follows
VRKNTM._forward_pass.
"""
from typing import Callable

import torch

Elems = tuple[torch.Tensor, ...]


def _slice(x: torch.Tensor, start, stop, step, dim: int) -> torch.Tensor:
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(start, stop, step)
    return x[tuple(idx)]


def _interleave(even: torch.Tensor, odd: torch.Tensor, dim: int) -> torch.Tensor:
    """Merge so that the result is even[0], odd[0], even[1], odd[1], ..."""
    n_even, n_odd = even.shape[dim], odd.shape[dim]
    if n_even > n_odd:
        pad = _slice(odd, 0, 1, None, dim)
        odd = torch.cat([odd, torch.zeros_like(pad)], dim=dim)
    out = torch.stack([even, odd], dim=dim + 1).flatten(dim, dim + 1)
    return _slice(out, 0, n_even + n_odd, None, dim)


def associative_scan(fn: Callable[[Elems, Elems], Elems], elems: Elems, dim: int = 1,
                     reverse: bool = False) -> Elems:
    """Inclusive scan of an associative ``fn`` along ``dim`` in O(log T) sequential steps.

    ``fn(a, b)`` combines a with the element that follows it. Same algorithm as
    jax.lax.associative_scan. With ``reverse=True`` element t holds fn(x_t, fn(x_{t+1}, ...)).
    """
    if reverse:
        flipped = tuple(e.flip(dim) for e in elems)
        out = associative_scan(lambda a, b: fn(b, a), flipped, dim=dim)
        return tuple(o.flip(dim) for o in out)

    n = elems[0].shape[dim]
    if n < 2:
        return elems
    reduced = fn(tuple(_slice(e, 0, -1, 2, dim) for e in elems),
                 tuple(_slice(e, 1, None, 2, dim) for e in elems))
    odd = associative_scan(fn, reduced, dim=dim)
    if n % 2 == 0:
        even = fn(tuple(_slice(o, 0, -1, None, dim) for o in odd),
                  tuple(_slice(e, 2, None, 2, dim) for e in elems))
    else:
        even = fn(odd, tuple(_slice(e, 2, None, 2, dim) for e in elems))
    even = tuple(torch.cat([_slice(e, 0, 1, None, dim), ev], dim=dim) for e, ev in zip(elems, even))
    return tuple(_interleave(ev, od, dim) for ev, od in zip(even, odd))


def _filter_combine(a: Elems, b: Elems) -> Elems:
    a_A, a_b, a_C, a_eta, a_J = a
    b_A, b_b, b_C, b_eta, b_J = b
    d = 1.0 / (1.0 + a_C * b_J)
    return (b_A * d * a_A,
            b_A * d * (a_b + a_C * b_eta) + b_b,
            b_A.square() * d * a_C + b_C,
            a_A * d * (b_eta - b_J * a_b) + a_eta,
            a_A.square() * d * b_J + a_J)


def _smoother_combine(a: Elems, b: Elems) -> Elems:
    # a is the earlier time step, b the later one.
    a_E, a_g, a_L = a
    b_E, b_g, b_L = b
    return a_E * b_E, a_E * b_g + a_g, a_E.square() * b_L + a_L


def _prepend(first: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    return torch.cat([first.unsqueeze(1), rest], dim=1)


def parallel_filter(init_mean: torch.Tensor, init_var: torch.Tensor,
                    tm: torch.Tensor, offset: torch.Tensor, trans_var: torch.Tensor,
                    obs: torch.Tensor, obs_var: torch.Tensor) -> dict[str, torch.Tensor]:
    """Filtered beliefs q(z_t | w_{<=t}) and the one-step priors, all of shape (B, T, D).

    init_mean, init_var: (B, D).  tm, offset, trans_var: (B, T-1, D).  obs, obs_var: (B, T, D).
    """
    # Step 0 has no transition: F = 0 and the prior N(m_0, P_0) enters as offset and noise.
    F = _prepend(torch.zeros_like(init_mean), tm)
    u = _prepend(init_mean, offset)
    Q = _prepend(init_var, trans_var)

    s = Q + obs_var
    gain = Q / s
    residual = obs - u
    elems = ((1 - gain) * F,
             u + gain * residual,
             Q * obs_var / s,
             F * residual / s,
             F.square() / s)
    _, post_mean, post_var, _, _ = associative_scan(_filter_combine, elems, dim=1)

    prior_mean = _prepend(init_mean, tm * post_mean[:, :-1] + offset)
    prior_var = _prepend(init_var, tm.square() * post_var[:, :-1] + trans_var)
    return {"post_mean": post_mean, "post_var": post_var,
            "prior_mean": prior_mean, "prior_var": prior_var}


def parallel_smoother(post_mean: torch.Tensor, post_var: torch.Tensor,
                      prior_mean: torch.Tensor, prior_var: torch.Tensor,
                      tm: torch.Tensor) -> dict[str, torch.Tensor]:
    """RTS-smoothed beliefs q(z_t | w_{0..T-1}) and the smoother gains, all (B, T, D).

    gain[:, t] is the VRKN's ``c`` for the step t -> t+1; the last entry is zero.
    """
    gain = post_var[:, :-1] * tm / prior_var[:, 1:]
    g = post_mean[:, :-1] - gain * prior_mean[:, 1:]
    L = post_var[:, :-1] - gain.square() * prior_var[:, 1:]

    # The last step is its own smoothed belief: E = 0, g = m_{T-1}, L = P_{T-1}.
    gain = torch.cat([gain, torch.zeros_like(post_var[:, -1:])], dim=1)
    g = torch.cat([g, post_mean[:, -1:]], dim=1)
    L = torch.cat([L, post_var[:, -1:]], dim=1)
    _, smoothed_mean, smoothed_var = associative_scan(_smoother_combine, (gain, g, L), dim=1, reverse=True)
    return {"mean": smoothed_mean, "var": smoothed_var, "gain": gain}


def smoothed_conditionals(smoothed_mean: torch.Tensor, smoothed_var: torch.Tensor,
                          smoothed_sample: torch.Tensor, gain: torch.Tensor,
                          tm: torch.Tensor, offset: torch.Tensor,
                          trans_var: torch.Tensor) -> dict[str, torch.Tensor]:
    """Parameters of q(z_t | z_{t-1}, w_{>=t}) and p(z_t | z_{t-1}) for t = 1 .. T-1, each (B, T-1, D).

    Element-wise version of VRKNTM.extended_rts_backward, evaluated at a sample of the
    smoothed belief of z_{t-1}.
    """
    c = gain[:, :-1]
    cond_dyn_mat = smoothed_var[:, 1:] * c / smoothed_var[:, :-1]
    residual = smoothed_sample[:, :-1] - smoothed_mean[:, :-1]
    return {"cond_dyn_mean": smoothed_mean[:, 1:] + cond_dyn_mat * residual,
            "cond_dyn_var": smoothed_var[:, 1:] - cond_dyn_mat * c * smoothed_var[:, 1:],
            "dyn_mean": tm * smoothed_sample[:, :-1] + offset,
            "dyn_var": trans_var}


def filter_step(mean: torch.Tensor, var: torch.Tensor, tm: torch.Tensor, offset: torch.Tensor,
                trans_var: torch.Tensor, obs: torch.Tensor,
                obs_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequential predict + update, used when acting in the environment."""
    prior_mean = tm * mean + offset
    prior_var = tm.square() * var + trans_var
    return update_step(prior_mean, prior_var, obs, obs_var)


def update_step(mean: torch.Tensor, var: torch.Tensor, obs: torch.Tensor,
                obs_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gain = var / (var + obs_var)
    return mean + gain * (obs - mean), (1 - gain) * var
