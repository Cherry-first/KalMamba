"""Time the parallel Kalman filter + smoother against a sequential loop for growing sequence lengths.

This covers only the latent inference part of the paper's Figure 6 (left); the full SSM forward
pass also includes the encoder and the Mamba backbone.
    python scripts/bench_kalman_scan.py --batch 32 --dim 230
"""
import argparse
import time

import torch

from ssm_rl.kalmamba import parallel_kalman as pk


def sequential(init_mean, init_var, tm, offset, trans_var, obs, obs_var):
    T = obs.shape[1]
    m, v = pk.update_step(init_mean, init_var, obs[:, 0], obs_var[:, 0])
    post_m, post_v, prior_m, prior_v = [m], [v], [init_mean], [init_var]
    for t in range(1, T):
        pm = tm[:, t - 1] * m + offset[:, t - 1]
        pv = tm[:, t - 1].square() * v + trans_var[:, t - 1]
        m, v = pk.update_step(pm, pv, obs[:, t], obs_var[:, t])
        post_m.append(m), post_v.append(v), prior_m.append(pm), prior_v.append(pv)
    sm, sv = [post_m[-1]], [post_v[-1]]
    for t in range(T - 2, -1, -1):
        c = post_v[t] * tm[:, t] / prior_v[t + 1]
        sm.append(post_m[t] + c * (sm[-1] - prior_m[t + 1]))
        sv.append(post_v[t] + c.square() * (sv[-1] - prior_v[t + 1]))
    return torch.stack(sm[::-1], 1), torch.stack(sv[::-1], 1)


def parallel(init_mean, init_var, tm, offset, trans_var, obs, obs_var):
    f = pk.parallel_filter(init_mean, init_var, tm, offset, trans_var, obs, obs_var)
    s = pk.parallel_smoother(f["post_mean"], f["post_var"], f["prior_mean"], f["prior_var"], tm)
    return s["mean"], s["var"]


def timeit(fn, args, backward, reps):
    def run():
        mean, var = fn(*args)
        if backward:
            (mean.sum() + var.sum()).backward()
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dim", type=int, default=230)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    args = parser.parse_args()

    dev = "cuda"
    print(f"{torch.cuda.get_device_name(0)}, batch {args.batch}, dim {args.dim}, mean of {args.reps} reps")
    print(f"{'T':>6} {'seq fwd':>10} {'par fwd':>10} {'seq f+b':>10} {'par f+b':>10}   (ms)")
    for T in args.lengths:
        B, D = args.batch, args.dim
        x = lambda *s, lo=-1.0, hi=1.0: (lo + (hi - lo) * torch.rand(*s, device=dev)).requires_grad_()
        inputs = (x(B, D), x(B, D, lo=0.1, hi=1), x(B, T - 1, D, lo=0.4, hi=0.99), x(B, T - 1, D),
                  x(B, T - 1, D, lo=0.01, hi=1), x(B, T, D), x(B, T, D, lo=0.01, hi=1))
        times = []
        for backward in (False, True):
            ctx = torch.enable_grad() if backward else torch.no_grad()
            with ctx:
                times += [timeit(fn, inputs, backward, args.reps) * 1e3 for fn in (sequential, parallel)]
        print(f"{T:>6} {times[0]:>10.2f} {times[1]:>10.2f} {times[2]:>10.2f} {times[3]:>10.2f}")


if __name__ == "__main__":
    main()
