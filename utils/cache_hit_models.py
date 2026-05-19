import numpy as np
from scipy.optimize import brentq


def che_characteristic_time(qs, C, t0=1e-9, grow=10.0, max_iter=60):
    """
    Solve for t_C in: C = sum_i (1 - exp(-q_i * t_C)).
    qs is a probability vector or request-rate vector with q_i >= 0.
    """
    qs = np.asarray(qs, dtype=float)
    if np.any(~np.isfinite(qs)) or np.any(qs < 0):
        raise ValueError("qs must be finite and >= 0")

    m = int(np.sum(qs > 0))
    if C <= 0:
        return 0.0
    if C >= m:
        return np.inf

    qpos = qs[qs > 0]
    qbar = float(np.mean(qpos)) if qpos.size > 0 else 1.0
    r = np.where(qs > 0, qs / qbar, 0.0)

    def f_tau(tau):
        return np.sum(1.0 - np.exp(-r * tau)) - C

    a, b = t0, t0
    fa = f_tau(a)
    for _ in range(max_iter):
        b *= grow
        fb = f_tau(b)
        if fa * fb <= 0:
            break
    else:
        a, b = 1e-12, 1e12

    tau = brentq(f_tau, a, b)
    return tau / qbar


def lru_hit_ratio(qs, C, Q=0):
    """
    Che's approximation for LRU.
    qs must be a probability vector summing to 1.
    """
    m = int(np.sum(qs > 0))
    if C >= m:
        return 1.0

    t_C = che_characteristic_time(qs, C)
    if np.isinf(t_C):
        return 1.0

    hit_rates = 1.0 - np.exp(-qs * t_C)
    return float(np.sum(qs * hit_rates))


def lfu_hit_ratio(qs, C):
    """
    Exact LFU hit ratio under IRM.
    qs must be a probability vector summing to 1.
    """
    if C <= 0:
        return 0.0

    qs = np.asarray(qs, dtype=np.float64)
    qs = qs[qs > 0]
    if qs.size == 0:
        return 0.0

    m = qs.size
    if C >= m:
        return 1.0

    qs_sorted = np.sort(qs)[::-1]
    return float(np.sum(qs_sorted[:int(C)]))

def lfu_hit_ratio_cold(qs, C, Q, rho_mode="none", gamma=1.0):
    """
    Finite-trace LFU hit ratio with cold-start correction.

    qs: page popularity distribution, sum(qs)=1
    C : cache capacity in pages
    Q : trace length (#page requests)
    rho_mode:
      - "none"   : rho = 1
      - "linear" : rho = max(0, 1 - gamma * C / Q)
      - "exp"    : rho = exp(-gamma * C / Q)
    """
    qs = np.asarray(qs, dtype=np.float64)
    qs = qs[qs > 0]
    if qs.size == 0 or C <= 0 or Q <= 0:
        return 0.0

    qs = qs / qs.sum()
    m = qs.size
    if C >= m:
        # all pages can eventually fit, but still pay compulsory misses
        steady = 1.0
        cold = np.sum(1.0 - np.power(1.0 - qs, Q)) / Q
        return max(0.0, steady - cold)

    qs_sorted = np.sort(qs)[::-1]
    top = qs_sorted[:int(C)]

    steady = float(np.sum(top))
    cold = float(np.sum(1.0 - np.power(1.0 - top, Q)) / Q)

    if rho_mode == "none":
        rho = 1.0
    elif rho_mode == "linear":
        rho = max(0.0, 1.0 - gamma * C / Q)
    elif rho_mode == "exp":
        rho = float(np.exp(-gamma * C / Q))
    else:
        raise ValueError(f"Unknown rho_mode: {rho_mode}")

    h = rho * steady - cold
    return max(0.0, min(1.0, h))

def fifo_random_characteristic_time(qs, C, t0=1e-12, grow=10.0, max_iter=80):
    """
    Solve tau_C from:
        C = sum_i (q_i * tau) / (1 - q_i + q_i * tau)
    qs should sum to 1.
    """
    qs = np.asarray(qs, dtype=np.float64)
    if np.any(qs < 0) or np.any(~np.isfinite(qs)):
        raise ValueError("qs must be finite and non-negative")
    qs = qs[qs > 0]
    m = qs.size
    if C <= 0:
        return 0.0
    if C >= m:
        return np.inf

    def f(tau):
        vals = (qs * tau) / (1.0 - qs + qs * tau)
        return np.sum(vals) - C

    a = t0
    fa = f(a)
    b = a
    for _ in range(max_iter):
        b *= grow
        fb = f(b)
        if fa * fb <= 0:
            return brentq(f, a, b)
    return np.inf


def fifo_random_hit_rates(qs, tau_C):
    """
    Per-object hit probability under FIFO/RANDOM approximation.
    """
    qs = np.asarray(qs, dtype=np.float64)
    if np.isinf(tau_C):
        return np.ones_like(qs)
    return (qs * tau_C) / (1.0 - qs + qs * tau_C)


def fifo_random_hit_ratio(qs, C):
    """
    Approximate FIFO/RANDOM hit ratio under IRM.
    """
    qs = np.asarray(qs, dtype=np.float64)
    qs = qs[qs > 0]
    if qs.size == 0 or C <= 0:
        return 0.0

    m = qs.size
    if C >= m:
        return 1.0

    tau_C = fifo_random_characteristic_time(qs, C)
    if np.isinf(tau_C):
        return 1.0

    h_i = fifo_random_hit_rates(qs, tau_C)
    return float(np.sum(qs * h_i))


def validate_ratio(ratio):
    if ratio >= 1.0:
        return 1.0
    if ratio <= 0.0:
        return 0.0
    return ratio


def cache_hit_ratio(policy, C, qs, Q=0):
    """
    policy: "LRU" | "LFU" | "FIFO" | "RANDOM"
    C: cache capacity in pages
    qs: page popularity distribution (sum(qs)=1)
    """
    qs = np.asarray(qs, dtype=np.float64)
    if qs.size == 0 or C <= 0:
        return 0.0

    qs = qs[qs > 0]
    if qs.size == 0:
        return 0.0

    s = qs.sum()
    if s <= 0:
        return 0.0
    qs = qs / s

    policy = policy.upper()
    if policy == "LRU":
        return lru_hit_ratio(qs, C, Q)
    if policy == "LFU":
        return lfu_hit_ratio(qs, C)
        # return lfu_hit_ratio_cold(qs, C, Q)
    if policy in ("FIFO", "RANDOM"):
        return fifo_random_hit_ratio(qs, C)
    raise ValueError(f"Unknown policy: {policy}")
