"""
Stress tests. run_simulations.py checks each method on clean, well behaved
data. Real metrics aren't like that: revenue is skewed, a few whales
dominate, most users buy nothing, conversions can be 1 in 1,000, users
have many sessions, sometimes whole cities get randomized together, values
go missing, and the effect isn't the same for everyone.

For each kind of messy data this runs thousands of made up experiments
with no effect (to measure false positives) and with a known effect (to
measure coverage, power and bias), and times every method.

Run:  python simulations/stress.py          (about 6 minutes on one core)
      python simulations/stress.py --quick  (a smoke test, under a minute)
      python simulations/stress.py --only sessions,clusters_8   (just those)
      python simulations/stress.py --plot   (redraw the figures from results/stress.json)
Writes results/stress.json and figures/stress_*.png.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402

QUICK = "--quick" in sys.argv
N_NULL, N_ALT = (200, 100) if QUICK else (2000, 1000)
LOOKS = 50
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, GREY, RED = "#2F4BC4", "#9AA3B2", "#C2413B"


# ------------------------------------------------------------ data makers
# each returns a dict with control "c" and treatment "t" outcomes, the true
# difference in population means "true", and optionally a pre period
# covariate (xc, xt) or cluster ids (gc, gt)

def lognormal(sigma, nc, nt):
    mu = np.exp(sigma**2 / 2)                       # mean of exp(N(0, sigma^2))
    def make(rng, lift):
        # pre period spend shares the user's latent level with this period's spend
        zc, zt = rng.normal(0, 1, nc), rng.normal(0, 1, nt)
        xc = np.exp(sigma * (0.7 * zc + np.sqrt(0.51) * rng.normal(0, 1, nc)))
        xt = np.exp(sigma * (0.7 * zt + np.sqrt(0.51) * rng.normal(0, 1, nt)))
        c = np.exp(sigma * zc)
        t = np.exp(sigma * zt) * (1 + lift)
        return {"c": c, "t": t, "xc": xc, "xt": xt, "true": mu * lift}
    sd = np.sqrt((np.exp(sigma**2) - 1) * np.exp(sigma**2))
    return make, sd / mu, 2.8 * sd * np.sqrt(1 / nc + 1 / nt) / mu


def pareto(shape, n):
    # classic Pareto with minimum 1: mean shape / (shape - 1), infinite variance below 2
    mu = shape / (shape - 1)
    def make(rng, lift):
        return {"c": rng.pareto(shape, n) + 1, "t": (rng.pareto(shape, n) + 1) * (1 + lift),
                "true": mu * lift}
    return make


def student_t(df, n):
    def make(rng, eff):
        return {"c": rng.standard_t(df, n), "t": rng.standard_t(df, n) + eff, "true": eff}
    return make


def zero_inflated(buy, sigma, n):
    # most users spend nothing; each user has their own chance to buy, and
    # buyers spend a lognormal amount. pre period = purchases in 3 past months
    mu = buy * np.exp(sigma**2 / 2)
    def make(rng, lift):
        uc, ut = rng.beta(1, 1 / buy - 1, n), rng.beta(1, 1 / buy - 1, n)
        c = (rng.random(n) < uc) * np.exp(sigma * rng.normal(0, 1, n))
        t = (rng.random(n) < ut * (1 + lift)) * np.exp(sigma * rng.normal(0, 1, n))
        xc, xt = rng.binomial(3, uc).astype(float), rng.binomial(3, ut).astype(float)
        return {"c": c, "t": t, "xc": xc, "xt": xt, "true": mu * lift}
    return make


def binary(p, n):
    def make(rng, eff):
        return {"c": (rng.random(n) < p).astype(float), "t": (rng.random(n) < p + eff).astype(float),
                "true": eff}
    return make


def sessions(n_users, icc_a, icc_b):
    # user randomized, analyzed per session: each user has their own click rate
    # and between 1 and ~12 sessions
    base = icc_a / (icc_a + icc_b)
    def make(rng, eff):
        def arm(shift):
            rate = np.clip(rng.beta(icc_a, icc_b, n_users) + shift, 0, 1)
            k = rng.poisson(4, n_users) + 1
            users = np.repeat(np.arange(n_users), k)
            return (rng.random(k.sum()) < rate[users]).astype(float), users
        c, gc = arm(0.0)
        t, gt = arm(eff)
        return {"c": c, "t": t, "gc": gc, "gt": gt, "true": eff}
    return make, base


def clustered(n_clusters, mean_size, icc):
    # whole clusters (cities, schools) randomized; sizes vary a lot
    def make(rng, eff):
        def arm(shift):
            size = np.maximum(1, rng.lognormal(np.log(mean_size) - 0.5, 1.0, n_clusters).astype(int))
            g = np.repeat(np.arange(n_clusters), size)
            b = rng.normal(0, np.sqrt(icc), n_clusters)
            return b[g] + rng.normal(0, np.sqrt(1 - icc), len(g)) + shift, g
        c, gc = arm(0.0)
        t, gt = arm(eff)
        return {"c": c, "t": t, "gc": gc, "gt": gt, "true": eff}
    return make


def missing(kind, n):
    # outcome y ~ lognormal-ish spend with a pre period covariate; some values
    # go missing before analysis
    def make(rng, eff):
        zc, zt = rng.normal(0, 1, n), rng.normal(0, 1, n)
        xc, xt = zc + rng.normal(0, 0.7, n), zt + rng.normal(0, 0.7, n)
        c, t = zc, zt + eff
        if kind == "mcar":
            mc, mt = rng.random(n) < 0.2, rng.random(n) < 0.2
        elif kind == "mar":          # missing more often for low pre period activity
            mc = rng.random(n) < 1 / (1 + np.exp(1.5 + 1.5 * xc))
            mt = rng.random(n) < 1 / (1 + np.exp(1.5 + 1.5 * xt))
        elif kind == "mnar":         # low outcomes go missing more, same rule in both arms
            mc = rng.random(n) < 1 / (1 + np.exp(1.5 + 1.5 * c))
            mt = rng.random(n) < 1 / (1 + np.exp(1.5 + 1.5 * t))
        else:                        # "differential": treatment makes low users drop out of logging
            mc = rng.random(n) < 0.10
            mt = rng.random(n) < np.where(t < -1, 0.30, 0.10)
        return {"c": c[~mc], "t": t[~mt], "xc": xc[~mc], "xt": xt[~mt], "true": eff,
                "miss_c": mc.astype(float), "miss_t": mt.astype(float)}
    return make


def hte(nc, nt, gamma):
    # effect differs by user: eff + gamma * x, x is pre period (mean 0)
    def make(rng, eff):
        xc, xt = rng.normal(0, 1, nc), rng.normal(0, 1, nt)
        c = 0.7 * xc + rng.normal(0, 0.7, nc)
        t = 0.7 * xt + rng.normal(0, 0.7, nt) + eff + gamma * xt
        return {"c": c, "t": t, "xc": xc, "xt": xt, "true": eff}
    return make


# ----------------------------------------------------------------- methods
# each returns (estimate, ci_low, ci_high, p)

def m_welch(d, tau):
    r = ab.mean_test(d["c"], d["t"])
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_capped(d, tau):
    cap = np.quantile(np.concatenate([d["c"], d["t"]]), 0.99)
    r = ab.mean_test(np.minimum(d["c"], cap), np.minimum(d["t"], cap))
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_cuped(d, tau):
    nc = len(d["c"])
    y_adj, _ = ab.cuped_adjust(np.concatenate([d["c"], d["t"]]), np.concatenate([d["xc"], d["xt"]]))
    r = ab.mean_test(y_adj[:nc], y_adj[nc:])
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_lin(d, tau):
    r = ab.adjusted_test(d["c"], d["t"], d["xc"], d["xt"])
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_cluster(d, tau):
    r = ab.cluster_test(d["c"], d["t"], d["gc"], d["gt"])
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_cluster_perm(d, tau):
    r = ab.cluster_test(d["c"], d["t"], d["gc"], d["gt"], permutations=999, seed=int(tau * 1e6) % 1000)
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_delta_normal(d, tau):
    # the same cluster level delta method with a plain normal reference
    sc, kc = np.bincount(d["gc"], weights=d["c"]), np.bincount(d["gc"]).astype(float)
    st, kt = np.bincount(d["gt"], weights=d["t"]), np.bincount(d["gt"]).astype(float)
    r = ab.ratio_test(sc, kc, st, kt)
    return r.lift, r.ci[0], r.ci[1], r.p_value


def m_bayes(d, tau):
    r = ab.bayes_test(d["c"], d["t"], draws=4000)
    p = 2 * min(r.chance_treatment_better, 1 - r.chance_treatment_better)
    return r.lift_interval[0] / 2 + r.lift_interval[1] / 2, r.lift_interval[0], r.lift_interval[1], p


METHODS = {"welch": m_welch, "capped_99": m_capped, "cuped": m_cuped, "lin": m_lin,
           "cluster": m_cluster, "cluster_perm": m_cluster_perm, "delta_normal": m_delta_normal, "bayes_flat": m_bayes}


def run_msprt(d, tau):
    r = ab.msprt(d["c"], d["t"], tau=tau, n_looks=LOOKS)
    miss = bool(((r.ci_low > d["true"]) | (r.ci_high < d["true"])).any())
    share = (r.n_control[r.stop_index] + r.n_treatment[r.stop_index]) / (len(d["c"]) + len(d["t"])) \
        if r.stopped else np.nan
    empty = bool((r.ci_low > r.ci_high).any())
    return r.stopped, miss, share, empty


# ------------------------------------------------------------- scenarios
ln1, skew1, lift1 = lognormal(1.0, 2000, 2000)
ln2, skew2, lift2 = lognormal(2.0, 2000, 2000)
ln2_big, _, lift2_big = lognormal(2.0, 20000, 20000)
ln2_uneven, _, lift2_uneven = lognormal(2.0, 1000, 19000)
sess, sess_base = sessions(2000, 2, 8)

FIXED = ["welch", "bayes_flat"]
SKEWED = ["welch", "capped_99", "cuped", "lin", "bayes_flat"]
SCENARIOS = [
    # name, group, label, maker, effect under the alternative, methods
    ("normal", "baseline", "normal, 2,000 per group", student_t(1e9, 2000), 0.0885, FIXED + ["msprt"]),
    ("lognormal_s1", "skewed", "lognormal sd 1, 2,000 per group", ln1, lift1, SKEWED + ["msprt"]),
    ("lognormal_s2", "skewed", "lognormal sd 2, 2,000 per group", ln2, lift2, SKEWED + ["msprt"]),
    ("lognormal_s2_big", "skewed", "lognormal sd 2, 20,000 per group", ln2_big, lift2_big, SKEWED + ["msprt"]),
    ("lognormal_s2_uneven", "skewed", "lognormal sd 2, 1,000 vs 19,000", ln2_uneven, lift2_uneven, SKEWED),
    ("pareto_2.5", "heavy tails", "Pareto shape 2.5, 5,000 per group", pareto(2.5, 5000), 0.05,
     ["welch", "capped_99", "bayes_flat", "msprt"]),
    ("pareto_1.5", "heavy tails", "Pareto shape 1.5 (infinite variance), 5,000 per group", pareto(1.5, 5000), 0.10,
     ["welch", "capped_99", "bayes_flat", "msprt"]),
    ("t3", "heavy tails", "t with 3 df, 2,000 per group", student_t(3, 2000), 0.15, FIXED + ["msprt"]),
    ("zero_inflated", "zeros", "5% buyers, lognormal spend, 10,000 per group", zero_inflated(0.05, 1.0, 10000),
     0.12, SKEWED + ["msprt"]),
    ("binary_0.001", "rare events", "0.1% conversion, 20,000 per group", binary(0.001, 20000), 0.00089,
     FIXED + ["msprt"]),
    ("binary_0.003", "rare events", "0.3% conversion, 20,000 per group", binary(0.003, 20000), 0.00154,
     FIXED + ["msprt"]),
    ("binary_0.01", "rare events", "1% conversion, 20,000 per group", binary(0.01, 20000), 0.00279,
     FIXED + ["msprt"]),
    ("binary_0.001_big", "rare events", "0.1% conversion, 200,000 per group", binary(0.001, 200000), 0.00028,
     FIXED + ["msprt"]),
    ("sessions", "correlated", "2,000 users per group, about 5 sessions each", sess, 0.02,
     ["welch", "cluster", "msprt"]),
    ("clusters_50", "correlated", "50 clusters per group randomized whole, ICC 0.05", clustered(50, 40, 0.05), 0.15,
     ["welch", "cluster", "cluster_perm", "delta_normal"]),
    ("clusters_8", "correlated", "8 clusters per group randomized whole, ICC 0.05", clustered(8, 250, 0.05), 0.35,
     ["welch", "cluster", "cluster_perm", "delta_normal"]),
    ("missing_mcar", "missing", "20% missing completely at random", missing("mcar", 4000), 0.07, ["welch", "cuped"]),
    ("missing_mar", "missing", "missing more for low pre period activity", missing("mar", 4000), 0.07,
     ["welch", "cuped"]),
    ("missing_mnar", "missing", "low outcomes go missing, same rule both groups", missing("mnar", 4000), 0.07,
     ["welch", "cuped"]),
    ("missing_differential", "missing", "treatment makes low users drop out of logging", missing("diff", 4000),
     0.07, ["welch", "cuped"]),
    ("hte_even", "heterogeneous", "effect varies with a pre period trait, 50/50", hte(4000, 4000, 0.5), 0.06,
     ["welch", "cuped", "lin", "msprt"]),
    ("hte_uneven", "heterogeneous", "effect varies with a pre period trait, 10/90", hte(1000, 9000, 0.5), 0.1,
     ["welch", "cuped", "lin"]),
]


def rate(k, n):
    return {"rate": k / n, "mc_se": float(np.sqrt(max(k / n * (1 - k / n), 1 / n) / n))}


def run_scenario(name, maker, eff, methods, rng):
    out = {"effect": float(eff), "methods": {}}
    fixed = [m for m in methods if m != "msprt"]
    rec = {m: {"null": [], "alt": [], "time": 0.0, "calls": 0} for m in fixed}
    seq = {"null": [], "alt": [], "time": 0.0, "calls": 0}
    extra = {"srm_rows_null": [], "srm_clusters_null": [], "missing_check_alt": [], "skew_flag": []}
    true_alt = None
    for phase, sims, e in (("null", N_NULL, 0.0), ("alt", N_ALT, eff)):
        for _ in range(sims):
            d = maker(rng, e)
            if phase == "alt":
                true_alt = d["true"]
            tau = abs(true_alt) if true_alt else abs(eff)
            for m in fixed:
                t0 = time.perf_counter()
                res = METHODS[m](d, tau)
                rec[m]["time"] += time.perf_counter() - t0
                rec[m]["calls"] += 1
                rec[m][phase].append(res)
            if "msprt" in methods:
                t0 = time.perf_counter()
                seq[phase].append(run_msprt(d, tau))
                seq["time"] += time.perf_counter() - t0
                seq["calls"] += 1
            if phase == "null" and "welch" in methods:
                extra["skew_flag"].append(abs(ab.core._diff_skewness(d["c"], d["t"])) > ab.core.SKEW_LIMIT)
            if "gc" in d and phase == "null":
                extra["srm_rows_null"].append(ab.srm_test(len(d["c"]), len(d["t"])) < 0.01)
                extra["srm_clusters_null"].append(ab.srm_test(len(np.unique(d["gc"])), len(np.unique(d["gt"]))) < 0.01)
            if "miss_c" in d and phase == "alt":
                extra["missing_check_alt"].append(ab.mean_test(d["miss_c"], d["miss_t"]).p_value < 0.01)
    for m in fixed:
        null = np.array(rec[m]["null"], dtype=float)
        alt = np.array(rec[m]["alt"], dtype=float)
        n0, n1 = len(null), len(alt)
        est = alt[:, 0]
        sd = est.std(ddof=1)
        out["methods"][m] = {
            "type1": rate(int((null[:, 3] < 0.05).sum()), n0),
            # which side the false positives land on; each should be about 2.5%
            "type1_low": rate(int((null[:, 2] < 0).sum()), n0),
            "type1_high": rate(int((null[:, 1] > 0).sum()), n0),
            "coverage": rate(int(((alt[:, 1] <= true_alt) & (true_alt <= alt[:, 2])).sum()), n1),
            "power": rate(int(((alt[:, 3] < 0.05) & (np.sign(est) == np.sign(true_alt))).sum()), n1),
            "bias_pct_of_effect": float((est.mean() - true_alt) / true_alt * 100),
            "bias_mc_se_pct": float(sd / np.sqrt(n1) / abs(true_alt) * 100),
            "rmse": float(np.sqrt(((est - true_alt) ** 2).mean())),
            "ms_per_call": rec[m]["time"] / rec[m]["calls"] * 1000,
        }
    if "msprt" in methods:
        null, alt = seq["null"], seq["alt"]
        shares = [s for _, _, s, _ in alt if not np.isnan(s)]
        out["methods"]["msprt"] = {
            "type1": rate(sum(s for s, _, _, _ in null), len(null)),
            "power": rate(sum(s for s, _, _, _ in alt), len(alt)),
            "cs_coverage": rate(sum(not mi for _, mi, _, _ in alt), len(alt)),
            "avg_share_when_stopped": float(np.mean(shares)) if shares else None,
            "empty_interval_rate": (sum(e for *_, e in null) + sum(e for *_, e in alt)) / (len(null) + len(alt)),
            "ms_per_call": seq["time"] / seq["calls"] * 1000,
        }
    if extra["srm_rows_null"]:
        out["srm_false_alarm_counting_rows"] = rate(int(np.sum(extra["srm_rows_null"])), N_NULL)
        out["srm_false_alarm_counting_clusters"] = rate(int(np.sum(extra["srm_clusters_null"])), N_NULL)
    if extra["skew_flag"]:
        out["skew_note_rate_null"] = float(np.mean(extra["skew_flag"]))
    if extra["missing_check_alt"]:
        out["missing_check_fires"] = rate(int(np.sum(extra["missing_check_alt"])), N_ALT)
    out["true_effect"] = float(true_alt)
    return out


# ------------------------------------------- skewness rule of thumb check
# Kohavi et al. (2014): the normal approximation is fine once each group has
# more than 355 * skewness^2 users. How does the worst tail error move with n?
def skew_rule_curve(rng):
    sigma = 1.5
    skew = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)
    rows = []
    sims = 400 if QUICK else 4000
    for n in [200, 1000, 5000, 20000]:
        for split in ("even", "uneven"):
            nc, nt = (n, n) if split == "even" else (n, 9 * n)
            low = high = 0
            for _ in range(sims):
                r = ab.mean_test(np.exp(sigma * rng.normal(0, 1, nc)), np.exp(sigma * rng.normal(0, 1, nt)))
                low += r.ci[1] < 0
                high += r.ci[0] > 0
            rows.append({"n_control": nc, "n_treatment": nt, "split": split,
                         "rule_needs": float(355 * skew**2), "type1_low": rate(low, sims),
                         "type1_high": rate(high, sims)})
    return {"sigma": sigma, "skewness": float(skew), "rows": rows}


# ------------------------------------------------------------- timing
def timing(rng):
    rows = []
    for n in [10_000, 100_000, 1_000_000]:
        c, t = rng.lognormal(0, 1, n), rng.lognormal(0.01, 1, n)
        x = c * 0.5 + rng.lognormal(0, 1, n)
        xt = t * 0.5 + rng.lognormal(0, 1, n)
        g = rng.integers(0, n // 5, n)
        jobs = {
            "mean_test": lambda: ab.mean_test(c, t),
            "cuped": lambda: ab.cuped_adjust(np.concatenate([c, t]), np.concatenate([x, xt])),
            "ratio_test": lambda: ab.ratio_test(c, x, t, xt),
            "cluster_test": lambda: ab.cluster_test(c, t, g, g),
            "msprt_50_looks": lambda: ab.msprt(c, t, tau=0.01),
            "bayes_200k_draws": lambda: ab.bayes_test(c, t),
        }
        for name, f in jobs.items():
            reps = 3 if n >= 1_000_000 else 7
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
            rows.append({"n_per_group": n, "method": name, "ms": float(np.median(ts) * 1000)})
    return rows


LABELS = {"welch": "t test (Welch)", "capped_99": "t test, capped at p99", "cuped": "CUPED", "lin": "Lin adjustment",
          "bayes_flat": "Bayes, flat prior", "cluster": "cluster_test", "cluster_perm": "cluster_test, permutations",
          "delta_normal": "delta method, normal", "msprt": "mSPRT, 50 looks"}


def plot(out):
    sc = out["scenarios"]
    names = list(sc)
    fig, ax = plt.subplots(1, 2, figsize=(10, 6.6), sharey=True)
    y = np.arange(len(names))[::-1]
    marks = (("welch", RED, "o"), ("cuped", GREY, "^"), ("cluster", BLUE, "s"), ("cluster_perm", BLUE, "D"),
             ("msprt", "#333", "x"))
    for yi, name in zip(y, names):
        v = sc[name]["methods"]
        for m, col, mk in marks:
            if m not in v:
                continue
            r = v[m]["type1"]
            ax[0].errorbar(min(r["rate"], 0.3) * 100, yi, xerr=2 * r["mc_se"] * 100, fmt=mk, color=col, ms=5, lw=1)
            if m != "msprt":
                c = v[m]["coverage"]
                ax[1].errorbar(max(c["rate"], 0.3) * 100, yi, xerr=2 * c["mc_se"] * 100, fmt=mk, color=col, ms=5,
                               lw=1)
    ax[0].axvline(5, color="k", lw=0.8, ls="--"); ax[1].axvline(95, color="k", lw=0.8, ls="--")
    ax[0].set_yticks(y, [sc[n]["label"] for n in names], fontsize=7)
    ax[0].set_xlabel("false positive rate, % (capped at 30)"); ax[1].set_xlabel("95% interval coverage, % (floor 30)")
    ax[0].set_title("No real effect: how often it says there is one"); ax[1].set_title("Real effect: does the interval hold?")
    for m, col, mk in marks:
        ax[0].plot([], [], mk, color=col, label=LABELS[m])
    ax[0].legend(frameon=False, fontsize=7, loc="upper right", title="bars: 2 Monte Carlo SE", title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "stress_overview.png"); plt.close(fig)

    rows = out["skew_rule"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for split, col in (("even", BLUE), ("uneven", RED)):
        rr = [r for r in rows if r["split"] == split]
        n = [r["n_control"] for r in rr]
        for side, ls in (("type1_high", "-"), ("type1_low", "--")):
            ax.errorbar(n, [r[side]["rate"] * 100 for r in rr], yerr=[2 * r[side]["mc_se"] * 100 for r in rr],
                        color=col, ls=ls, marker="o", ms=4, lw=1.6,
                        label=f"{'same size groups' if split == 'even' else 'treatment 9x bigger'}, "
                              f"{'upper' if side == 'type1_high' else 'lower'} tail")
    ax.axhline(2.5, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks([200, 1000, 5000, 20000], ["200", "1,000", "5,000", "20,000"])
    ax.minorticks_off(); ax.set_xlabel("users in the smaller group"); ax.set_ylabel("one sided error, % (should be 2.5)")
    ax.set_title("Skewed revenue only hurts when the groups are uneven")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(FIG / "stress_skew.png"); plt.close(fig)


if __name__ == "__main__" and "--plot" in sys.argv:
    plot(json.loads((RES / "stress.json").read_text()))
    sys.exit()

if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(2027)
    out = {"sims_null": N_NULL, "sims_alt": N_ALT, "looks": LOOKS, "scenarios": {}}
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    for name, group, label, maker, eff, methods in SCENARIOS:
        if only and name not in only:
            continue
        t0 = time.time()
        res = run_scenario(name, maker, eff, methods, rng)
        res.update({"group": group, "label": label})
        out["scenarios"][name] = res
        w = res["methods"].get("welch", {})
        print(f"{name:<22} {time.time() - t0:5.1f}s  welch type1 {w.get('type1', {}).get('rate', float('nan')):.3f} "
              f"coverage {w.get('coverage', {}).get('rate', float('nan')):.3f}", flush=True)
    if not only:
        out["skew_rule"] = skew_rule_curve(rng)
        out["timing"] = timing(rng)
    out["runtime_minutes"] = (time.time() - t_start) / 60
    if not (QUICK or only):
        (RES / "stress.json").write_text(json.dumps(out, indent=2))
        plot(out)
    else:
        print(json.dumps(out, indent=1)[:6000])
