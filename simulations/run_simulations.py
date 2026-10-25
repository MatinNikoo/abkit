"""
Checks that every method in abkit does what it claims, using experiments
where I know the true answer because I made the data up.

Run:  python simulations/run_simulations.py
Writes figures/ and results/simulations.json. Takes a few minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402
from abkit.sequential import naive_peeking  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
out: dict = {}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, GREY, RED = "#2F4BC4", "#9AA3B2", "#C2413B"

# ---------------------------------------------------------------- 1. peeking
# A/A tests: no real difference. How often does each method say "significant"
# if you check after every look and stop at the first hit?
N_SIMS, N, LOOKS = 2000, 10000, 100
looks_axis = None
naive_first = np.full(N_SIMS, np.inf)
seq_first = np.full(N_SIMS, np.inf)
for s in range(N_SIMS):
    a, b = rng.normal(0, 1, N), rng.normal(0, 1, N)
    pn = naive_peeking(a, b, n_looks=LOOKS, min_per_arm=100)
    r = ab.msprt(a, b, tau=0.05, n_looks=LOOKS, min_per_arm=100)
    hn, hs = np.flatnonzero(pn < 0.05), np.flatnonzero(r.p_value < 0.05)
    if len(hn): naive_first[s] = hn[0]
    if len(hs): seq_first[s] = hs[0]
    if looks_axis is None:
        looks_axis = np.arange(1, len(pn) + 1)
k = len(looks_axis)
naive_curve = np.array([(naive_first < i).mean() for i in range(1, k + 1)])
seq_curve = np.array([(seq_first < i).mean() for i in range(1, k + 1)])
out["peeking"] = {"sims": N_SIMS, "looks": int(k),
                  "naive_false_positive_after_all_looks": float(naive_curve[-1]),
                  "naive_false_positive_after_10_looks": float(naive_curve[9]),
                  "msprt_false_positive_after_all_looks": float(seq_curve[-1])}

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(looks_axis, naive_curve, color=RED, lw=2, label="normal p value, checked every look")
ax.plot(looks_axis, seq_curve, color=BLUE, lw=2, label="mSPRT (always valid)")
ax.axhline(0.05, color="k", lw=0.8, ls="--")
ax.text(k, 0.055, "promised 5%", ha="right", va="bottom", fontsize=8)
ax.set_xlabel("number of times you've checked"); ax.set_ylabel("false positive rate")
ax.set_title("A/A tests: peeking at a normal p value breaks it")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "peeking.png"); plt.close(fig)

# ------------------------------------------------- 2. sequential power, speed
# With a real effect, how often does each approach find it, and how early?
effects = [0.0, 0.02, 0.04, 0.06, 0.08]
N2, SIMS2 = 10000, 500
seq_power, fixed_power, avg_stop = [], [], []
for eff in effects:
    hits_s = hits_f = 0
    stops = []
    for _ in range(SIMS2):
        a, b = rng.normal(0, 1, N2), rng.normal(eff, 1, N2)
        r = ab.msprt(a, b, tau=0.05, n_looks=50)
        if r.stopped:
            hits_s += 1
            stops.append((r.n_control[r.stop_index] + r.n_treatment[r.stop_index]) / (2 * N2))
        hits_f += ab.mean_test(a, b).p_value < 0.05
    seq_power.append(hits_s / SIMS2); fixed_power.append(hits_f / SIMS2)
    avg_stop.append(float(np.mean(stops)) if stops else float("nan"))
out["sequential_vs_fixed"] = {"effects_sd": effects, "msprt_power": seq_power,
                              "fixed_power": fixed_power, "msprt_avg_share_of_sample_when_stopped": avg_stop}

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(effects, fixed_power, "o-", color=GREY, lw=2, label="fixed horizon, one look at the end")
ax.plot(effects, seq_power, "o-", color=BLUE, lw=2, label="mSPRT, 50 looks")
for e, p, s in zip(effects[2:], seq_power[2:], avg_stop[2:]):
    ax.annotate(f"stops on average at\n{s:.0%} of the sample", (e, p), textcoords="offset points", xytext=(6, -22),
                ha="left", fontsize=7, color=BLUE)
ax.set_xlim(-0.003, 0.095)
ax.set_xlabel("true effect (in standard deviations)"); ax.set_ylabel("chance of detecting it")
ax.set_title("What always valid costs: a little power, often an early stop")
ax.legend(frameon=False, fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(FIG / "sequential_power.png"); plt.close(fig)

# ------------------------------------------------------------------ 3. CUPED
rhos = [0.0, 0.3, 0.5, 0.7, 0.9]
N3, SIMS3, EFF = 4000, 800, 0.05
emp, coverage_plain, coverage_cuped, power_plain, power_cuped = [], [], [], [], []
for rho in rhos:
    se_plain, se_cuped = [], []
    cov_p = cov_c = pw_p = pw_c = 0
    for _ in range(SIMS3):
        x = rng.normal(0, 1, 2 * N3)
        y = rho * x + np.sqrt(1 - rho**2) * rng.normal(0, 1, 2 * N3)
        y[N3:] += EFF
        y_adj, _ = ab.cuped_adjust(y, x)
        rp = ab.mean_test(y[:N3], y[N3:])
        rc = ab.mean_test(y_adj[:N3], y_adj[N3:])
        se_plain.append(rp.se); se_cuped.append(rc.se)
        cov_p += rp.ci[0] <= EFF <= rp.ci[1]; cov_c += rc.ci[0] <= EFF <= rc.ci[1]
        pw_p += rp.p_value < 0.05; pw_c += rc.p_value < 0.05
    emp.append(1 - (np.mean(se_cuped) / np.mean(se_plain)) ** 2)
    coverage_plain.append(cov_p / SIMS3); coverage_cuped.append(cov_c / SIMS3)
    power_plain.append(pw_p / SIMS3); power_cuped.append(pw_c / SIMS3)
out["cuped"] = {"rho": rhos, "variance_cut_measured": emp, "variance_cut_theory": [r * r for r in rhos],
                "ci_coverage_plain": coverage_plain, "ci_coverage_cuped": coverage_cuped,
                "power_plain": power_plain, "power_cuped": power_cuped}

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
rr = np.linspace(0, 0.95, 50)
ax[0].plot(rr, rr**2, color=GREY, lw=2, label="theory, rho squared")
ax[0].plot(rhos, emp, "o", color=BLUE, label="measured")
ax[0].set_xlabel("correlation with pre period metric"); ax[0].set_ylabel("variance removed")
ax[0].set_title("CUPED removes rho squared of the noise"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(rhos, power_plain, "o-", color=GREY, lw=2, label="plain")
ax[1].plot(rhos, power_cuped, "o-", color=BLUE, lw=2, label="CUPED")
ax[1].set_xlabel("correlation with pre period metric"); ax[1].set_ylabel("power")
ax[1].set_title(f"Same users, more power (effect {EFF} sd)"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "cuped.png"); plt.close(fig)

# --------------------------------------------- 4. ratio metric, naive vs delta
# Clicks per session where users have different numbers of sessions.
# Treating sessions as independent gives intervals that are too narrow.
SIMS4, N4 = 1000, 2000
cov_delta = cov_naive = 0
for _ in range(SIMS4):
    def sim(n):
        rate = rng.beta(2, 5, n)                     # each user has their own click rate
        s = rng.poisson(6, n) + 1
        return rng.binomial(s, rate), s
    nc, dc = sim(N4); nt, dt = sim(N4)
    r = ab.ratio_test(nc, dc, nt, dt)
    cov_delta += r.ci[0] <= 0 <= r.ci[1]
    pc, pt = nc.sum() / dc.sum(), nt.sum() / dt.sum()
    se_naive = np.sqrt(pc * (1 - pc) / dc.sum() + pt * (1 - pt) / dt.sum())
    cov_naive += abs(pt - pc) <= 1.96 * se_naive
out["ratio_metric"] = {"coverage_delta_method": cov_delta / SIMS4, "coverage_naive_sessions": cov_naive / SIMS4}

# ---------------------------------------------------- 5. sample size formula
targets = [0.02, 0.05, 0.1]
achieved = []
for mde in targets:
    n, _ = ab.sample_size(sd=1, mde=mde, power=0.8)
    hits = sum(ab.mean_test(rng.normal(0, 1, n), rng.normal(mde, 1, n)).p_value < 0.05 for _ in range(400))
    achieved.append(hits / 400)
out["sample_size"] = {"mde_sd": targets, "promised_power": 0.8, "achieved_power": achieved}

# ------------------------------------------------- 6. many metrics at once
SIMS6, M = 2000, 20
any_raw = any_holm = 0
fdr_bh = []
for _ in range(SIMS6):
    # 15 metrics with no effect, 5 with a real one
    z = rng.normal(0, 1, M); z[:5] += 3
    p = 2 * (1 - __import__("scipy").stats.norm.cdf(np.abs(z)))
    null = np.arange(M) >= 5
    any_raw += (p[null] < 0.05).any()
    any_holm += (ab.holm(p)[null] < 0.05).any()
    wins = ab.benjamini_hochberg(p) < 0.05
    fdr_bh.append((wins & null).sum() / max(wins.sum(), 1))
out["multiple_testing"] = {"metrics": M, "null_metrics": 15,
                           "chance_any_false_win_raw": any_raw / SIMS6,
                           "chance_any_false_win_holm": any_holm / SIMS6,
                           "false_discovery_rate_bh": float(np.mean(fdr_bh)),
                           "false_discovery_rate_bh_mc_se": float(np.std(fdr_bh, ddof=1) / np.sqrt(SIMS6))}


# ------------------------------------------- 7. CUPED with several covariates
# Two pre period metrics, each partly predictive. One covariate vs both.
SIMS7, N7 = 600, 4000
se_none, se_one, se_two = [], [], []
cov_two = 0
for _ in range(SIMS7):
    X = rng.normal(0, 1, (2 * N7, 2))
    y = 0.5 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, np.sqrt(0.5), 2 * N7)
    y[N7:] += 0.05
    r0 = ab.mean_test(y[:N7], y[N7:])
    y1, _ = ab.cuped_adjust(y, X[:, 0])
    y2, _ = ab.cuped_adjust(y, X)
    r1, r2 = ab.mean_test(y1[:N7], y1[N7:]), ab.mean_test(y2[:N7], y2[N7:])
    se_none.append(r0.se); se_one.append(r1.se); se_two.append(r2.se)
    cov_two += r2.ci[0] <= 0.05 <= r2.ci[1]
out["cuped_multi"] = {"variance_cut_one_covariate": 1 - (np.mean(se_one) / np.mean(se_none)) ** 2,
                      "variance_cut_two_covariates": 1 - (np.mean(se_two) / np.mean(se_none)) ** 2,
                      "theory_one": 0.25, "theory_two": 0.5, "ci_coverage_two": cov_two / SIMS7}

# ----------------------------------------------------- 8. slicing by segment
# The treatment does nothing in any of 10 segments (say, countries).
# How often does at least one segment still look like a win or a loss?
SIMS8, K, NS = 1000, 10, 2000
any_raw = any_bh = het_hits = 0
for _ in range(SIMS8):
    est, ses, ps = [], [], []
    for _k in range(K):
        r = ab.mean_test(rng.normal(0, 1, NS), rng.normal(0, 1, NS))
        est.append(r.lift); ses.append(r.se); ps.append(r.p_value)
    ps = np.array(ps)
    any_raw += (ps < 0.05).any()
    any_bh += (ab.benjamini_hochberg(ps) < 0.05).any()
    het_hits += ab.heterogeneity_test(est, ses) < 0.05
out["segments"] = {"segments": K,
                   "chance_some_segment_significant_raw": any_raw / SIMS8,
                   "chance_some_segment_significant_bh": any_bh / SIMS8,
                   "heterogeneity_test_false_alarm_rate": het_hits / SIMS8}


# ------------------------------------------------ 9. Bayesian calibration
# When it says "85% chance B is better", is B better 85% of the time?
# Real product effects are usually tiny compared to the noise, so true
# lifts here are drawn with a spread about equal to one standard error.
SIMS9 = 3000
flat, prior, truth = [], [], []
for _ in range(SIMS9):
    true = rng.normal(0, 0.03)
    a, b = rng.normal(0, 1, 2000), rng.normal(true, 1, 2000)
    flat.append(ab.bayes_test(a, b, draws=8000).chance_treatment_better)
    prior.append(ab.bayes_test(a, b, draws=8000, prior_sd=0.03).chance_treatment_better)
    truth.append(true > 0)
flat, prior, truth = map(np.array, (flat, prior, truth))
bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.0001])
def calib(said):
    # fold so every call is about the side it favors
    conf = np.where(said >= 0.5, said, 1 - said)
    hit = np.where(said >= 0.5, truth, ~truth)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 30:
            rows.append((float(conf[m].mean()), float(hit[m].mean()), int(m.sum())))
    return rows
cf, cp = calib(flat), calib(prior)
out["bayes_calibration"] = {"flat_prior": cf, "prior_from_past_tests": cp,
                            "note": "each row is (stated chance, how often it was right, count)"}
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0.5, 1], [0.5, 1], color="k", lw=0.8, ls="--")
ax.plot([r[0] for r in cf], [r[1] for r in cf], "o-", color=RED, lw=2, label="flat prior")
ax.plot([r[0] for r in cp], [r[1] for r in cp], "o-", color=BLUE, lw=2, label="prior from past tests")
ax.set_xlabel("stated chance the favored version is better"); ax.set_ylabel("how often it actually was")
ax.set_title("Is \"chance to win\" honest?"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "bayes_calibration.png"); plt.close(fig)

(RES / "simulations.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
