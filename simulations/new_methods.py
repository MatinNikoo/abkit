"""
Simulation checks for the methods added in 0.2: group sequential designs
(and how they trade off against the mSPRT), CUPAC, the ratio metric power
calculator, Dunnett, non-inferiority, the bootstrap and rank tools, the
time checks, effect screening and the A/A simulator.

Same idea as run_simulations.py: made up experiments where I know the
truth, enough of them that the Monte Carlo error is small next to what
I'm measuring.

Run:  python simulations/new_methods.py            (roughly 30 minutes on one core)
      python simulations/new_methods.py --resume   (continue a run that was stopped)
      python simulations/new_methods.py --only cupac,dunnett
      python simulations/new_methods.py --quick    (smoke test)
Writes results/new_methods.json and figures/gsd_tradeoff.png, figures/cupac.png.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402

QUICK = "--quick" in sys.argv
S = 0.1 if QUICK else 1.0          # scales every simulation count
FIG, RES = ROOT / "figures", ROOT / "results"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, GREY, RED, GREEN = "#2F4BC4", "#9AA3B2", "#C2413B", "#3A8F5C"
rng = np.random.default_rng(31)
out: dict = {}
START = time.time()


def n_(k):
    return max(20, int(k * S))


def rate(k, n):
    return {"rate": k / n, "mc_se": float(np.sqrt(max(k / n * (1 - k / n), 1 / n) / n))}


def section(name):
    print(f"{name} ...", flush=True)
    return time.time()


# each section has its own seed and is saved as soon as it finishes, so a run
# that gets stopped can pick up where it left off with --resume
CHECKPOINT = ROOT / "data" / "new_methods_partial.json"
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
if "--resume" in sys.argv and CHECKPOINT.exists() and not QUICK:
    out.update(json.loads(CHECKPOINT.read_text()))


def want(name):
    if ONLY and name not in ONLY:
        return False
    if name in out:
        print(f"{name}: already done, skipping", flush=True)
        return False
    return True


def save(name):
    if not QUICK:
        CHECKPOINT.parent.mkdir(exist_ok=True)
        CHECKPOINT.write_text(json.dumps(out, indent=1, default=float))


# ------------------------------------------- group sequential vs mSPRT
if want("group_sequential"):
    rng = np.random.default_rng([31, 1])
    t0 = section("group sequential")
    N = 10000
    effects = [0.0, 0.02, 0.03, 0.04, 0.05]
    designs = {"obf_5": ("obrien_fleming", 5), "pocock_5": ("pocock", 5)}
    res = {k: {"power": [], "avg_share": [], "rci_coverage": []} for k in ["fixed", "msprt_50", *designs]}
    for eff in effects:
        sims = n_(4000 if eff == 0 else 1000)
        hits = {k: 0 for k in res}
        share = {k: [] for k in res}
        cover = {k: 0 for k in designs}
        for _ in range(sims):
            a, b = rng.normal(0, 1, N), rng.normal(eff, 1, N)
            hits["fixed"] += ab.mean_test(a, b).p_value < 0.05
            share["fixed"].append(1.0)
            r = ab.msprt(a, b, tau=0.03, n_looks=50)
            hits["msprt_50"] += r.stopped
            share["msprt_50"].append((r.n_control[r.stop_index] + r.n_treatment[r.stop_index]) / (2 * N)
                                     if r.stopped else 1.0)
            for k, (kind, looks) in designs.items():
                g = ab.group_sequential_test(a, b, looks, kind=kind)
                hits[k] += g.stopped
                share[k].append(g.design.info[g.stop_index] if g.stopped else 1.0)
                cover[k] += bool(((g.ci_low <= eff) & (eff <= g.ci_high)).all())
        for k in res:
            res[k]["power"].append(rate(int(hits[k]), sims))
            res[k]["avg_share"].append(float(np.mean(share[k])))
            if k in designs:
                res[k]["rci_coverage"].append(rate(cover[k], sims))
    # numerical integration should agree with the simulation
    theory = {}
    for k, (kind, looks) in designs.items():
        d = ab.group_sequential_design(looks, kind=kind)
        drift = [e / np.sqrt(2 / N) for e in effects]
        theory[k] = {"power": [d.power(x) for x in drift], "avg_share": [d.expected_sample(x) for x in drift],
                     "bounds": d.bounds.tolist(), "inflation_80": d.inflation(0.8)}
    out["group_sequential"] = {"n_per_group": N, "effects_sd": effects, "simulated": res, "theory": theory,
                               "msprt_tau": 0.03, "seconds": time.time() - t0}

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    lab = {"fixed": "fixed, one look", "msprt_50": "mSPRT, 50 looks", "obf_5": "O'Brien Fleming, 5 looks",
           "pocock_5": "Pocock, 5 looks"}
    col = {"fixed": GREY, "msprt_50": BLUE, "obf_5": GREEN, "pocock_5": RED}
    for k in res:
        ax[0].plot(effects, [p["rate"] for p in res[k]["power"]], "o-", color=col[k], lw=2, label=lab[k])
        ax[1].plot(effects, res[k]["avg_share"], "o-", color=col[k], lw=2, label=lab[k])
    ax[0].set_xlabel("true effect (sd)"); ax[0].set_ylabel("chance of a significant result")
    ax[0].set_title("Power"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_xlabel("true effect (sd)"); ax[1].set_ylabel("share of users used, on average")
    ax[1].set_title("How early each one stops")
    fig.tight_layout(); fig.savefig(FIG / "gsd_tradeoff.png"); plt.close(fig)
    save("group_sequential")


# ---------------------------------------------------------------- CUPAC
if want("cupac"):
    rng = np.random.default_rng([31, 2])
    t0 = section("cupac")
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
    from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

    N2, P = 1500, 5

    def make_users(n, eff):
        X = rng.normal(0, 1, (2 * n, P))
        signal = 1.2 * np.sin(2 * X[:, 0]) + 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.abs(X[:, 3]) + 0.4 * X[:, 4]
        y = signal + rng.normal(0, 1, 2 * n)
        y[n:] += eff
        return X, y

    cupac_methods = ["plain", "cuped_linear", "cupac_crossfit", "cupac_in_sample_gb", "cupac_in_sample_knn"]
    rows = {m: {"null": [], "alt": []} for m in cupac_methods}
    EFF2 = 0.08
    gb = HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, random_state=0)
    for phase, sims, eff in (("null", n_(300), 0.0), ("alt", n_(300), EFF2)):
        for i in range(sims):
            X, y = make_users(N2, eff)
            res_ = {"plain": ab.mean_test(y[:N2], y[N2:])}
            ya, _ = ab.cuped_adjust(y, X)
            res_["cuped_linear"] = ab.mean_test(ya[:N2], ya[N2:])
            ya, _, _ = ab.cupac_adjust(y, X, model=gb, folds=2, seed=i)
            res_["cupac_crossfit"] = ab.mean_test(ya[:N2], ya[N2:])
            # the mistake: fit the model and predict the same rows it learned from.
            # boosting overfits a little, 5 nearest neighbors (each user is one of
            # their own 5 neighbors) a lot
            for name, model in (("gb", gb), ("knn", KNeighborsRegressor(n_neighbors=5))):
                pred = model.fit(X, y).predict(X)
                ya, _ = ab.cuped_adjust(y, pred)
                res_[f"cupac_in_sample_{name}"] = ab.mean_test(ya[:N2], ya[N2:])
            for m, r in res_.items():
                rows[m][phase].append((r.lift, r.ci[0], r.ci[1], r.p_value, r.se))
    cupac = {}
    for m in cupac_methods:
        null, alt = np.array(rows[m]["null"]), np.array(rows[m]["alt"])
        cupac[m] = {"type1": rate(int((null[:, 3] < 0.05).sum()), len(null)),
                    "coverage": rate(int(((alt[:, 1] <= EFF2) & (EFF2 <= alt[:, 2])).sum()), len(alt)),
                    "power": rate(int((alt[:, 3] < 0.05).sum()), len(alt)),
                    "bias_pct_of_effect": float((alt[:, 0].mean() - EFF2) / EFF2 * 100),
                    "bias_mc_se_pct": float(alt[:, 0].std(ddof=1) / np.sqrt(len(alt)) / EFF2 * 100),
                    "avg_se": float(alt[:, 4].mean())}
    base = cupac["plain"]["avg_se"]
    for m in cupac_methods:
        cupac[m]["variance_cut"] = 1 - (cupac[m]["avg_se"] / base) ** 2
    out["cupac"] = {"n_per_group": N2, "features": P, "effect": EFF2, "methods": cupac, "seconds": time.time() - t0}

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    names = {"plain": "no adjustment", "cuped_linear": "CUPED, 5 linear\ncovariates",
             "cupac_crossfit": "CUPAC,\ncross fit", "cupac_in_sample_gb": "boosting,\nno cross fit",
             "cupac_in_sample_knn": "5 neighbors,\nno cross fit"}
    xs = np.arange(len(cupac_methods))
    ax[0].bar(xs, [cupac[m]["variance_cut"] for m in cupac_methods], color=[GREY, BLUE, GREEN, RED, RED])
    ax[0].set_xticks(xs, [names[m] for m in cupac_methods], fontsize=7); ax[0].set_ylabel("variance removed")
    ax[0].set_title("Noise removed")
    cv = [cupac[m]["coverage"]["rate"] for m in cupac_methods]
    ax[1].bar(xs, cv, color=[GREY, BLUE, GREEN, RED, RED])
    ax[1].errorbar(xs, cv, yerr=[2 * cupac[m]["coverage"]["mc_se"] for m in cupac_methods], fmt="none", color="k", lw=1)
    ax[1].axhline(0.95, color="k", lw=0.8, ls="--")
    ax[1].set_xticks(xs, [names[m] for m in cupac_methods], fontsize=7); ax[1].set_ylabel("95% interval coverage")
    ax[1].set_ylim(0, 1); ax[1].set_title("Does the interval still hold?")
    fig.tight_layout(); fig.savefig(FIG / "cupac.png"); plt.close(fig)
    save("cupac")


# ------------------------------------------------- ratio metric power
if want("ratio_power"):
    rng = np.random.default_rng([31, 3])
    t0 = section("ratio power")

    def users(n, shift=0.0):
        rate_ = np.clip(rng.beta(2, 8, n) + shift, 0, 1)
        s = rng.poisson(4, n) + 1
        return rng.binomial(s, rate_).astype(float), s.astype(float)

    pc, ps = users(20000)
    MDE3 = 0.01
    n_right, _ = ab.ratio_sample_size(pc, ps, MDE3)
    p0 = pc.sum() / ps.sum()
    # the usual mistake: the per session sd, then divide by sessions per user
    n_naive = int(np.ceil(ab.proportion_sample_size(p0, MDE3)[0] / ps.mean()))
    ach = {}
    for name, n in (("delta_method_n", n_right), ("naive_session_n", n_naive)):
        sims = n_(1000)
        hits = 0
        for _ in range(sims):
            nc, dc = users(n); nt, dt = users(n, MDE3)
            hits += ab.ratio_test(nc, dc, nt, dt).p_value < 0.05
        ach[name] = {"users_per_group": int(n), "achieved_power": rate(hits, sims)}
    # what the delta method n should deliver in this simulation exactly: clipping makes the true
    # lift a bit under 0.01 and the treatment group a bit noisier than the pilot
    bc, bd = users(2_000_000)
    tc_, td = users(2_000_000, MDE3)
    true_lift = tc_.sum() / td.sum() - bc.sum() / bd.sum()
    sd_c, sd_t = ab.ratio_sd(bc, bd), ab.ratio_sd(tc_, td)
    for name in ach:
        n = ach[name]["users_per_group"]
        se = np.sqrt((sd_c**2 + sd_t**2) / n)
        ach[name]["expected_power_given_true_lift"] = float(__import__("scipy").stats.norm.cdf(true_lift / se - 1.96))
    out["ratio_power"] = {"mde": MDE3, "baseline_rate": float(p0), "promised_power": 0.8, **ach,
                          "true_lift": float(true_lift), "sd_control": sd_c, "sd_treatment": sd_t,
                          "seconds": time.time() - t0}
    save("ratio_power")


# ------------------------------------------------------- A/A simulator
if want("aa"):
    rng = np.random.default_rng([31, 4])
    t0 = section("aa")
    y_norm = rng.normal(0, 1, 20000)
    k = rng.poisson(4, 4000) + 1
    g = np.repeat(np.arange(4000), k)
    y_sess = (rng.random(k.sum()) < rng.beta(2, 8, 4000)[g]).astype(float)
    out["aa"] = {
        "normal_rows": ab.aa_test(y_norm, n_(2000), seed=1).__dict__,
        "sessions_users_split_rows_tested": ab.aa_test(y_sess, n_(1000), clusters=g, test=ab.mean_test,
                                                       seed=2).__dict__,
        "sessions_users_split_cluster_test": ab.aa_test(y_sess, n_(1000), clusters=g, seed=3).__dict__,
    }
    for v in out["aa"].values():
        v.pop("p_values")
    save("aa")


# --------------------------------------------------------- Dunnett
if want("dunnett"):
    rng = np.random.default_rng([31, 5])
    t0 = section("dunnett")
    from scipy import stats as _st
    sims = n_(4000)
    any_raw = any_bonf = any_dun = any_scipy = 0
    pw_dun = pw_bonf = 0
    for i in range(sims):
        c = rng.normal(0, 1, 3000)
        ts = [rng.normal(0, 1, 3000) for _ in range(3)]
        d = ab.dunnett(c, ts)
        any_raw += (d["p"] < 0.05).any()
        any_bonf += (np.minimum(1, d["p"] * 3) < 0.05).any()
        any_dun += (d["p_dunnett"] < 0.05).any()
        # scipy's exact (pooled variance, t) Dunnett on the same data, as a reference
        any_scipy += (_st.dunnett(*ts, control=c).pvalue < 0.05).any()
        ts[0] = ts[0] + 0.07
        d = ab.dunnett(c, ts)
        pw_dun += d["p_dunnett"].iloc[0] < 0.05
        pw_bonf += min(1, d["p"].iloc[0] * 3) < 0.05
    out["dunnett"] = {"arms": 3, "n_per_group": 3000,
                      "any_false_win_raw": rate(any_raw, sims), "any_false_win_bonferroni": rate(any_bonf, sims),
                      "any_false_win_dunnett": rate(any_dun, sims),
                      "any_false_win_scipy_dunnett_same_data": rate(any_scipy, sims),
                      "power_dunnett": rate(pw_dun, sims), "power_bonferroni": rate(pw_bonf, sims)}
    save("dunnett")


# --------------------------------------------------- non-inferiority
if want("noninferiority"):
    rng = np.random.default_rng([31, 6])
    t0 = section("non-inferiority")
    sims = n_(4000)
    at_margin = at_zero = plain_says_fine = 0
    for _ in range(sims):
        c = rng.normal(10, 2, 2000)
        t_bad = rng.normal(10 - 0.15, 2, 2000)    # exactly at the margin
        t_ok = rng.normal(10, 2, 2000)
        at_margin += ab.noninferiority_test(c, t_bad, 0.15).p_value < 0.05
        at_zero += ab.noninferiority_test(c, t_ok, 0.15).p_value < 0.05
        plain_says_fine += ab.mean_test(c, t_bad).p_value >= 0.05
    out["noninferiority"] = {"margin": 0.15, "sd": 2, "n_per_group": 2000,
                             "passes_when_exactly_at_margin": rate(at_margin, sims),
                             "passes_when_no_change": rate(at_zero, sims),
                             "plain_test_misses_a_drop_of_one_margin": rate(plain_says_fine, sims)}
    save("noninferiority")


# ---------------------------------------- skewed metric tools
if want("skewed_tools"):
    rng = np.random.default_rng([31, 7])
    t0 = section("skewed tools")
    skew_out = {}
    for name, sigma, nc, nt in (("lognormal_s1_even", 1.0, 2000, 2000), ("lognormal_s1.5_uneven", 1.5, 1000, 9000)):
        sims = n_(600)
        lift = 0.10
        mu = np.exp(sigma**2 / 2)
        counts = {k: {"null_low": 0, "null_high": 0, "cover": 0, "power": 0}
                  for k in ("welch", "boot_percentile", "boot_t")}
        qcov = {0.5: 0, 0.9: 0}
        mw_null = 0
        for i in range(sims):
            c0, t0_ = np.exp(sigma * rng.normal(0, 1, nc)), np.exp(sigma * rng.normal(0, 1, nt))
            c1, t1 = np.exp(sigma * rng.normal(0, 1, nc)), np.exp(sigma * rng.normal(0, 1, nt)) * (1 + lift)
            for k, f in (("welch", lambda a, b: ab.mean_test(a, b)),
                         ("boot_percentile", lambda a, b: ab.bootstrap_test(a, b, n_boot=500, seed=i)),
                         ("boot_t", lambda a, b: ab.bootstrap_test(a, b, n_boot=500, seed=i, method="studentized"))):
                r0, r1 = f(c0, t0_), f(c1, t1)
                counts[k]["null_low"] += r0.ci[1] < 0
                counts[k]["null_high"] += r0.ci[0] > 0
                counts[k]["cover"] += r1.ci[0] <= mu * lift <= r1.ci[1]
                counts[k]["power"] += r1.p_value < 0.05
            for q in qcov:
                true_q = np.exp(sigma * __import__("scipy").stats.norm.ppf(q)) * lift
                r = ab.quantile_test(c1, t1, q, n_boot=500, seed=i)
                qcov[q] += r.ci[0] <= true_q <= r.ci[1]
            mw_null += ab.mann_whitney(c0, t0_).p_value < 0.05
        skew_out[name] = {
            "n_control": nc, "n_treatment": nt, "lift": lift,
            **{k: {"type1_low": rate(v["null_low"], sims), "type1_high": rate(v["null_high"], sims),
                   "coverage": rate(v["cover"], sims), "power": rate(v["power"], sims)} for k, v in counts.items()},
            "quantile_coverage": {str(q): rate(v, sims) for q, v in qcov.items()},
            "mann_whitney_type1": rate(mw_null, sims),
        }
    # same mean, different shape: a t test says nothing changed, Mann-Whitney fires
    sims = n_(1000)
    mw_fire = t_fire = 0
    for _ in range(sims):
        c = rng.exponential(1.0, 3000)                  # mean 1
        t = np.exp(rng.normal(-0.5 * 0.8**2, 0.8, 3000))  # lognormal with mean 1, different shape
        mw_fire += ab.mann_whitney(c, t).p_value < 0.05
        t_fire += ab.mean_test(c, t).p_value < 0.05
    skew_out["same_mean_different_shape"] = {"mann_whitney_rejects": rate(mw_fire, sims),
                                             "t_test_rejects": rate(t_fire, sims)}
    out["skewed_tools"] = skew_out
    save("skewed_tools")


# ------------------------------------------------------- time checks
if want("time_checks"):
    rng = np.random.default_rng([31, 8])
    t0 = section("time checks")
    days = pd.date_range("2026-03-02", periods=14, freq="D")      # starts on a Monday

    def daily(rng, lift_fn, per_day=1500):
        yc, yt, tc, tt = [], [], [], []
        for i, day in enumerate(days):
            yc.append(rng.normal(0, 1, per_day)); tc += [day] * per_day
            yt.append(rng.normal(lift_fn(i, day), 1, per_day)); tt += [day] * per_day
        return np.concatenate(yc), np.concatenate(yt), tc, tt

    cases = {"constant": lambda i, d: 0.05,
             "novelty": lambda i, d: 0.05 + 0.10 * np.exp(-i / 3),
             "weekend": lambda i, d: 0.05 + (0.12 if d.day_of_week >= 5 else 0.0)}
    tc_out = {}
    for name, f in cases.items():
        sims = n_(1000)
        trend = week = 0
        for _ in range(sims):
            a, b, ta, tb = daily(rng, f)
            r = ab.time_checks(a, b, ta, tb)
            trend += r.trend_p < 0.05
            week += r.weekday_p < 0.05
        tc_out[name] = {"trend_test_fires": rate(trend, sims), "weekday_test_fires": rate(week, sims)}
    out["time_checks"] = {"days": 14, "users_per_day_per_group": 1500, **tc_out}
    save("time_checks")


# ------------------------------------------------------ effect screening
if want("screening"):
    rng = np.random.default_rng([31, 9])
    t0 = section("screening")
    sims = n_(500)
    false_hit = real_hit = 0
    for i in range(sims):
        n = 8000
        feats = {f"f{j}": rng.normal(0, 1, n) for j in range(6)}
        feats["country"] = rng.choice(["US", "CA", "UK", "DE"], n)
        feats["platform"] = rng.choice(["ios", "android", "web"], n)
        g_ = rng.integers(0, 2, n)
        y0 = rng.normal(0, 1, n) + 0.05 * g_
        df = pd.DataFrame({"v": np.where(g_, "B", "A"), "y": y0, **feats})
        e = ab.Experiment(df, "v", "A", "B")
        false_hit += (e.screen("y", list(feats))["p_bh"] < 0.05).any()
        df["y"] = y0 + 0.25 * g_ * (feats["f2"] > 0)          # the effect lives in users with f2 > 0
        t = ab.Experiment(df, "v", "A", "B").screen("y", list(feats))
        real_hit += bool(((t["feature"] == "f2") & (t["p_bh"] < 0.05)).any())
    out["screening"] = {"features": 8, "users": 8000, "any_false_lead": rate(false_hit, sims),
                        "finds_the_real_modifier": rate(real_hit, sims)}
    save("screening")


out["runtime_minutes_last_run"] = (time.time() - START) / 60
done = all(k in out for k in ("group_sequential", "cupac", "ratio_power", "aa", "dunnett", "noninferiority",
                              "skewed_tools", "time_checks", "screening"))
if not QUICK and done:
    (RES / "new_methods.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=float)[:8000])
