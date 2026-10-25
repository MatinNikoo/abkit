"""
Where does the Cookie Cats sample ratio mismatch come from, and how much
should it change what I believe about retention?

The split is 44,700 (gate_30) vs 45,489 (gate_40), p = 0.009 against
50/50. That's unusual enough to look into, but a healthy test lands
there about 1 time in 116. So instead of throwing the test out or
ignoring the check, this looks at what the data can actually say.

The idea that makes this possible: the gate is at level 30 or 40. A player
who played fewer than 29 rounds can't have finished 29 levels, so they
never saw either gate and had exactly the same game in both versions
(assuming at least one round per level, which is how the game works).
So for those players:
  * how many land in each group can't depend on the treatment, and
  * their retention can't either (a placebo test).

Run python analysis/get_data.py first, then this.
Writes results/cookie_cats_srm.json and figures/cookie_cats_srm.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, GREY, RED = "#2F4BC4", "#9AA3B2", "#C2413B"

df = pd.read_csv(ROOT / "data" / "cookie_cats.csv")
df["t"] = (df["version"] == "gate_40").astype(int)
for c in ("retention_1", "retention_7"):
    df[c] = df[c].astype(int)
CUT = 29        # fewer rounds than this: never reached a gate
out: dict = {}


def split(mask):
    c = int(((df.t == 0) & mask).sum())
    t = int(((df.t == 1) & mask).sum())
    return {"gate_30": c, "gate_40": t, "share_gate_40": t / (c + t), "srm_p": ab.srm_test(c, t)}


# --------------------------------------------------- the overall check
nc, nt = int((df.t == 0).sum()), int((df.t == 1).sum())
p_all = ab.srm_test(nc, nt)
out["overall"] = {"gate_30": nc, "gate_40": nt, "srm_p": p_all, "one_in": 1 / p_all,
                  "passes_at_0.001": bool(p_all >= 0.001)}

# ------------------------------------ where in the player base the excess sits
buckets = [("0", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9), ("10-19", 10, 19), ("20-28", 20, 28),
           ("29-39", 29, 39), ("40-99", 40, 99), ("100+", 100, 10**9)]
out["by_rounds"] = [{"rounds": name, **split(df.sum_gamerounds.between(lo, hi))} for name, lo, hi in buckets]
out["by_rounds_note"] = ("buckets from 29 rounds up are moved by the gate itself (gate_30 players stall near "
                         "level 30), so only the 29+ total is a fair split check above the cut")
expected_excess = (nt - nc) / 2
for b in out["by_rounds"]:
    # the excess of gate_40 players this bucket carries, vs its share of all players
    b["excess"] = (b["gate_40"] - b["gate_30"]) / 2
    b["share_of_players"] = (b["gate_30"] + b["gate_40"]) / len(df)
out["excess_total"] = expected_excess

before = df.sum_gamerounds < CUT
out["never_saw_a_gate"] = split(before)
out["could_have_seen_a_gate"] = split(~before)
table = pd.crosstab(before, df.t)
out["share_below_cut_differs_p"] = float(stats.chi2_contingency(table, correction=False)[1])
out["share_below_cut_by_group"] = {"gate_30": float(before[df.t == 0].mean()), "gate_40": float(before[df.t == 1].mean())}
out["zero_rounds"] = split(df.sum_gamerounds == 0)
out["zero_rounds"]["share_of_excess"] = (out["zero_rounds"]["gate_40"] - out["zero_rounds"]["gate_30"]) / (nt - nc)

# the cutoff came from the game design, but I also looked at others, so here they all are
curve = []
for k in range(1, 61):
    lo, hi = split(df.sum_gamerounds < k), split(df.sum_gamerounds >= k)
    curve.append({"cut": k, "below_p": lo["srm_p"], "at_or_above_p": hi["srm_p"]})
out["cutoff_curve"] = curve

# userid might track install time; if the imbalance were a few bad days it could cluster in id ranges
dec = pd.qcut(df.userid, 10, labels=False)
out["userid_deciles_p"] = float(stats.chi2_contingency(pd.crosstab(dec, df.t))[1])

# ------------------------------------------------- placebo on retention
def lift(mask, metric):
    s = df[mask]
    r = ab.mean_test(s.loc[s.t == 0, metric], s.loc[s.t == 1, metric], metric=metric)
    return {"gate_30": r.control_mean, "gate_40": r.treatment_mean, "lift": r.lift, "ci": list(r.ci),
            "p": r.p_value, "n": r.n_control + r.n_treatment, "se": r.se}


out["placebo_never_saw_a_gate"] = {m: lift(before, m) for m in ("retention_1", "retention_7")}

# ---------------------------------------- sensitivity of the retention result
sens = {}
for m in ("retention_7", "retention_1"):
    rows = {}
    rows["all players (as reported)"] = lift(np.ones(len(df), bool), m)
    rows["drop players with 0 rounds"] = lift(df.sum_gamerounds > 0, m)
    rows[f"only players with {CUT}+ rounds"] = lift(~before, m)
    # post stratified: weight the two strata by their pooled shares, so a
    # different mix of low activity players can't move the answer
    lo, hi = lift(before, m), lift(~before, m)
    w = before.mean()
    est = w * lo["lift"] + (1 - w) * hi["lift"]
    se = np.sqrt(w**2 * lo["se"] ** 2 + (1 - w) ** 2 * hi["se"] ** 2)
    rows["reweighted by rounds group"] = {"lift": est, "ci": [est - 1.96 * se, est + 1.96 * se],
                                          "p": float(2 * stats.norm.sf(abs(est / se))), "se": se}
    # the gate can't matter below the cutoff, so the whole effect is the
    # effect among players who reached it, times their share
    est, se = (1 - w) * hi["lift"], (1 - w) * hi["se"]
    rows["effect only where a gate was possible"] = {"lift": est, "ci": [est - 1.96 * se, est + 1.96 * se],
                                                     "p": float(2 * stats.norm.sf(abs(est / se))), "se": se}
    # worst and best cases: the 789 extra gate_40 records were all retained,
    # or none were (trimming bounds in the spirit of Lee, 2009)
    k = nt - nc
    y_c, y_t = df.loc[df.t == 0, m].to_numpy(), df.loc[df.t == 1, m].to_numpy()
    for name, keep_t in (("extra gate_40 records were all retained", y_t.sum() - k),
                         ("extra gate_40 records were all not retained", y_t.sum())):
        pt = keep_t / (nt - k)
        pc = y_c.mean()
        se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / (nt - k))
        rows[name] = {"lift": pt - pc, "ci": [pt - pc - 1.96 * se, pt - pc + 1.96 * se],
                      "p": float(2 * stats.norm.sf(abs((pt - pc) / se))), "se": se}
    for r in rows.values():
        r["rel_lift"] = r["lift"] / df.loc[df.t == 0, m].mean()
    sens[m] = rows
out["sensitivity"] = sens

# split the full sample gap exactly: for two groups of players s,
#   share_t * rate_t - share_c * rate_c = avg share * (rate_t - rate_c) + (share_t - share_c) * avg rate
dec = {}
for m in ("retention_7", "retention_1"):
    parts = {}
    for name, mask in (("never_saw_a_gate", before), ("could_have_seen_a_gate", ~before)):
        sc, st = (mask & (df.t == 0)).sum() / nc, (mask & (df.t == 1)).sum() / nt
        rc, rt = df.loc[mask & (df.t == 0), m].mean(), df.loc[mask & (df.t == 1), m].mean()
        parts[name] = {"within": (sc + st) / 2 * (rt - rc), "mix": (st - sc) * (rc + rt) / 2}
    dec[m] = {"total": df.loc[df.t == 1, m].mean() - df.loc[df.t == 0, m].mean(),
              "mix_of_players": sum(v["mix"] for v in parts.values()),
              "within_never_saw_a_gate": parts["never_saw_a_gate"]["within"],
              "within_could_have_seen_a_gate": parts["could_have_seen_a_gate"]["within"]}
out["gap_decomposition"] = dec
out["cut"] = CUT
out["share_below_cut"] = float(before.mean())

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "cookie_cats_srm.json").write_text(json.dumps(out, indent=2, default=float))

# ------------------------------------------------------------- printout
print(f"split {nc:,} vs {nt:,}: p = {p_all:.4f} (a healthy test gets this 1 time in {1 / p_all:.0f})")
for b in out["by_rounds"]:
    print(f"  rounds {b['rounds']:>6}: {b['gate_30']:>6,} vs {b['gate_40']:>6,}  "
          f"gate_40 share {b['share_gate_40']:.3f}  p = {b['srm_p']:.4f}")
a, z = out["never_saw_a_gate"], out["could_have_seen_a_gate"]
print(f"never saw a gate (<{CUT} rounds): {a['gate_30']:,} vs {a['gate_40']:,}, p = {a['srm_p']:.4f}")
print(f"could have seen one ({CUT}+):      {z['gate_30']:,} vs {z['gate_40']:,}, p = {z['srm_p']:.4f}")
pl = out["placebo_never_saw_a_gate"]["retention_7"]
print(f"placebo, 7 day retention among players who never saw a gate: {pl['lift'] * 100:+.2f} points, p = {pl['p']:.3f}")
for name, r in sens["retention_7"].items():
    print(f"  {name:<46} {r['lift'] * 100:+.2f} points ({r['ci'][0] * 100:+.2f} to {r['ci'][1] * 100:+.2f}), p = {r['p']:.4f}")

# --------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
pts = [(b["rounds"], b) for b in out["by_rounds"][:5]] + [(f"{CUT}+", out["could_have_seen_a_gate"])]
names = [n_ for n_, _ in pts]
share = np.array([b["share_gate_40"] for _, b in pts])
n = np.array([b["gate_30"] + b["gate_40"] for _, b in pts])
err = 1.96 * np.sqrt(0.25 / n)
x = np.arange(len(names))
colors = [RED] * 5 + [BLUE]
ax[0].errorbar(x, share * 100, yerr=err * 100, fmt="none", ecolor=GREY, lw=1.2)
ax[0].scatter(x, share * 100, color=colors, zorder=3)
ax[0].axhline(50, color="k", lw=0.8, ls="--")
ax[0].text(2.55, 49.82, "intended 50%", fontsize=7, color="#555")
ax[0].axvspan(4.5, 5.5, color=BLUE, alpha=0.06, lw=0)
ax[0].text(2, 48.45, "under 29 rounds", ha="center", fontsize=8, color=RED)
ax[0].text(5, 48.45, "29 or more", ha="center", fontsize=8, color=BLUE)
ax[0].set_xticks(x, names, fontsize=8)
ax[0].set_xlabel("game rounds played"); ax[0].set_ylabel("% of players in gate_40")
ax[0].set_ylim(48.2, 53.2)
ax[0].set_title("Share of players in gate_40 by rounds played")
rows = sens["retention_7"]
labels = list(rows)
y = np.arange(len(labels))[::-1]
for yi, name in zip(y, labels):
    r = rows[name]
    ax[1].plot([r["ci"][0] * 100, r["ci"][1] * 100], [yi, yi], color=GREY, lw=2)
    ax[1].plot(r["lift"] * 100, yi, "o", color=BLUE if name.startswith("all") else "#333")
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_yticks(y, labels, fontsize=7.5)
ax[1].set_xlabel("gate_40 minus gate_30, 7 day retention (points)")
ax[1].set_title("7 day retention under each reading")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "cookie_cats_srm.png"); plt.close(fig)
