"""
A real A/B test: Cookie Cats, a mobile puzzle game.

90,189 new players were randomly put in one of two versions. In gate_30 the
first gate (where you wait or pay to keep playing) is at level 30. In
gate_40 it's at level 40. The question was whether moving it later hurts
retention.

Run python analysis/get_data.py first, then this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402

df = pd.read_csv(ROOT / "data" / "cookie_cats.csv")
df["retention_1"] = df["retention_1"].astype(int)
df["retention_7"] = df["retention_7"].astype(int)
cap = df["sum_gamerounds"].quantile(0.99)
df["rounds_capped"] = df["sum_gamerounds"].clip(upper=cap)

exp = ab.Experiment(df, variant="version", control="gate_30", treatment="gate_40")
out: dict = {}

# health check before anything else. A failed split check is worth chasing
# down but doesn't show the test is broken, see cookie_cats_srm.py
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    health = exp.check()
out["health"] = {**health, "srm_warning": bool(w)}
print(f"players: {health['n_control']:,} in gate_30, {health['n_treatment']:,} in gate_40")
print(f"sample ratio check p = {health['srm_p']:.4f}" + ("  <-- mismatch" if w else ""))

# the metrics, corrected for testing several at once
metrics = ["retention_1", "retention_7", "rounds_capped"]
table = exp.analyze_many(metrics, correction="bh")
print("\n" + table.round(4).to_string(index=False))
out["metrics"] = table.to_dict(orient="records")
for m in metrics:
    print("\n" + str(exp.analyze(m)))

# how many players would you need to trust a result this small?
r7 = exp.analyze("retention_7")
n_needed, _ = ab.proportion_sample_size(r7.control_mean, abs(r7.lift))
out["retention_7_players_needed_per_group_for_80pct_power"] = n_needed
print(f"\nto detect a {abs(r7.lift):.4f} change in 7 day retention with 80% power you'd need "
      f"{n_needed:,} players per group; this test had about {health['n_control']:,}")

# a replay as if someone checked the test 50 times while it ran. The data
# has NO timestamps, so this can't be the real order players arrived in. It
# shuffles the players at random (200 orders, as in the first version) and
# asks how often an always valid test would have called the 7 day retention
# result before the end. A shuffled replay can't show anything that depends
# on time (novelty, weekdays, a bug fixed halfway through).
rng = np.random.default_rng(0)
c = df.loc[df.version == "gate_30", "retention_7"].to_numpy(float)
t = df.loc[df.version == "gate_40", "retention_7"].to_numpy(float)
stopped, naive_stopped, stop_share = 0, 0, []
from abkit.sequential import naive_peeking
for _ in range(200):
    cc, tt = rng.permutation(c), rng.permutation(t)
    r = ab.msprt(cc, tt, tau=0.01, n_looks=50)
    if r.stopped:
        stopped += 1
        i = r.stop_index
        stop_share.append((r.n_control[i] + r.n_treatment[i]) / (len(c) + len(t)))
    naive_stopped += (naive_peeking(cc, tt, n_looks=50) < 0.05).any()
out["sequential_replay"] = {"orders": 200, "order_type": "random shuffles (the data has no timestamps)",
                            "msprt_called_it": stopped / 200,
                            "naive_peeking_called_it": naive_stopped / 200,
                            "msprt_avg_share_when_called": float(np.mean(stop_share)) if stop_share else None}
print(f"\nreplayed in 200 shuffled orders (no timestamps): mSPRT called it {stopped / 200:.0%} of the time, "
      f"naive peeking {naive_stopped / 200:.0%}")

# how much the stopping point depends on the shuffle: 1,000 more orders
rng2 = np.random.default_rng(1)
shares, paths = [], []
for i in range(1000):
    cc, tt = rng2.permutation(c), rng2.permutation(t)
    r = ab.msprt(cc, tt, tau=0.01, n_looks=50)
    shares.append((r.n_control[r.stop_index] + r.n_treatment[r.stop_index]) / (len(c) + len(t))
                  if r.stopped else np.nan)
    if i < 40:
        paths.append(r.estimate * 100)
shares = np.array(shares)
called = shares[~np.isnan(shares)]
out["shuffled_replay_spread"] = {
    "orders": 1000, "order_type": "random shuffles (the data has no timestamps)",
    "msprt_called_it": float(len(called) / 1000),
    "stop_share_quantiles": {str(q): float(np.quantile(called, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    "stop_share_mean": float(called.mean()),
}
qs = out["shuffled_replay_spread"]["stop_share_quantiles"]
print(f"over 1,000 shuffles it stopped after {qs['0.05']:.0%} to {qs['0.95']:.0%} of players (5th to 95th "
      f"percentile), median {qs['0.5']:.0%}")

# chart: 40 shuffled paths plus one path's always valid interval, and where it stops
cc, tt = np.random.default_rng(3).permutation(c), np.random.default_rng(4).permutation(t)
r = ab.msprt(cc, tt, tau=0.01, n_looks=50)
users = r.n_control + r.n_treatment
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
for path in paths:
    ax[0].plot(users, path, color="#9AA3B2", lw=0.6, alpha=0.6)
ax[0].fill_between(users, r.ci_low * 100, r.ci_high * 100, color="#2F4BC4", alpha=0.18,
                   label="always valid interval, one shuffle")
ax[0].plot(users, r.estimate * 100, color="#2F4BC4", lw=2, label="estimate, same shuffle")
ax[0].plot([], [], color="#9AA3B2", lw=1, label="estimate, 40 other shuffles")
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xlabel("players so far"); ax[0].set_ylabel("gate_40 minus gate_30 (points)")
ax[0].set_title("7 day retention, players in shuffled order")
ax[0].set_ylim(-3, 3); ax[0].legend(frameon=False, fontsize=7, loc="upper right")
ax[1].hist(called * 100, bins=np.arange(0, 101, 5), color="#2F4BC4", alpha=0.8)
ax[1].set_xlabel("% of players seen when it stopped"); ax[1].set_ylabel("shuffles (of 1,000)")
ax[1].set_title("Where it stops depends on the shuffle")
for a_ in ax:
    a_.grid(alpha=0.25); [a_.spines[s].set_visible(False) for s in ("top", "right")]
fig.tight_layout(); fig.savefig(ROOT / "figures" / "cookie_cats_sequential.png", dpi=150)

# and the full readout a PM would get, written to examples/
from abkit.report import readout  # noqa: E402
(ROOT / "examples").mkdir(exist_ok=True)
for fmt, ext in (("markdown", "md"), ("html", "html")):
    (ROOT / "examples" / f"cookie_cats_readout.{ext}").write_text(
        readout(exp, ["retention_7", "retention_1", "rounds_capped"], primary="retention_7",
                guardrails={"retention_1": 0.01}, fmt=fmt,
                title="Cookie Cats: moving the first gate from level 30 to 40"))

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "cookie_cats.json").write_text(json.dumps(out, indent=2, default=float))
