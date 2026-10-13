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

# 1. health check before anything else
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    health = exp.check()
out["health"] = {**health, "srm_warning": bool(w)}
print(f"players: {health['n_control']:,} in gate_30, {health['n_treatment']:,} in gate_40")
print(f"sample ratio check p = {health['srm_p']:.4f}" + ("  <-- mismatch" if w else ""))

# 2. the metrics, corrected for testing several at once
metrics = ["retention_1", "retention_7", "rounds_capped"]
table = exp.analyze_many(metrics, correction="bh")
print("\n" + table.round(4).to_string(index=False))
out["metrics"] = table.to_dict(orient="records")
for m in metrics:
    print("\n" + str(exp.analyze(m)))

# 3. how many players would you need to trust a result this small?
r7 = exp.analyze("retention_7")
n_needed, _ = ab.proportion_sample_size(r7.control_mean, abs(r7.lift))
out["retention_7_players_needed_per_group_for_80pct_power"] = n_needed
print(f"\nto detect a {abs(r7.lift):.4f} change in 7 day retention with 80% power you'd need "
      f"{n_needed:,} players per group; this test had about {health['n_control']:,}")

# 4. a sequential replay. The data has no timestamps, so this uses 200 random
# arrival orders and asks how often an always valid test, checked 50 times,
# would have called the 7 day retention result before the end.
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
out["sequential_replay"] = {"orders": 200, "msprt_called_it": stopped / 200,
                            "naive_peeking_called_it": naive_stopped / 200,
                            "msprt_avg_share_when_called": float(np.mean(stop_share)) if stop_share else None}
print(f"\nreplayed in 200 random arrival orders: mSPRT called it {stopped / 200:.0%} of the time, "
      f"naive peeking {naive_stopped / 200:.0%}")

# one example path for the chart
cc, tt = np.random.default_rng(3).permutation(c), np.random.default_rng(4).permutation(t)
r = ab.msprt(cc, tt, tau=0.01, n_looks=50)
users = r.n_control + r.n_treatment
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.fill_between(users, r.ci_low * 100, r.ci_high * 100, color="#2F4BC4", alpha=0.18, label="always valid interval")
ax.plot(users, r.estimate * 100, color="#2F4BC4", lw=2, label="estimate")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("players so far"); ax.set_ylabel("gate_40 minus gate_30 (points)")
ax.set_title("7 day retention, replayed as if checked 50 times")
ax.set_ylim(-3, 3); ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25); [ax.spines[s].set_visible(False) for s in ("top", "right")]
fig.tight_layout(); fig.savefig(ROOT / "figures" / "cookie_cats_sequential.png", dpi=150)

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "cookie_cats.json").write_text(json.dumps(out, indent=2, default=float))
