"""
Builds docs/validation.md: one table with every method, the data it was
simulated on, and pass or fail, straight from the saved results (nothing
typed by hand). Run after the simulations:

    python tools/validation_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(r, se=None):
    if r is None:
        return ""
    return f"{r * 100:.1f}" + (f" ± {se * 100:.1f}" if se is not None else "")


def judge(t1=None, cov=None, tails=None, nominal=0.05, bias=None):
    """The rule, applied the same way to every row."""
    bad, notes = [], []
    if t1 is not None:
        if t1["rate"] > nominal + 2 * t1["mc_se"]:
            bad.append("too many false positives")
        elif t1["rate"] < nominal - 2 * t1["mc_se"]:
            notes.append("conservative")
    if cov is not None and cov["rate"] < 0.95 - 2 * cov["mc_se"]:
        bad.append("interval too short")
    if tails is not None:
        worst = max(tails, key=lambda x: x["rate"])
        if worst["rate"] > nominal / 2 + 2 * worst["mc_se"]:
            bad.append("errors pile up on one side")
    if bias is not None and abs(bias[0]) > max(5, 3 * bias[1]):
        bad.append(f"biased ({bias[0]:+.0f}% of the effect)")
    return ("fail" if bad else "pass"), "; ".join(bad + notes)


ROWS = []
# a baseline is the naive version of something abkit offers (a row level test on clustered data, peeking,
# no cross fitting, no multiplicity correction) or an estimate of a different quantity (capped means)
BASELINE_MARKS = ("by design", "the mistake", "the problem", "documented trap", "treated as independent",
                  "row level problem")


def row(method, data, sims, t1=None, cov=None, tails=None, bias=None, nominal=0.05, remark=""):
    verdict, why = judge(t1, cov, tails, nominal, bias)
    note = "; ".join(x for x in (why, remark) if x)
    ROWS.append({"baseline": any(k in remark for k in BASELINE_MARKS),
                 "method": method, "data": data, "sims": sims,
                 "type1": pct(t1["rate"], t1["mc_se"]) if t1 else "",
                 "coverage": pct(cov["rate"], cov["mc_se"]) if cov else "",
                 "bias": f"{bias[0]:+.1f} ± {bias[1]:.1f}" if bias else "",
                 "verdict": verdict, "note": note})


NAMES = {"welch": "mean_test (Welch)", "capped_99": "mean_test after winsorize(0.99)", "cuped": "CUPED",
         "lin": "adjusted_test (Lin)", "bayes_flat": "bayes_test, flat prior (95% interval)",
         "cluster": "cluster_test", "cluster_perm": "cluster_test, permutations", "delta_normal":
         "ratio_test on cluster sums (normal)"}
REMARK = {("capped_99",): "estimates the capped mean, so this is the bias of capping, by design",
          ("welch", "sessions"): "rows of the same user treated as independent",
          ("welch", "clusters_50"): "rows of the same cluster treated as independent",
          ("welch", "clusters_8"): "rows of the same cluster treated as independent",
          ("welch", "missing_differential"): "no method can fix this; the missing data check fired, see below",
          ("cuped", "missing_differential"): "no method can fix this",
          ("welch", "missing_mnar"): "missingness depends on the outcome",
          ("cuped", "missing_mnar"): "missingness depends on the outcome",
          ("welch", "lognormal_s2_uneven"): "the skew note fires here",
          ("cuped", "lognormal_s2"): "slightly short at this sample size, fine at 20,000 per group",
          ("lin", "lognormal_s2"): "slightly short at this sample size, fine at 20,000 per group",
          ("cuped", "lognormal_s2_uneven"): "same pile up as the t test, adjusting does not fix uneven groups",
          ("lin", "lognormal_s2_uneven"): "same pile up as the t test, adjusting does not fix uneven groups",
          ("cluster_perm", "clusters_50"): "p value from the randomization test, interval is still t based",
          ("cluster_perm", "clusters_8"): "p value from the randomization test, the interval is still t based",
          ("bayes_flat", "lognormal_s2_uneven"): "same normal approximation as the t test"}


def stress_rows(st):
    n0, n1 = st["sims_null"], st["sims_alt"]
    for name, sc in st["scenarios"].items():
        for m, r in sc["methods"].items():
            if m == "msprt":
                continue
            rem = REMARK.get((m, name)) or REMARK.get((m,), "")
            row(NAMES[m], sc["label"], f"{n0:,} + {n1:,}", r["type1"], r["coverage"],
                None if m == "cluster_perm" else [r["type1_low"], r["type1_high"]], (r["bias_pct_of_effect"], r["bias_mc_se_pct"]),
                remark=rem)
        if "msprt" in sc["methods"]:
            r = sc["methods"]["msprt"]
            row("msprt, 50 looks (any look)", sc["label"], f"{n0:,} + {n1:,}", r["type1"], r["cs_coverage"],
                remark="rows of the same user treated as independent" if name == "sessions" else "")


def original_rows(sim):
    pk = sim["peeking"]
    k = int(round(pk["msprt_false_positive_after_all_looks"] * pk["sims"]))
    row("msprt, 100 looks (any look)", "normal, 10,000 per group", f"{pk['sims']:,}", rate(k, pk["sims"]))
    k = int(round(pk["naive_false_positive_after_all_looks"] * pk["sims"]))
    row("ordinary p value checked 100 times", "normal, 10,000 per group", f"{pk['sims']:,}", rate(k, pk["sims"]),
        remark="the mistake the sequential tests exist to fix")
    cu = sim["cuped"]
    for rho, c in zip(cu["rho"], cu["ci_coverage_cuped"]):
        row("CUPED", f"normal, covariate correlation {rho}", "800", cov=rate(int(round(c * 800)), 800))
    rm = sim["ratio_metric"]
    row("ratio_test (delta method)", "clicks per session, 2,000 users per group", "1,000",
        cov=rate(int(round(rm["coverage_delta_method"] * 1000)), 1000))
    row("sessions treated as independent", "clicks per session, 2,000 users per group", "1,000",
        cov=rate(int(round(rm["coverage_naive_sessions"] * 1000)), 1000), remark="the mistake ratio_test fixes")
    mt = sim["multiple_testing"]
    row("holm (any false win)", "20 metrics, 15 with no effect", "2,000",
        rate(int(round(mt["chance_any_false_win_holm"] * 2000)), 2000))
    row("benjamini_hochberg (false discovery rate)", "20 metrics, 15 with no effect", "2,000",
        {"rate": mt["false_discovery_rate_bh"], "mc_se": mt["false_discovery_rate_bh_mc_se"]})
    sg = sim["segments"]
    row("heterogeneity_test (Cochran's Q)", "10 segments, same effect everywhere", "1,000",
        rate(int(round(sg["heterogeneity_test_false_alarm_rate"] * 1000)), 1000))
    row("segments with BH (any segment significant)", "10 segments, no effect", "1,000",
        rate(int(round(sg["chance_some_segment_significant_bh"] * 1000)), 1000))


def rate(k, n):
    return {"rate": k / n, "mc_se": max(k / n * (1 - k / n), 1 / n) ** 0.5 / n ** 0.5}


def new_rows(nm):
    gs = nm["group_sequential"]
    names = {"obf_5": "group_sequential_test, O'Brien Fleming, 5 looks",
             "pocock_5": "group_sequential_test, Pocock, 5 looks", "msprt_50": "msprt, 50 looks (any look)",
             "fixed": "mean_test, one look"}
    for k, label in names.items():
        sim = gs["simulated"][k]
        t1 = sim["power"][0]
        cov = sim["rci_coverage"][2] if sim.get("rci_coverage") else None
        row(label, f"normal, 10,000 per group; coverage at effect {gs['effects_sd'][2]} sd", "4,000 + 1,000",
            t1, cov)
    cp = nm["cupac"]
    lab = {"plain": "mean_test, no adjustment", "cuped_linear": "CUPED, 5 linear covariates",
           "cupac_crossfit": "cupac_adjust (cross fit)", "cupac_in_sample_gb": "CUPAC without cross fitting (boosting)",
           "cupac_in_sample_knn": "CUPAC without cross fitting (5 neighbors)"}
    for k, r in cp["methods"].items():
        row(lab[k], f"nonlinear outcome, {cp['n_per_group']:,} per group", "300 + 300", r["type1"], r["coverage"],
            bias=(r["bias_pct_of_effect"], r["bias_mc_se_pct"]),
            remark="the mistake cross fitting prevents" if "in_sample" in k else "")
    d = nm["dunnett"]
    n = f"{round(d['any_false_win_dunnett']['rate'] * (1 - d['any_false_win_dunnett']['rate']) / d['any_false_win_dunnett']['mc_se'] ** 2):,}"
    sp = d.get("any_false_win_scipy_dunnett_same_data")
    row("dunnett (any false win)", "3 arms vs control, no effect", n, d["any_false_win_dunnett"],
        remark=f"scipy.stats.dunnett on the same data: {pct(sp['rate'])}%" if sp else "")
    if sp:
        row("scipy.stats.dunnett (any false win), reference", "same simulated data", n, sp)
    row("bonferroni (any false win)", "3 arms vs control, no effect", n, d["any_false_win_bonferroni"])
    row("no correction (any false win)", "3 arms vs control, no effect", n, d["any_false_win_raw"],
        remark="the problem dunnett fixes")
    ni = nm["noninferiority"]
    row("noninferiority_test (passes a bad treatment)", "true change exactly at the margin", "4,000",
        ni["passes_when_exactly_at_margin"])
    for name, s in nm["skewed_tools"].items():
        if name == "same_mean_different_shape":
            continue
        data = f"lognormal sd {name.split('_s')[1].split('_')[0]}, {s['n_control']:,} vs {s['n_treatment']:,}"
        for k, lab_ in (("welch", "mean_test (Welch)"), ("boot_percentile", "bootstrap_test, percentile"),
                        ("boot_t", "bootstrap_test, studentized")):
            r = s[k]
            t1 = {"rate": r["type1_low"]["rate"] + r["type1_high"]["rate"],
                  "mc_se": (r["type1_low"]["mc_se"] ** 2 + r["type1_high"]["mc_se"] ** 2) ** 0.5}
            row(lab_, data, "600", t1, r["coverage"], [r["type1_low"], r["type1_high"]])
        for q, c in s["quantile_coverage"].items():
            row(f"quantile_test, q = {q}", data, "600", cov=c)
        row("mann_whitney", data + ", same distribution", "600", s["mann_whitney_type1"])
    sm = nm["skewed_tools"]["same_mean_different_shape"]
    row("mann_whitney (flags a difference)", "same mean, different shape", "1,000", sm["mann_whitney_rejects"],
        remark="not a test of means; this is the documented trap")
    tc = nm["time_checks"]["constant"]
    row("time_checks trend test", "14 days, effect constant", "1,000", tc["trend_test_fires"])
    row("time_checks day of week test", "14 days, effect constant", "1,000", tc["weekday_test_fires"])
    sc = nm["screening"]
    row("Experiment.screen (any false lead)", "8 features, none change the effect", "500", sc["any_false_lead"])
    aa = nm["aa"]
    for k, lab_ in (("normal_rows", "aa_test on clean data"),
                    ("sessions_users_split_rows_tested", "aa_test, users split, rows tested"),
                    ("sessions_users_split_cluster_test", "aa_test, users split, cluster_test")):
        r = aa[k]
        row(lab_, "real style session data" if "sessions" in k else "normal, 20,000 rows", f"{r['sims']:,}",
            {"rate": r["false_positive_rate"], "mc_se": r["mc_se"]},
            remark="shows the row level problem on your own data" if "rows_tested" in k else "")


def main():
    st, sim, nm = load("stress.json"), load("simulations.json"), load("new_methods.json")
    if sim:
        original_rows(sim)
    if st:
        stress_rows(st)
    if nm:
        new_rows(nm)
    passed = sum(r["verdict"] == "pass" for r in ROWS)
    failed = [r for r in ROWS if r["verdict"] == "fail"]
    base_fail = [r for r in failed if r["baseline"]]
    lines = [
        "# Validation",
        "",
        "Every method, every kind of data it was simulated on, and whether it held up. This file is written by "
        "`tools/validation_table.py` from `results/simulations.json`, `results/stress.json` and "
        "`results/new_methods.json`, so nothing here is typed by hand.",
        "",
        "**The rule.** A row passes when the false positive rate is not above the nominal 5% by more than 2 Monte "
        "Carlo standard errors, the 95% interval covers the truth at least 95% of the time minus 2 standard errors, "
        "neither tail gets more than 2.5% plus 2 standard errors of the false positives (when tails were "
        "recorded), and the estimate's bias is under 5% of the effect (or within 3 standard errors). "
        "Conservative rows (well under 5%) pass and are marked.",
        "",
        "**How to read it.** Type I is the share of simulated tests with no real effect that came out significant "
        "(for sequential tests, at any look). Coverage is the share of 95% intervals that contained the true "
        "effect when there was one (for sequential tests, at every look at once). Numbers are percent ± one Monte "
        "Carlo standard error. Bias is percent of the true effect ± one standard error. Rows marked with a remark "
        "like \"the mistake ... fixes\" are deliberate baselines that are supposed to fail.",
        "",
        f"{passed} of {len(ROWS)} rows pass. {len(failed)} fail, and {len(base_fail)} of those are baselines "
        f"(marked in the Baseline column) that exist to show the mistake or the different estimate. The other "
        f"{len(failed) - len(base_fail)} are real limits of the library, each with its reason in the Note column.",
        "",
        "| Method | Data | Sims (null + effect) | Type I % | Coverage % | Bias % | Verdict | Baseline | Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ROWS:
        lines.append(f"| {r['method']} | {r['data']} | {r['sims']} | {r['type1']} | {r['coverage']} | {r['bias']} | "
                     f"{r['verdict']} | {'yes' if r['baseline'] else ''} | {r['note']} |")
    if nm:
        rp = nm["ratio_power"]
        g = nm["group_sequential"]
        lines += ["", "## Power checks", "",
                  "The ratio check plans for a lift of 0.01 from pilot data. In the simulation, clipping rates at "
                  "1 makes the true lift a little smaller and the treatment group a little noisier, so the third "
                  "column is the exact power the delta method number should deliver here.", "",
                  "| Check | Promised | Expected here | Got |", "|---|---|---|---|"]
        for key, lab_ in (("delta_method_n", "ratio_sample_size, delta method"),
                          ("naive_session_n", "per session shortcut")):
            r = rp[key]
            exp_ = r.get("expected_power_given_true_lift")
            lines.append(f"| {lab_}: {r['users_per_group']:,} users per group | 80% | "
                         f"{pct(exp_) + '%' if exp_ is not None else ''} | "
                         f"{pct(r['achieved_power']['rate'], r['achieved_power']['mc_se'])}% |")
        lines += ["", "| Check | Numerical integration | Simulation |", "|---|---|---|"]
        for k, label in (("obf_5", "O'Brien Fleming, 5 looks"), ("pocock_5", "Pocock, 5 looks")):
            for i, e in enumerate(g["effects_sd"]):
                if e == 0:
                    continue
                sim_ = g["simulated"][k]["power"][i]
                lines.append(f"| {label}, power at effect {e} sd | "
                             f"{g['theory'][k]['power'][i] * 100:.1f}% | {pct(sim_['rate'], sim_['mc_se'])}% |")
    if st:
        lines += ["", "## Other checks from the stress tests", "", "| Check | Result |", "|---|---|"]
        for name, sc in st["scenarios"].items():
            if "srm_false_alarm_counting_rows" in sc:
                lines.append(f"| split check counting rows, {sc['label']} (no real problem) | flagged "
                             f"{pct(sc['srm_false_alarm_counting_rows']['rate'])}% of tests |")
                lines.append(f"| split check counting clusters, same data | flagged "
                             f"{pct(sc['srm_false_alarm_counting_clusters']['rate'])}% of tests |")
            if "missing_check_fires" in sc:
                lines.append(f"| missing value check, {sc['label']} | fired in "
                             f"{pct(sc['missing_check_fires']['rate'])}% of tests |")
            if sc.get("skew_note_rate_null", 0) > 0:
                lines.append(f"| skew note on mean_test, {sc['label']} | shown on "
                             f"{pct(sc['skew_note_rate_null'])}% of tests |")
        quiet = [sc["label"] for sc in st["scenarios"].values() if sc.get("skew_note_rate_null") == 0]
        lines.append(f"| skew note on mean_test, every other scenario | never shown ({len(quiet)} scenarios) |")
        lines += ["", "## Run time", "", "Median milliseconds per call on one shared core.", "",
                  "| Method | 10,000 per group | 100,000 | 1,000,000 |", "|---|---|---|---|"]
        tm = {}
        for r in st["timing"]:
            tm.setdefault(r["method"], {})[r["n_per_group"]] = r["ms"]
        for m, v in tm.items():
            lines.append(f"| {m} | {v[10000]:.1f} | {v[100000]:.1f} | {v[1000000]:.0f} |")
    (ROOT / "docs" / "validation.md").write_text("\n".join(lines) + "\n")
    (RES / "validation_summary.json").write_text(json.dumps(
        {"rows": len(ROWS), "passed": passed, "failed": len(failed), "failed_baselines": len(base_fail),
         "failed_real_limits": len(failed) - len(base_fail),
         "baseline_rows_that_fail": [f"{r['method']} | {r['data']}" for r in base_fail],
         "real_limit_rows": [f"{r['method']} | {r['data']}" for r in failed if not r["baseline"]]}, indent=2))
    print(f"{passed} of {len(ROWS)} rows pass; wrote docs/validation.md")


if __name__ == "__main__":
    main()
