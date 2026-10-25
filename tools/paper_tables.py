"""
Writes the LaTeX tables in paper/tables/ from results/*.json, so the
paper's tables can't drift from the code that produced them.

    python tools/paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "tables"


def load(name):
    return json.loads((RES / name).read_text())


def p(x, se=None, d=1):
    s = f"{x * 100:.{d}f}"
    return s + (f" $\\pm$ {se * 100:.{d}f}" if se is not None else "")


def tex(s):
    return s.replace("_", "\\_").replace("%", "\\%")


def stress_table(st):
    keep = ["normal", "lognormal_s2", "lognormal_s2_uneven", "pareto_1.5", "zero_inflated", "binary_0.001",
            "sessions", "clusters_50", "clusters_8", "missing_mnar", "missing_differential", "hte_uneven"]
    rows = []
    for name in keep:
        sc = st["scenarios"][name]
        w = sc["methods"]["welch"]
        fix = None
        for m in ("cluster_perm", "cluster", "cuped", "lin"):
            if m in sc["methods"] and m != "welch":
                fix = m
                break
        f = sc["methods"].get(fix) if fix else None
        tails = f"{w['type1_low']['rate'] * 100:.1f} / {w['type1_high']['rate'] * 100:.1f}"
        fx = (f"{fix.replace('_', ' ')}: {p(f['type1']['rate'])}, {p(f['coverage']['rate'])}" if f else "")
        label = sc["label"].replace("%", "\\%").replace("_", " ")
        rows.append(f"{label} & {p(w['type1']['rate'], w['type1']['mc_se'])} & {tails} & "
                    f"{p(w['coverage']['rate'], w['coverage']['mc_se'])} & {fx} \\\\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{p{4.3cm}rrrp{4.4cm}}\n\\toprule\n"
            "Data & Type I (\\%) & lower / upper tail & Coverage (\\%) & Alternative: Type I, coverage \\\\\n"
            "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def gsd_table(nm):
    g = nm["group_sequential"]
    names = {"fixed": "Single look", "msprt_50": "mSPRT, 50 looks", "obf_5": "O'Brien Fleming, 5 looks",
             "pocock_5": "Pocock, 5 looks"}
    head = " & ".join(f"{e}" for e in g["effects_sd"])
    rows = []
    for k, lab in names.items():
        s = g["simulated"][k]
        pw = " & ".join(p(x["rate"], d=0) for x in s["power"])
        sh = " & ".join(f"{x * 100:.0f}" for x in s["avg_share"])
        rows.append(f"{lab} & {pw} & {sh} \\\\")
    n = len(g["effects_sd"])
    return (f"\\begin{{tabular}}{{l{'r' * n}|{'r' * n}}}\n\\toprule\n"
            f" & \\multicolumn{{{n}}}{{c|}}{{Rejection rate (\\%) at effect (sd)}} & "
            f"\\multicolumn{{{n}}}{{c}}{{Average share of users used (\\%)}} \\\\\n"
            f" & {head} & {head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def cupac_table(nm):
    c = nm["cupac"]["methods"]
    names = {"plain": "No adjustment", "cuped_linear": "CUPED, 5 linear covariates",
             "cupac_crossfit": "CUPAC, cross fit", "cupac_in_sample_gb": "CUPAC, no cross fit (boosting)",
             "cupac_in_sample_knn": "CUPAC, no cross fit (5 neighbors)"}
    rows = [f"{lab} & {p(c[k]['variance_cut'], d=0)} & {p(c[k]['type1']['rate'], c[k]['type1']['mc_se'])} & "
            f"{p(c[k]['coverage']['rate'], c[k]['coverage']['mc_se'])} & {c[k]['bias_pct_of_effect']:+.0f} & "
            f"{p(c[k]['power']['rate'], d=0)} \\\\" for k, lab in names.items()]
    return ("\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & Variance cut (\\%) & Type I (\\%) & Coverage (\\%) & "
            "Bias (\\% of effect) & Power (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def srm_table(srm):
    rows = []
    for name, r in srm["sensitivity"]["retention_7"].items():
        pv = "$<$0.0001" if r["p"] < 1e-4 else f"{r['p']:.4f}"
        rows.append(f"{tex(name)} & ${r['lift'] * 100:+.2f}$ & ${r['ci'][0] * 100:+.2f}$ to ${r['ci'][1] * 100:+.2f}$ & {pv} \\\\")
    return ("\\begin{tabular}{lrrr}\n\\toprule\nAnalysis & Change (points) & 95\\% CI & $p$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stress.tex").write_text(stress_table(load("stress.json")))
    nm = load("new_methods.json")
    (OUT / "gsd.tex").write_text(gsd_table(nm))
    (OUT / "cupac.tex").write_text(cupac_table(nm))
    (OUT / "srm.tex").write_text(srm_table(load("cookie_cats_srm.json")))
    print("wrote paper/tables/*.tex")


if __name__ == "__main__":
    main()
