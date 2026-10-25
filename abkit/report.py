"""
A readout a product manager can read: health checks first, then the
results, guardrails, segments, a decision, and the caveats that come
with it. Markdown, JSON or a single self contained HTML file.

The decision rule is deliberately simple and written out in the report,
so nobody has to guess how it was made:

* split check fails  -> hold and investigate
* primary metric better, guardrails pass -> ship
* primary better, a guardrail not shown safe -> look closer first
* primary worse -> don't ship
* nothing significant -> no clear effect, with the smallest effect the
  test could have found
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
import warnings

import numpy as np
from scipy import stats

from .data import plain
from .presets import resolve_guardrail


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.2%}"


def _num(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.4g}"


def _p(x):
    return "n/a" if x is None or not np.isfinite(x) else ("< 0.0001" if x < 1e-4 else f"{x:.4f}")


def build(exp, metrics, primary=None, guardrails=None, covariate=None, segment=None, cluster=None,
          higher_is_better=None, srm_threshold: float = 0.01, prior_sd=None, title=None) -> dict:
    """Run every check and test and return the pieces of the report as plain data."""
    metrics = list(metrics)
    primary = primary or metrics[0]
    guardrails = guardrails or {}
    hib = {m: True for m in [*metrics, *guardrails]}
    for m, spec in guardrails.items():
        hib[m] = resolve_guardrail(spec)[1] is not False
    hib.update(higher_is_better or {})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        health = exp.check(pre_period=[covariate] if isinstance(covariate, str) else covariate,
                           srm_threshold=srm_threshold, unit=cluster, metrics=metrics)
    rows, notes = [], []
    pvals = []
    for m in metrics:
        r = exp.analyze(m, covariate, cluster=cluster, dropna=True)
        b = exp.bayes(m, prior_sd) if prior_sd is not None and cluster is None and exp._c[m].notna().all() and exp._t[m].notna().all() \
            else None
        better = b.chance_treatment_better if b is not None else float("nan")
        if b is not None and not hib[m]:
            better = 1 - better
        rows.append({"metric": m, "control": r.control_mean, "treatment": r.treatment_mean,
                     "rel_lift": r.rel_lift, "rel_ci": r.rel_ci, "p": r.p_value, "chance_better": better,
                     "lift": r.lift, "se": r.se, "higher_is_better": hib[m], "n": r.n_control + r.n_treatment})
        pvals.append(r.p_value)
        notes += [f"{m}: {n}" for n in r.notes if "CUPED" not in n]
    from .multiple import benjamini_hochberg
    adj = benjamini_hochberg(pvals) if len(pvals) > 1 else pvals
    for row, a in zip(rows, adj):
        row["p_adjusted"] = float(a)

    guard = None
    if guardrails:
        g = exp.guardrails(guardrails, {m: hib[m] for m in guardrails})
        guard = g.to_dict(orient="records")

    seg = None
    if segment:
        table, het = exp.segments(primary, segment, covariate=covariate)
        seg = {"by": segment, "heterogeneity_p": het, "rows": table.to_dict(orient="records")}

    prim = next(r for r in rows if r["metric"] == primary)
    # mde = (z_alpha/2 + z_power) * se, the same formula as design.mde
    detectable = (stats.norm.isf(exp.alpha / 2) + stats.norm.ppf(0.8)) * prim["se"]
    detectable_rel = detectable / abs(prim["control"]) if prim["control"] else float("nan")

    srm_bad = health["srm_p"] < srm_threshold
    good = (prim["lift"] > 0) == prim["higher_is_better"]
    sig = prim["p"] < exp.alpha
    failed = [g_["metric"] for g_ in (guard or []) if not g_["passes"]]
    if srm_bad:
        decision = "Hold"
        why = (f"The split check failed (p = {_p(health['srm_p'])}). That alone doesn't show the test is "
               "broken, but find out why before acting on the numbers below.")
    elif sig and good and not failed:
        decision = "Ship"
        why = f"{primary} improved ({_pct(prim['rel_lift'])}, p = {_p(prim['p'])})" + \
              (" and every guardrail passed." if guard else ".")
    elif sig and good:
        decision = "Look closer"
        why = (f"{primary} improved ({_pct(prim['rel_lift'])}, p = {_p(prim['p'])}), but this test couldn't show "
               f"{', '.join(failed)} stayed within the margin.")
    elif sig:
        decision = "Don't ship"
        why = f"{primary} got worse ({_pct(prim['rel_lift'])}, p = {_p(prim['p'])})."
    else:
        decision = "No clear effect"
        why = (f"{primary} moved {_pct(prim['rel_lift'])} but that's within noise (p = {_p(prim['p'])}). "
               f"With this many users the test had 80% power for a change of about {_pct(detectable_rel)}; "
               "anything smaller could easily be missed.")

    caveats = list(notes)
    if srm_bad:
        caveats.append("Sample ratio mismatch: check assignment, logging, bots and anything that drops users from "
                       "one group. By chance alone 1 healthy test in 100 lands below 0.01.")
    for m, r in (health.get("missing") or {}).items():
        if r["missing_control"] or r["missing_treatment"]:
            caveats.append(f"{m} is missing for {r['missing_control']:.1%} of control and "
                           f"{r['missing_treatment']:.1%} of treatment (p = {_p(r['p'])}); those rows were dropped.")
    if len(metrics) > 1:
        caveats.append(f"{len(metrics)} metrics were tested, so their p values are also shown adjusted "
                       "(Benjamini Hochberg). The primary metric is judged on its own p value because it was "
                       "picked before the test.")
    if seg and seg["heterogeneity_p"] >= 0.05:
        caveats.append(f"Segments by {segment} don't differ by more than noise (p = {_p(seg['heterogeneity_p'])}), "
                       "so read single segment results as the same effect plus noise.")
    return {"title": title or f"Experiment readout: {exp.treatment} vs {exp.control}", "decision": decision,
            "why": why, "health": health, "srm_bad": srm_bad, "rows": rows, "primary": primary,
            "guardrails": guard, "segments": seg, "caveats": caveats, "alpha": exp.alpha,
            "control": str(exp.control), "treatment": str(exp.treatment),
            "detectable_rel": detectable_rel}


def _tables(rep):
    h = rep["health"]
    out = []
    checks = [["Users", f"{h['n_control']:,} {rep['control']}, {h['n_treatment']:,} {rep['treatment']}"],
              ["Split check (SRM)", f"p = {_p(h['srm_p'])}" + ("  (failed, investigate)" if rep["srm_bad"] else "  ok")]]
    for col, p in (h.get("balance") or {}).items():
        checks.append([f"Balance on {col}", f"p = {_p(p)}" + ("  (unbalanced)" if p < 0.001 else "  ok")])
    out.append(("Health checks", ["Check", "Result"], checks))
    # the chance to win is only shown with a prior, because a flat prior overstates it when effects are small
    show_chance = any(np.isfinite(r["chance_better"]) for r in rep["rows"])
    res = [[r["metric"] + (" (primary)" if r["metric"] == rep["primary"] else ""), _num(r["control"]),
            _num(r["treatment"]), _pct(r["rel_lift"]), f"{_pct(r['rel_ci'][0])} to {_pct(r['rel_ci'][1])}",
            _p(r["p"]), _p(r["p_adjusted"])] +
           ([f"{r['chance_better']:.0%}" if np.isfinite(r["chance_better"]) else "n/a"] if show_chance else [])
           for r in rep["rows"]]
    head = ["Metric", rep["control"], rep["treatment"], "Change", f"{1 - rep['alpha']:.0%} CI", "p", "Adjusted p"]
    out.append(("Results", head + (["Chance better"] if show_chance else []), res))
    if rep["guardrails"]:
        g = [[x["metric"], _num(x["control"]), _num(x["treatment"]), _pct(x["rel_lift"]),
              f"{x['margin']:.2%} " + ("worse" if x["higher_is_better"] else "higher"), _p(x["p"]),
              "pass" if x["passes"] else "not shown"] for x in rep["guardrails"]]
        out.append(("Guardrails (non-inferiority)", ["Metric", rep["control"], rep["treatment"], "Change",
                                                     "Worst allowed", "p", "Result"], g))
    if rep["segments"]:
        s = rep["segments"]
        rows = [[str(x["segment"]), f"{int(x['n']):,}", _pct(x["rel_lift"]), _p(x["p"]), _p(x["p_bh"])]
                for x in s["rows"]]
        out.append((f"{rep['primary']} by {s['by']} (do segments differ? p = {_p(s['heterogeneity_p'])})",
                    ["Segment", "Users", "Change", "p", "Adjusted p"], rows))
    return out


def to_json(rep) -> str:
    return json.dumps(plain(rep), indent=2)


def _chart(rep):
    """The effects chart as a base64 PNG, or None when matplotlib isn't installed."""
    try:
        from .plots import effects
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig = effects(rep["rows"], title=f"Change in {rep['treatment']} vs {rep['control']}, {1 - rep['alpha']:.0%} intervals")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def to_markdown(rep) -> str:
    lines = [f"# {rep['title']}", "", f"**Decision: {rep['decision']}.** {rep['why']}", ""]
    for name, head, rows in _tables(rep):
        lines += [f"## {name}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        lines.append("")
    if rep["caveats"]:
        lines += ["## Caveats", ""] + [f"* {c}" for c in rep["caveats"]] + [""]
    lines += ["## How the decision was made", "",
              "Split check first: if it fails, hold. Otherwise the primary metric decides: significantly better "
              "and every guardrail passes means ship, significantly worse means don't ship, anything else is no "
              "clear effect. Made with abkit.", ""]
    return "\n".join(lines)


_CSS = """
:root { --bg: #ffffff; --fg: #1d2330; --muted: #5d6677; --line: #e3e6ec; --accent: #2F4BC4;
        --ship: #2f7d4f; --hold: #a86b00; --stop: #b8322d; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171d; --fg: #e7e9ee; --muted: #9aa3b2; --line: #2a2f39; --accent: #8fa2ff;
          --ship: #6cc08f; --hold: #e0a83c; --stop: #f07a72; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 900px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 1.5em; margin: 0 0 12px; } h2 { font-size: 1.1em; margin: 28px 0 8px; }
.decision { border-left: 4px solid var(--accent); padding: 10px 14px; background: rgba(127,127,127,.08); }
.decision b { font-size: 1.1em; } .Ship { border-color: var(--ship); } .Hold, .Look { border-color: var(--hold); }
.Don { border-color: var(--stop); }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.92em; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; } li { margin: 4px 0; } .small { color: var(--muted); font-size: 0.85em; }
"""


def to_html(rep) -> str:
    e = _html.escape
    cls = rep["decision"].split()[0].replace("'", "")
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             f"<title>{e(rep['title'])}</title><style>{_CSS}</style></head><body>",
             f"<h1>{e(rep['title'])}</h1>",
             f"<div class='decision {e(cls)}'><b>Decision: {e(rep['decision'])}.</b> {e(rep['why'])}</div>"]
    chart = _chart(rep)
    for name, head, rows in _tables(rep):
        parts.append(f"<h2>{e(name)}</h2><div class='wrap'><table><thead><tr>"
                     + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr></thead><tbody>")
        parts += ["<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in row) + "</tr>" for row in rows]
        parts.append("</tbody></table></div>")
        if name == "Results" and chart:
            parts.append(f"<p><img alt='Change in each metric with its interval' style='max-width:100%;"
                         f"background:#fff;border-radius:6px' src='data:image/png;base64,{chart}'></p>")
    if rep["caveats"]:
        parts.append("<h2>Caveats</h2><ul>" + "".join(f"<li>{e(c)}</li>" for c in rep["caveats"]) + "</ul>")
    parts.append("<p class='small'>Split check first: if it fails, hold. Otherwise the primary metric decides: "
                 "significantly better and every guardrail passes means ship, significantly worse means don't ship, "
                 "anything else is no clear effect. Made with abkit.</p></body></html>")
    return "\n".join(parts)


def readout(exp, metrics, primary=None, guardrails=None, covariate=None, segment=None, cluster=None,
            higher_is_better=None, fmt: str = "markdown", **kw) -> str:
    """
    Full experiment readout as markdown (fmt="markdown"), HTML (fmt="html")
    or JSON for pipelines (fmt="json").
    guardrails maps metric -> worst acceptable relative change (0.01 = 1%).
    higher_is_better maps metric -> False for metrics where lower is better
    (load time, crashes).
    """
    rep = build(exp, metrics, primary, guardrails, covariate, segment, cluster, higher_is_better, **kw)
    if fmt == "markdown":
        return to_markdown(rep)
    if fmt == "html":
        return to_html(rep)
    if fmt == "json":
        return to_json(rep)
    raise ValueError("fmt must be 'markdown', 'html' or 'json'")
