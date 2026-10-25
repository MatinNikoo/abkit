"""
A written plan for a test, made before it runs: the question, the
metrics, the sample size, how it will be analyzed, and what decision each
result leads to. Writing the rule down first is what stops a result from
choosing its own story afterward.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from .core import sample_size
from .design import run_length
from .presets import resolve_guardrail
from .sequential import group_sequential_design


def _list(x):
    return [] if x is None else ([x] if isinstance(x, str) else list(x))


def build(name, primary, hypothesis=None, baseline=None, sd=None, mde=None, relative_mde=None,
          alpha=0.05, power=0.8, split=0.5, daily_users=None, traffic_share=1.0, secondary=None,
          guardrails=None, higher_is_better=None, covariate=None, cuped_rho=None, cluster=None,
          segments=None, sequential=None, looks=5, owner=None, date=None) -> dict:
    """
    Work out the design and collect every choice into one dict (see to_markdown).

    Give the primary metric's spread as baseline (a conversion rate) or sd
    (an average, or a ratio metric's per user sd from ratio_sd), and the
    smallest change worth finding as mde (absolute) or relative_mde.
    cuped_rho is the correlation you expect between the covariate and the
    metric; it shrinks the sample by 1 - rho squared. sequential is None,
    "group" (planned looks, O'Brien Fleming) or "msprt" (look any time).
    """
    if (baseline is None) == (sd is None):
        raise ValueError("give either baseline (a conversion rate) or sd (an average's standard deviation)")
    if (mde is None) == (relative_mde is None):
        raise ValueError("give either mde (absolute) or relative_mde (0.05 is 5%)")
    if relative_mde is not None:
        if baseline is None:
            raise ValueError("relative_mde needs baseline, the metric's current value")
        mde = round(relative_mde * baseline, 12)
    if not 0 < split < 1:
        raise ValueError("split is the share of users in treatment, between 0 and 1")
    if cuped_rho is not None and not 0 <= abs(cuped_rho) < 1:
        raise ValueError("cuped_rho is a correlation, between -1 and 1")
    if sequential not in (None, "group", "msprt"):
        raise ValueError("sequential is None, 'group' or 'msprt'")
    ratio = split / (1 - split)
    eff_sd = np.sqrt(baseline * (1 - baseline)) if sd is None else sd
    keep = 1.0 if cuped_rho is None else 1 - cuped_rho**2
    eff_sd = eff_sd * np.sqrt(keep)
    nc, nt = sample_size(eff_sd, mde, alpha, power, ratio)
    out = {"name": name, "hypothesis": hypothesis, "primary": primary, "owner": owner,
           "date": date or _dt.date.today().isoformat(), "alpha": alpha, "power": power, "split": split,
           "baseline": baseline, "sd": sd, "mde": mde, "relative_mde": relative_mde if relative_mde is not None
           else (mde / baseline if baseline else None),
           "cuped_rho": cuped_rho, "covariate": _list(covariate), "cluster": cluster,
           "secondary": _list(secondary), "segments": _list(segments), "sequential": sequential,
           "higher_is_better": {primary: True, **(higher_is_better or {})}}
    out["n_control"], out["n_treatment"] = nc, nt
    guards = []
    for m, spec in (guardrails or {}).items():
        margin, hib = resolve_guardrail(spec)
        if hib is None:
            hib = out["higher_is_better"].get(m, True)
        guards.append({"metric": m, "margin": margin, "higher_is_better": hib})
    out["guardrails"] = guards
    if sequential == "group":
        d = group_sequential_design(looks, alpha)
        infl = d.inflation(power)
        out["looks"] = {"info": d.info.tolist(), "bounds": d.bounds.tolist(), "inflation": infl,
                        "n_control": int(np.ceil(nc * infl)), "n_treatment": int(np.ceil(nt * infl)),
                        "expected_share_if_effect": d.expected_sample(_drift(d, power)),
                        "expected_share_if_none": d.expected_sample(0.0)}
    nmax = out["looks"]["n_control"] + out["looks"]["n_treatment"] if sequential == "group" else nc + nt
    out["n_total"] = nmax
    if daily_users:
        rp = run_length(nmax // 2, daily_users, 2, traffic_share)
        out["days"], out["days_recommended"] = rp.days, rp.days_recommended
    out["daily_users"], out["traffic_share"] = daily_users, traffic_share
    return out


def _drift(design, power):
    from scipy import optimize
    return optimize.brentq(lambda d: design.power(d) - power, 0.1, 20, xtol=1e-10)


def to_markdown(p: dict) -> str:
    a, rel = p["alpha"], p["relative_mde"]
    hib = p["higher_is_better"]
    L = [f"# Experiment plan: {p['name']}", "",
         f"Written {p['date']}" + (f" by {p['owner']}" if p["owner"] else "") +
         ", before any results were looked at. If the plan changes after launch, add a line under "
         "Changes below with the date and the reason.", ""]
    L += ["## Question", "", p["hypothesis"] or f"Does the change move {p['primary']}?", ""]

    L += ["## Metrics", "", "| Role | Metric | Good direction | How it is judged |", "|---|---|---|---|"]
    L.append(f"| Primary | {p['primary']} | {'up' if hib[p['primary']] else 'down'} | "
             f"its own p value at alpha {a}, chosen before the test |")
    for m in p["secondary"]:
        L.append(f"| Secondary | {m} | {'up' if hib.get(m, True) else 'down'} | "
                 "Benjamini Hochberg adjusted p value across secondary metrics |")
    for g in p["guardrails"]:
        L.append(f"| Guardrail | {g['metric']} | {'up' if g['higher_is_better'] else 'down'} | "
                 f"non-inferiority, worst allowed change {g['margin']:.1%} "
                 f"{'lower' if g['higher_is_better'] else 'higher'} |")
    L.append("")

    unit = f"clusters of {p['cluster']}" if p["cluster"] else "users"
    L += ["## Size and length", ""]
    L.append(f"* Smallest change worth finding on {p['primary']}: {p['mde']:.4g}"
             + (f" ({rel:.1%} relative to {p['baseline']:.4g})" if rel and p["baseline"] else "")
             + f", with {p['power']:.0%} power at alpha {a}, two sided.")
    if p["baseline"] is not None:
        L.append(f"* Baseline {p['primary']}: {p['baseline']:.4g}, so the per user sd is "
                 f"{np.sqrt(p['baseline'] * (1 - p['baseline'])):.4g}.")
    else:
        L.append(f"* Per user sd of {p['primary']}: {p['sd']:.4g}.")
    if p["cuped_rho"] is not None:
        L.append(f"* Planned CUPED on {', '.join(p['covariate']) or 'a pre period metric'} with an expected correlation "
                 f"of {p['cuped_rho']:.2f}, which cuts the sample by {p['cuped_rho'] ** 2:.0%}. If the real correlation "
                 "comes out lower, the test will be underpowered by that much.")
    L.append(f"* Split: {1 - p['split']:.0%} control, {p['split']:.0%} treatment.")
    L.append(f"* Fixed sample: {p['n_control']:,} control and {p['n_treatment']:,} treatment {unit} "
             f"({p['n_control'] + p['n_treatment']:,} in all).")
    if p["sequential"] == "group":
        lk = p["looks"]
        L.append(f"* Planned looks: {len(lk['info'])} evenly spaced, O'Brien Fleming boundaries (Lan DeMets). "
                 f"The most it can need is {lk['n_control'] + lk['n_treatment']:,} {unit}, "
                 f"{lk['inflation']:.3f} times the fixed sample. If the effect is the size above it uses about "
                 f"{lk['expected_share_if_effect']:.0%} of that on average, and about "
                 f"{lk['expected_share_if_none']:.0%} when nothing changes.")
        L += ["", "| Look | Share of maximum sample | Stop if abs(z) is at least |", "|---|---|---|"]
        L += [f"| {i + 1} | {t:.0%} | {b:.3f} |" for i, (t, b) in enumerate(zip(lk["info"], lk["bounds"]))]
        L.append("")
    elif p["sequential"] == "msprt":
        L.append(f"* Checked whenever anyone likes with an always valid (mSPRT) test tuned to {p['mde']:.4g}. "
                 "The p value is valid at every look. It can stop early on a big effect, and it needs somewhat "
                 f"more than the fixed sample to match its power, so treat {p['n_control'] + p['n_treatment']:,} as "
                 "a floor and set a cap before launch.")
    if p.get("days"):
        L.append(f"* At {p['daily_users']:,.0f} new users a day and {p['traffic_share']:.0%} of traffic in the test, "
                 f"that is {p['days']} days. Plan for {p['days_recommended']} so every day of the week shows up "
                 "equally often.")
    else:
        L.append("* Length is not set yet; it depends on how many new users arrive each day.")
    L.append("")

    L += ["## Analysis", ""]
    if p["sequential"] == "group":
        L.append("* Primary test: the Welch z score at each planned look against the boundaries above, "
                 "with repeated confidence intervals that stay valid at every look.")
    elif p["sequential"] == "msprt":
        L.append("* Primary test: the mSPRT always valid p value and confidence sequence.")
    else:
        L.append("* Primary test: Welch's t test on the difference in means"
                 + (", run on cluster means so repeated rows of one unit don't count as independent"
                    if p["cluster"] else "") + f". Intervals are {1 - a:.0%}.")
    if p["covariate"]:
        L.append(f"* Variance reduction: CUPED on {', '.join(p['covariate'])}, fit on both groups pooled.")
    if p["secondary"]:
        L.append("* Secondary metrics get Benjamini Hochberg adjusted p values. Only the primary metric makes the decision.")
    if p["segments"]:
        L.append(f"* Segments: {', '.join(p['segments'])}. Each gets an adjusted p value and one test of whether "
                 "the segments really differ. A segment result is a lead for the next test, not a finding.")
    L.append("* Nothing outside this list is part of the plan. Anything else that gets looked at is labeled exploratory.")
    L.append("")

    L += ["## Checks before reading any result", "",
          "* Sample ratio check on the planned split. A p value under 0.01 means hold and investigate.",
          "* Balance on pre period columns" + (f" ({', '.join(p['covariate'])})" if p["covariate"] else "") + ".",
          "* Missing values: does one group lose more of a metric than the other.",
          ""]

    L += ["## Decision rule", "",
          "* Split check fails: hold.",
          f"* {p['primary']} significantly {'better' if hib[p['primary']] else 'lower'} and every guardrail passes: ship.",
          f"* {p['primary']} better but a guardrail not shown safe: look closer before shipping.",
          f"* {p['primary']} significantly worse: don't ship.",
          f"* Anything else: no clear effect. The test can only rule out changes bigger than about {p['mde']:.4g}.",
          "", "## Changes", "", "None yet.", ""]
    return "\n".join(L)
