"""
Command line interface. Run `abkit COMMAND --help` for the options.

    abkit report data.csv --variant version --control gate_30 --treatment gate_40 \
        --metrics retention_1 retention_7 --primary retention_7 --guardrail retention_1:0.01 \
        --html readout.html
    abkit report --config experiment.toml          # the same options kept in a file
    abkit plan --name "Gate 40" --primary retention_7 --baseline 0.19 --relative-mde 0.05 --out plan.md
    abkit size --baseline 0.19 --mde 0.01 --daily-users 4000
    abkit aa data.csv --metric revenue
    abkit presets

Data can be CSV, TSV, Parquet, Excel or JSON. Any option can also sit in a
config file (TOML or JSON) and the command line wins over the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from . import __version__
from .core import fmt_p, sample_size
from .data import InputError, plain, read_table
from .design import aa_test, mde, ratio_sd, run_length
from .experiment import Experiment
from .presets import describe, resolve_guardrail


def _load(a) -> pd.DataFrame:
    if not a.data:
        raise InputError("no data file given; pass one or set data in the config file")
    path = Path(a.data)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls") and getattr(a, "sheet", None) is not None:
        df = pd.read_excel(path, sheet_name=a.sheet)
    else:
        df = read_table(path)
    if getattr(a, "where", None):
        try:
            df = df.query(a.where)
        except Exception as e:
            raise InputError(f"couldn't apply --where {a.where!r}: {e}") from e
        if df.empty:
            raise InputError(f"--where {a.where!r} leaves no rows")
    return df


def _guardrails(specs) -> dict:
    """METRIC:MARGIN or METRIC:PRESET strings (or a dict from a config file) to {metric: margin or preset}."""
    if isinstance(specs, dict):
        return dict(specs)
    out = {}
    for g in specs or []:
        name, sep, margin = str(g).partition(":")
        if not sep or not margin:
            raise InputError(f"--guardrail wants METRIC:MARGIN or METRIC:PRESET, like retention_1:0.01 "
                             f"or load_ms:latency, not {g!r}")
        resolve_guardrail(margin)
        out[name] = margin
    return out


def _emit(text: str, target: str) -> None:
    """Write to a file, or to stdout when the target is a dash."""
    if target == "-":
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    else:
        Path(target).write_text(text, encoding="utf-8")


def _report(a) -> int:
    from . import report as rp

    df = _load(a)
    guard = _guardrails(a.guardrail)
    lower = {m: False for m in a.lower_is_better or []}
    exp = Experiment(df, a.variant, a.control, a.treatment, a.split, a.alpha)
    metrics = list(dict.fromkeys([*([a.primary] if a.primary else []), *a.metrics]))
    kw = dict(primary=a.primary, guardrails=guard or None, covariate=a.covariate, segment=a.segment,
              cluster=a.cluster, higher_is_better=lower, prior_sd=a.prior_sd, srm_threshold=a.srm_threshold)
    outputs = [(a.markdown, "markdown"), (a.html, "html"), (a.json, "json")]
    if any(path for path, _ in outputs) or a.plot:
        rep = rp.build(exp, metrics, **{**kw, "title": a.title})
        to_stdout = False
        for path, fmt in outputs:
            if path:
                text = {"markdown": rp.to_markdown, "html": rp.to_html, "json": rp.to_json}[fmt](rep)
                _emit(text, path)
                to_stdout |= path == "-"
                if path != "-":
                    print(f"wrote {path}", file=sys.stderr if to_stdout else sys.stdout)
        if a.plot:
            from .plots import effects
            effects(rep["rows"], title=f"Change in {a.treatment} vs {a.control}").savefig(a.plot, dpi=150)
            print(f"wrote {a.plot}", file=sys.stderr if to_stdout else sys.stdout)
        return 0
    return _print_report(a, exp, metrics, kw, guard)


def _print_report(a, exp, metrics, kw, guard) -> int:
    from . import report as rp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = exp.check(pre_period=a.covariate, srm_threshold=a.srm_threshold, unit=a.cluster)
    print(f"{a.control}: {h['n_control']:,} users   {a.treatment}: {h['n_treatment']:,} users")
    print(f"sample ratio check: {fmt_p(h['srm_p'])}"
          + ("   MISMATCH, investigate before trusting results" if caught else "   ok"))
    for col, p in (h.get("balance") or {}).items():
        print(f"balance on {col}: {fmt_p(p)}" + ("   unbalanced" if p < 0.001 else "   ok"))
    print()

    for m in metrics:
        print(exp.analyze(m, a.covariate, cluster=a.cluster, dropna=True))
        if a.prior_sd is not None and a.cluster is None:
            b = exp.bayes(m, a.prior_sd)
            print(f"  {b.chance_treatment_better:.1%} chance {a.treatment} is better; "
                  f"expected loss if you ship it {b.expected_loss_ship:.4g}")
        print()
    if len(metrics) > 1:
        t = exp.analyze_many(metrics, correction=a.correction)
        print(f"adjusted for testing {len(metrics)} metrics ({a.correction}):")
        for _, r in t.iterrows():
            print(f"  {r['metric']:<24} {fmt_p(r['p'])}   {fmt_p(r[f'p_{a.correction}'], 'adjusted p')}")
        print()
    if guard:
        g = exp.guardrails(guard, kw["higher_is_better"])
        print("guardrails (non-inferiority):")
        for _, r in g.iterrows():
            print(f"  {r['metric']:<24} change {r['rel_lift']:+.2%}   worst allowed {r['margin']:.2%}   "
                  f"{fmt_p(r['p'])}   {'pass' if r['passes'] else 'not shown safe'}")
        print()
    if a.segment:
        for m in metrics:
            table, het = exp.segments(m, a.segment, a.correction, a.covariate)
            print(f"{m} by {a.segment} (do segments differ? {fmt_p(het)})")
            for _, r in table.iterrows():
                print(f"  {str(r['segment']):<16} n = {int(r['n']):>9,}   lift {r['rel_lift']:+.2%}   "
                      f"{fmt_p(r[f'p_{a.correction}'], 'adjusted p')}")
            print()
    rep = rp.build(exp, metrics, **kw)
    print(f"decision: {rep['decision']}. {rep['why']}")
    if a.prior_sd is None:
        print("(add --prior-sd to see the chance the treatment is better; a flat prior overstates it)")
    return 0


def _plan_sd(a):
    """Per user sd from --sd, --baseline, or a pilot file for a ratio metric."""
    if a.pilot is not None:
        if not (a.numerator and a.denominator):
            raise InputError("--pilot needs --numerator and --denominator columns")
        df = read_table(a.pilot)
        from .data import need_columns
        need_columns(df, [a.numerator, a.denominator], numeric=True)
        return ratio_sd(df[a.numerator], df[a.denominator]), len(df)
    return None, None


def _size(a) -> int:
    given = sum(x is not None for x in (a.baseline, a.sd, a.pilot))
    if given != 1:
        raise InputError("give one of --baseline (a conversion rate), --sd (for an average) "
                         "or --pilot (a file for a ratio metric)")
    if sum(x is not None for x in (a.mde, a.relative_mde, a.users)) != 1:
        raise InputError("give one of --mde or --relative-mde (to get users) or --users "
                         "(to get the smallest detectable change)")
    if a.relative_mde is not None and a.baseline is None:
        raise InputError("--relative-mde needs --baseline, the current value of the metric")
    pilot_sd, n_pilot = _plan_sd(a)
    sd = pilot_sd if pilot_sd is not None else (a.sd if a.sd is not None else (a.baseline * (1 - a.baseline)) ** 0.5)
    out = {"alpha": a.alpha, "power": a.power}
    lines = []
    if pilot_sd is not None:
        lines.append(f"per user sd of {a.numerator} per {a.denominator}: {sd:.4g} (delta method, from {n_pilot:,} users)")
    if a.cuped_rho is not None:
        if not 0 <= abs(a.cuped_rho) < 1:
            raise InputError("--cuped-rho is a correlation, between -1 and 1")
        sd *= (1 - a.cuped_rho**2) ** 0.5
        lines.append(f"with CUPED at correlation {a.cuped_rho:g} the sd drops to {sd:.4g}")
        out["cuped_rho"] = a.cuped_rho
    out["sd"] = sd
    if a.users is not None:
        d = mde(sd, a.users, a.users, a.alpha, a.power)
        rel = f" ({d / a.baseline:.1%} relative)" if a.baseline else ""
        lines.append(f"with {a.users:,} users per group the smallest change you can detect with {a.power:.0%} power "
                     f"is {d:.4g}{rel}")
        n = a.users
        out.update(users_per_group=n, mde=d)
    else:
        change = a.mde if a.mde is not None else round(a.relative_mde * a.baseline, 12)
        n, _ = sample_size(sd, change, a.alpha, a.power)
        lines.append(f"{n:,} users per group ({2 * n:,} total) to detect a change of {change:.4g}"
                     + (f" ({a.relative_mde:.1%} relative)" if a.relative_mde is not None else "")
                     + f" with {a.power:.0%} power at alpha {a.alpha}")
        out.update(users_per_group=n, users_total=2 * n, mde=change)
    if a.daily_users:
        rp = run_length(n, a.daily_users, traffic_share=a.traffic_share)
        lines.append(str(rp))
        out.update(days=rp.days, days_recommended=rp.days_recommended)
    if a.json:
        _emit(json.dumps(plain(out), indent=2), a.json)
        if a.json == "-":
            return 0
    print("\n".join(lines))
    return 0


def _plan(a) -> int:
    from . import plan as pl

    pilot_sd, _ = _plan_sd(a)
    p = pl.build(a.name, a.primary, hypothesis=a.hypothesis, baseline=a.baseline,
                 sd=pilot_sd if pilot_sd is not None else a.sd, mde=a.mde, relative_mde=a.relative_mde,
                 alpha=a.alpha, power=a.power, split=a.split, daily_users=a.daily_users,
                 traffic_share=a.traffic_share, secondary=a.secondary, guardrails=_guardrails(a.guardrail),
                 higher_is_better={m: False for m in a.lower_is_better or []}, covariate=a.covariate,
                 cuped_rho=a.cuped_rho, cluster=a.cluster, segments=a.segment, sequential=a.sequential,
                 looks=a.looks, owner=a.owner, date=a.date)
    if a.json:
        _emit(json.dumps(plain(p), indent=2), a.json)
    text = pl.to_markdown(p)
    if a.out:
        _emit(text, a.out)
        if a.out != "-":
            print(f"wrote {a.out}", file=sys.stderr if a.json == "-" else sys.stdout)
    elif a.json != "-":
        print(text)
    return 0


def _aa(a) -> int:
    from .data import need_columns
    df = _load(a)
    need_columns(df, [a.metric], numeric=True)
    need_columns(df, [a.cluster] if a.cluster else [])
    y = df[a.metric].astype(float).to_numpy()
    g = df[a.cluster].to_numpy() if a.cluster else None
    res = aa_test(y, a.sims, clusters=g, seed=a.seed)
    if a.json:
        _emit(json.dumps(plain({"sims": res.sims, "false_positive_rate": res.false_positive_rate,
                                 "mc_se": res.mc_se, "uniform_p": res.uniform_p}), indent=2), a.json)
        if a.json == "-":
            return 0
    print(res)
    return 0


def _presets(a) -> int:
    print(describe())
    return 0


def _parser():
    p = argparse.ArgumentParser(prog="abkit", description="Analyze and plan A/B tests.")
    p.add_argument("--version", action="version", version=f"abkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    cfg = "options file (TOML or JSON); anything on the command line overrides it"

    def data_args(x):
        x.add_argument("data", nargs="?", help="CSV, TSV, Parquet, Excel or JSON file, one row per user")
        x.add_argument("--config", metavar="FILE", help=cfg)
        x.add_argument("--sheet", help="Excel sheet name or number (default: the first)")
        x.add_argument("--where", metavar="EXPR", help="keep only rows matching a pandas query, like \"country == 'US'\"")

    r = sub.add_parser("report", help="analyze an experiment and write a readout")
    data_args(r)
    r.add_argument("--variant", help="column with the group name")
    r.add_argument("--control")
    r.add_argument("--treatment")
    r.add_argument("--metrics", nargs="+")
    r.add_argument("--covariate", nargs="+", help="columns measured before the test, for CUPED")
    r.add_argument("--segment", help="column to break results down by")
    r.add_argument("--correction", choices=["bh", "holm", "none"], default="bh")
    r.add_argument("--split", type=float, default=0.5, help="intended share in treatment")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--prior-sd", type=float, default=None,
                   help="spread of lifts in past tests, for the Bayesian readout")
    r.add_argument("--primary", help="the metric the decision is based on (default: the first metric)")
    r.add_argument("--guardrail", nargs="+", metavar="METRIC:MARGIN",
                   help="metrics that must not get worse by more than MARGIN (relative, 0.01 = 1%%), "
                        "or a preset name like load_ms:latency (see abkit presets)")
    r.add_argument("--lower-is-better", nargs="+", metavar="METRIC", help="metrics where down is good")
    r.add_argument("--cluster", help="column that was randomized, if rows aren't independent (user id)")
    r.add_argument("--srm-threshold", type=float, default=0.01)
    r.add_argument("--title", help="title for the readout")
    r.add_argument("--markdown", metavar="FILE", help="write the readout as markdown (- for the terminal)")
    r.add_argument("--html", metavar="FILE", help="write the readout as one HTML file")
    r.add_argument("--json", metavar="FILE", help="write every number as JSON (- for the terminal)")
    r.add_argument("--plot", metavar="FILE", help="save a chart of each metric's change with its interval")
    r.set_defaults(func=_report, need=["data", "variant", "control", "treatment", "metrics"])

    s = sub.add_parser("size", help="how many users a test needs, or what it can detect")
    s.add_argument("--config", metavar="FILE", help=cfg)
    s.add_argument("--baseline", type=float, help="current conversion rate, like 0.19")
    s.add_argument("--sd", type=float, help="standard deviation, for an average metric")
    s.add_argument("--pilot", help="data file with one row per user, for a ratio metric")
    s.add_argument("--numerator", help="ratio numerator column in the pilot file (clicks)")
    s.add_argument("--denominator", help="ratio denominator column in the pilot file (sessions)")
    s.add_argument("--mde", type=float, help="smallest absolute change you care about")
    s.add_argument("--relative-mde", type=float, help="smallest relative change, 0.05 for 5%% (needs --baseline)")
    s.add_argument("--users", type=int, help="users per group you have; prints the smallest detectable change")
    s.add_argument("--cuped-rho", type=float, help="expected correlation with the CUPED covariate")
    s.add_argument("--daily-users", type=float, help="new users a day, to plan how long the test runs")
    s.add_argument("--traffic-share", type=float, default=1.0, help="share of traffic in the test")
    s.add_argument("--power", type=float, default=0.8)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--json", metavar="FILE", help="write the answer as JSON (- for the terminal)")
    s.set_defaults(func=_size, need=[])

    n = sub.add_parser("plan", help="write the plan for a test before it runs")
    n.add_argument("--config", metavar="FILE", help=cfg)
    n.add_argument("--name", help="what the experiment is called")
    n.add_argument("--primary", help="the metric the decision is based on")
    n.add_argument("--hypothesis", help="the question, in a sentence")
    n.add_argument("--baseline", type=float, help="current value of a conversion rate")
    n.add_argument("--sd", type=float, help="standard deviation, for an average metric")
    n.add_argument("--pilot", help="data file with one row per user, for a ratio metric")
    n.add_argument("--numerator")
    n.add_argument("--denominator")
    n.add_argument("--mde", type=float, help="smallest absolute change worth finding")
    n.add_argument("--relative-mde", type=float, help="smallest relative change, 0.05 for 5%% (needs --baseline)")
    n.add_argument("--alpha", type=float, default=0.05)
    n.add_argument("--power", type=float, default=0.8)
    n.add_argument("--split", type=float, default=0.5, help="share of users in treatment")
    n.add_argument("--daily-users", type=float)
    n.add_argument("--traffic-share", type=float, default=1.0)
    n.add_argument("--secondary", nargs="+", help="other metrics you will report")
    n.add_argument("--guardrail", nargs="+", metavar="METRIC:MARGIN")
    n.add_argument("--lower-is-better", nargs="+", metavar="METRIC")
    n.add_argument("--covariate", nargs="+", help="pre period columns for CUPED")
    n.add_argument("--cuped-rho", type=float, help="expected correlation with the covariate")
    n.add_argument("--cluster", help="column that is randomized, if rows come in groups")
    n.add_argument("--segment", nargs="+", help="columns you will break results down by")
    n.add_argument("--sequential", choices=["group", "msprt"], help="plan to check results before the end")
    n.add_argument("--looks", type=int, default=5, help="planned looks for a group sequential test")
    n.add_argument("--owner")
    n.add_argument("--date", help="date to print (default: today)")
    n.add_argument("--out", metavar="FILE", help="write the plan as markdown (default: print it)")
    n.add_argument("--json", metavar="FILE", help="also write the plan as JSON (- for the terminal)")
    n.set_defaults(func=_plan, need=["name", "primary"])

    q = sub.add_parser("aa", help="A/A check: does the t test behave on this metric?")
    data_args(q)
    q.add_argument("--metric")
    q.add_argument("--cluster", help="column that is randomized, if rows come in groups")
    q.add_argument("--sims", type=int, default=1000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--json", metavar="FILE", help="write the answer as JSON (- for the terminal)")
    q.set_defaults(func=_aa, need=["data", "metric"])

    pr = sub.add_parser("presets", help="list the guardrail presets")
    pr.set_defaults(func=_presets, need=[])
    return p, sub


def _read_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise InputError(f"no config file at {path}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(p, "rb") as f:
        return tomllib.load(f)


def _apply_config(parser_for_cmd, all_dests, cmd, path):
    """Turn config file keys into argparse defaults for one command."""
    raw = _read_config(path)
    section = raw.pop(cmd, {}) if isinstance(raw.get(cmd), dict) else {}
    for k in [k for k, v in raw.items() if isinstance(v, dict) and k not in ("guardrail", "guardrails")]:
        raw.pop(k)                      # tables for other commands
    raw.update(section)
    mine = {act.dest for act in parser_for_cmd._actions}
    out = {}
    for key, val in raw.items():
        dest = key.replace("-", "_")
        if dest == "guardrails":
            dest = "guardrail"
        if dest in mine:
            out[dest] = val
        elif dest not in all_dests:
            raise InputError(f"unknown option {key!r} in {path}")
    for k in ("metrics", "covariate", "lower_is_better", "secondary"):
        if isinstance(out.get(k), str):
            out[k] = [out[k]]
    return out


def main(argv=None) -> int:
    parser, sub = _parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        cmd = next((x for x in argv if x in sub.choices), None)
        if cmd and "--config" in argv:
            i = argv.index("--config")
            if i + 1 < len(argv):
                dests = {act.dest for sp in sub.choices.values() for act in sp._actions}
                sub.choices[cmd].set_defaults(**_apply_config(sub.choices[cmd], dests, cmd, argv[i + 1]))
        a = parser.parse_args(argv)
        for name in a.need:
            if getattr(a, name, None) in (None, []):
                flag = "the data file" if name == "data" else f"--{name.replace('_', '-')}"
                raise InputError(f"{flag} is required (on the command line or in the config file)")
        return a.func(a)
    except BrokenPipeError:
        return 0
    except (InputError, KeyError, ValueError, OSError, ImportError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
