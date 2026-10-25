"""
Starting points for guardrails, so a team doesn't argue the margin from
scratch every test. Each preset is the worst relative change you'd accept
and which direction is bad. They are defaults to edit, not rules; the right
margin depends on what a change costs you.

    exp.guardrails({"page_load_ms": "latency", "crash_free_rate": "reliability"})
    abkit report ... --guardrail page_load_ms:latency
"""

from __future__ import annotations

GUARDRAILS = {
    "latency": (0.05, False, "page or request time, 5% slower is the most I'd accept"),
    "errors": (0.10, False, "error or failure rate, 10% more is the most I'd accept"),
    "crashes": (0.05, False, "crashes per session, 5% more is the most I'd accept"),
    "unsubscribes": (0.10, False, "opt outs, complaints or refunds, 10% more is the most I'd accept"),
    "retention": (0.01, True, "return rate, 1% fewer returning users is the most I'd accept"),
    "reliability": (0.005, True, "success or crash free rate, 0.5% lower is the most I'd accept"),
    "engagement": (0.02, True, "sessions, clicks or time spent, 2% lower is the most I'd accept"),
    "revenue": (0.02, True, "revenue per user, 2% lower is the most I'd accept"),
}


def resolve_guardrail(spec) -> tuple[float, bool | None]:
    """A margin as a number or a preset name. Returns (margin, higher_is_better), None if the spec doesn't say."""
    if isinstance(spec, str):
        try:
            margin, hib, _ = GUARDRAILS[spec.strip().lower()]
        except KeyError:
            try:
                return float(spec), None
            except ValueError:
                raise ValueError(f"{spec!r} is neither a number nor a guardrail preset; "
                                 f"presets are {', '.join(GUARDRAILS)}") from None
        return margin, hib
    return float(spec), None


def describe() -> str:
    width = max(map(len, GUARDRAILS))
    return "\n".join(f"{name:<{width}}  {m:>6.1%}  {'higher is better' if hib else 'lower is better':<17} {text}"
                     for name, (m, hib, text) in GUARDRAILS.items())
