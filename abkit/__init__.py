"""abkit: statistics for online A/B tests."""

__version__ = "0.3.0"

from .core import (
    TestResult,
    adjusted_test,
    cluster_test,
    cupac_adjust,
    cuped_adjust,
    heterogeneity_test,
    mean_test,
    noninferiority_test,
    proportion_sample_size,
    ratio_test,
    sample_size,
    srm_test,
)
from .sequential import (
    GroupSequentialDesign,
    GroupSequentialResult,
    SequentialResult,
    group_sequential_design,
    group_sequential_test,
    msprt,
    sequential_test,
)
from .multiple import benjamini_hochberg, dunnett, holm
from .bayes import BayesResult, bayes_test
from .design import AAResult, RunPlan, aa_test, mde, proportion_mde, ratio_mde, ratio_sample_size, ratio_sd, run_length
from .robust import RankResult, bootstrap_test, mann_whitney, quantile_test, winsorize
from .overtime import TimeChecks, time_checks, trend_test
from .experiment import Experiment
from .data import InputError, read_table
from .presets import GUARDRAILS
from .plan import build as build_plan, to_markdown as plan_markdown

__all__ = [
    "TestResult", "mean_test", "ratio_test", "cluster_test", "cuped_adjust", "cupac_adjust", "adjusted_test",
    "noninferiority_test", "srm_test", "heterogeneity_test",
    "sample_size", "proportion_sample_size", "mde", "proportion_mde", "ratio_sd", "ratio_sample_size",
    "ratio_mde", "run_length", "RunPlan", "aa_test", "AAResult",
    "SequentialResult", "msprt", "sequential_test",
    "GroupSequentialDesign", "GroupSequentialResult", "group_sequential_design", "group_sequential_test",
    "holm", "benjamini_hochberg", "dunnett",
    "BayesResult", "bayes_test",
    "bootstrap_test", "quantile_test", "mann_whitney", "winsorize", "RankResult",
    "time_checks", "trend_test", "TimeChecks",
    "Experiment", "InputError", "read_table", "GUARDRAILS", "build_plan", "plan_markdown",
]
