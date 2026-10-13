"""abkit: statistics for online A/B tests."""

from .core import (
    TestResult,
    cuped_adjust,
    heterogeneity_test,
    mean_test,
    proportion_sample_size,
    ratio_test,
    sample_size,
    srm_test,
)
from .sequential import SequentialResult, msprt, sequential_test
from .multiple import benjamini_hochberg, holm
from .bayes import BayesResult, bayes_test
from .experiment import Experiment

__all__ = [
    "TestResult", "mean_test", "ratio_test", "cuped_adjust", "srm_test", "heterogeneity_test",
    "sample_size", "proportion_sample_size",
    "SequentialResult", "msprt", "sequential_test",
    "holm", "benjamini_hochberg",
    "BayesResult", "bayes_test",
    "Experiment",
]
__version__ = "0.1.0"
