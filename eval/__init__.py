"""Small, dependency-free evaluation harness for the Socratic tutor."""

from .judge import JudgeClient, JudgeResult
from .runner import Evaluation, Transcript, evaluate_cases
from .scoring import score_compliance, score_utility

__all__ = [
    "Evaluation",
    "JudgeClient",
    "JudgeResult",
    "Transcript",
    "evaluate_cases",
    "score_compliance",
    "score_utility",
]
