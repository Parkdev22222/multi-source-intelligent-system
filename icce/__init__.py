"""
ICCE-Asia 2026 experiment harness for MSIS.

This package is additive: it never modifies the behaviour of the production
pipeline. It provides
  - datasets/  : loaders for public bi-temporal change-detection benchmarks
  - convert/   : benchmark <-> MSIS pipeline adapters (geo-referencing, instances)
  - metrics/   : pixel / instance / caption / factuality metrics
  - pairing_head/ : the learned instance-pairing head (contribution C2)
  - report/    : grounding-ablation report generation (contribution C3)
  - eval/      : experiment runners that emit JSON + LaTeX tables
"""

__all__ = ["datasets", "convert", "metrics", "pairing_head", "report", "eval"]
