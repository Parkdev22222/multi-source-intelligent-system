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


def _cap_torch_threads() -> None:
    """Stop PyTorch from fanning tiny tensors across every core on the box.

    The pairing head is ~30k parameters over batches of a few thousand rows.
    On a 252-core pod, the default `torch.set_num_threads(nproc)` spends all of
    its time in thread synchronisation: measured 21.5 s/epoch at 252 threads
    versus 0.05 s/epoch at 8 -- a 430x slowdown that looks exactly like a hang.
    Large ops (SAM3, the report LLM) run on the GPU and are unaffected by this
    cap, so there is nothing to trade off.

    Override with ICCE_TORCH_THREADS if a stage really is CPU-parallel.
    """
    import os

    try:
        import torch
    except ImportError:
        return

    try:
        requested = int(os.getenv("ICCE_TORCH_THREADS", "8"))
    except ValueError:
        requested = 8

    if requested > 0 and torch.get_num_threads() > requested:
        torch.set_num_threads(requested)


_cap_torch_threads()
