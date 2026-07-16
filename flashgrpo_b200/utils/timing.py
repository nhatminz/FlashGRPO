import time
from contextlib import contextmanager


def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}m"
    return f"{seconds / 3600:.2f}h"


@contextmanager
def elapsed_timer(sync_cuda: bool = False):
    if sync_cuda:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    start = time.time()
    box = {"elapsed": 0.0}
    try:
        yield box
    finally:
        if sync_cuda:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        box["elapsed"] = time.time() - start
