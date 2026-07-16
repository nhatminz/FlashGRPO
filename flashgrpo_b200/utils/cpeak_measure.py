import math


def recommended_cpeak(latencies: dict[int, float], max_latency_ratio: float = 1.25) -> int:
    """Pick the largest node count whose latency is close to the best observed latency."""
    if not latencies:
        return 16
    best = min(latencies.values())
    candidates = [nodes for nodes, latency in latencies.items() if latency <= best * max_latency_ratio]
    return max(candidates) if candidates else min(latencies, key=latencies.get)


def concurrency_budget(cpeak_nodes: int, active_batch: int, min_nodes: int, max_nodes: int) -> int:
    active_batch = max(1, int(active_batch))
    raw = math.floor(int(cpeak_nodes) / active_batch)
    return max(int(min_nodes), min(int(max_nodes), raw))
