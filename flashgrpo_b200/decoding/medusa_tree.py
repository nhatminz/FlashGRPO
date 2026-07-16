from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch


@dataclass
class CandidateTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.children: dict[int, list[int]] = {}
        for idx, parent in enumerate(self.parents):
            if parent >= 0:
                self.children.setdefault(parent, []).append(idx)

    def ancestors_including_self(self, node_idx: int) -> list[int]:
        out = [node_idx]
        parent = self.parents[node_idx]
        while parent >= 0:
            out.append(parent)
            parent = self.parents[parent]
        return list(reversed(out))

    @property
    def node_count(self) -> int:
        return len(self.tokens)


@dataclass
class TreePlan:
    node_budget_per_seq: int
    active_heads: int
    topk_by_depth: list[int]
    actual_nodes: int
    mode: str
    layout: str


def dense_node_count(topk_by_depth: list[int]) -> int:
    total = 1
    parents = 1
    for k in topk_by_depth:
        parents *= int(k)
        total += parents
    return total


def plan_tree(
    *,
    active_batch_size: int,
    num_medusa_heads: int,
    tree_mode: str,
    tree_layout: str,
    cpeak_nodes: int,
    min_tree_nodes_per_seq: int,
    max_tree_nodes_per_seq: int,
    max_tree_depth: int,
    fixed_tree_topk_by_depth: list[int],
) -> TreePlan:
    if tree_layout not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported tree_layout={tree_layout}")
    if tree_layout == "sparse":
        # TODO: Replace this compatibility mapping with score-pruned sparse
        # prefix trees. Dense keeps the main exact acceptance path runnable.
        warnings.warn("flashgrpo tree_layout='sparse' is a v1 skeleton; falling back to dense.", RuntimeWarning)
        tree_layout = "dense"
    active_batch_size = max(1, int(active_batch_size))
    if tree_mode == "fixed":
        budget = int(max_tree_nodes_per_seq)
        topk = [max(1, int(k)) for k in fixed_tree_topk_by_depth[:num_medusa_heads]]
        while topk and dense_node_count(topk) > budget:
            topk[-1] -= 1
            if topk[-1] <= 0:
                topk.pop()
        active_heads = min(len(topk), max(0, int(max_tree_depth) - 1))
        topk = topk[:active_heads]
    elif tree_mode == "concurrency_aware":
        budget = math.floor(int(cpeak_nodes) / active_batch_size)
        budget = max(int(min_tree_nodes_per_seq), min(int(max_tree_nodes_per_seq), budget))
        topk = []
        parent_paths = 1
        used_nodes = 1
        max_heads = min(int(num_medusa_heads), max(0, int(max_tree_depth) - 1))
        defaults = fixed_tree_topk_by_depth or [4, 3, 2, 1, 1]
        for depth in range(max_heads):
            default_k = int(defaults[min(depth, len(defaults) - 1)])
            room = budget - used_nodes
            if room <= 0:
                break
            k = min(max(1, default_k), max(1, room // parent_paths))
            if used_nodes + parent_paths * k > budget:
                k = room // parent_paths
            if k <= 0:
                break
            topk.append(int(k))
            parent_paths *= int(k)
            used_nodes += parent_paths
        active_heads = len(topk)
    else:
        raise ValueError(f"Unsupported tree_mode={tree_mode}")
    return TreePlan(
        node_budget_per_seq=budget,
        active_heads=active_heads,
        topk_by_depth=topk,
        actual_nodes=dense_node_count(topk),
        mode=tree_mode,
        layout=tree_layout,
    )


def _unique_topk(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    if k <= 0:
        return [], []
    values, indices = torch.topk(logits.float(), k=min(int(k) * 2, logits.shape[-1]), dim=-1)
    seen = set()
    toks: list[int] = []
    scores: list[float] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token = int(index)
        if token in seen:
            continue
        seen.add(token)
        toks.append(token)
        scores.append(float(value))
        if len(toks) >= k:
            break
    return toks, scores


def build_dense_tree(root_token: int, medusa_logits: list[torch.Tensor], plan: TreePlan) -> CandidateTree:
    tokens = [int(root_token)]
    parents = [-1]
    depths = [1]
    scores = [0.0]
    current_parents = [0]
    for depth_idx, k in enumerate(plan.topk_by_depth):
        top_tokens, top_scores = _unique_topk(medusa_logits[depth_idx], int(k))
        if not top_tokens:
            break
        next_parents = []
        for parent in current_parents:
            for token, score in zip(top_tokens, top_scores):
                tokens.append(int(token))
                parents.append(int(parent))
                depths.append(depth_idx + 2)
                scores.append(float(score))
                next_parents.append(len(tokens) - 1)
                if len(tokens) >= plan.node_budget_per_seq:
                    break
            if len(tokens) >= plan.node_budget_per_seq:
                break
        current_parents = next_parents
        if len(tokens) >= plan.node_budget_per_seq:
            break
    return CandidateTree(tokens=tokens, parents=parents, depths=depths, scores=scores)


def build_batch_trees(root_tokens: torch.Tensor, medusa_logits: list[torch.Tensor], plan: TreePlan) -> list[CandidateTree]:
    """Build standard MEDUSA trees with one top-k launch per depth.

    The previous implementation launched ``topk`` and synchronized CUDA once
    per sequence and depth. Standard MEDUSA heads are independent of the tree
    parent, so their top-k candidates can be extracted for the whole batch at
    once and the small tree structures can then be assembled on CPU.
    """

    batch_size = int(root_tokens.shape[0])
    root_cpu = root_tokens.detach().cpu().tolist()
    candidates: list[tuple[list[list[int]], list[list[float]]]] = []
    for depth_idx, requested_k in enumerate(plan.topk_by_depth[: plan.active_heads]):
        if depth_idx >= len(medusa_logits):
            break
        logits = medusa_logits[depth_idx]
        k = min(max(0, int(requested_k)), int(logits.shape[-1]))
        if k <= 0:
            break
        values, indices = torch.topk(logits.float(), k=k, dim=-1)
        candidates.append((indices.detach().cpu().tolist(), values.detach().cpu().tolist()))

    trees: list[CandidateTree] = []
    for row in range(batch_size):
        tokens = [int(root_cpu[row])]
        parents = [-1]
        depths = [1]
        scores = [0.0]
        current_parents = [0]
        for depth_idx, (token_rows, score_rows) in enumerate(candidates):
            next_parents: list[int] = []
            for parent in current_parents:
                for token, score in zip(token_rows[row], score_rows[row]):
                    tokens.append(int(token))
                    parents.append(int(parent))
                    depths.append(depth_idx + 2)
                    scores.append(float(score))
                    next_parents.append(len(tokens) - 1)
                    if len(tokens) >= plan.node_budget_per_seq:
                        break
                if len(tokens) >= plan.node_budget_per_seq:
                    break
            current_parents = next_parents
            if not current_parents or len(tokens) >= plan.node_budget_per_seq:
                break
        trees.append(CandidateTree(tokens=tokens, parents=parents, depths=depths, scores=scores))
    return trees
