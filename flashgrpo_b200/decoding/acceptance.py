from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_probs(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = logits.float()
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        top_values, top_indices = torch.topk(logits, k=int(top_k), dim=-1)
        filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = filtered.scatter(-1, top_indices, top_values)
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > float(top_p)
        mask = torch.roll(mask, shifts=1, dims=-1)
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return probs


def sample_from_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if (not do_sample) or temperature == 0:
        return torch.argmax(logits, dim=-1)
    probs = logits_to_probs(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(logits.shape[:-1])


def exact_accept_path(
    tree,
    tree_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> tuple[list[int], list[int], int]:
    """Walk one MEDUSA tree using target samples only.

    The root token has already been sampled from target logits before tree
    construction. Future tokens are accepted only when a fresh target sample
    from the current parent distribution is present among that parent node's
    children.
    """

    accepted_tokens = [int(tree.tokens[0])]
    accepted_nodes = [0]
    parent = 0
    while True:
        children = tree.children.get(parent, [])
        if not children:
            break
        sampled = int(
            sample_from_logits(
                tree_logits[parent : parent + 1],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).item()
        )
        match = None
        for child in children:
            if int(tree.tokens[child]) == sampled:
                match = child
                break
        if match is None:
            break
        accepted_tokens.append(sampled)
        accepted_nodes.append(match)
        parent = match
    return accepted_tokens, accepted_nodes, parent


@torch.no_grad()
def exact_accept_paths_batch(
    trees,
    tree_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Batched equivalent of :func:`exact_accept_path`.

    Target samples at the same tree depth are drawn in one CUDA operation.
    Candidate lookup remains on CPU because ``CandidateTree`` is a compact
    Python structure. The accepted distribution is unchanged: every token is
    still sampled solely from the target policy at its current parent node.
    """

    batch_size = len(trees)
    accepted_tokens = [[int(tree.tokens[0])] for tree in trees]
    accepted_nodes = [[0] for _ in trees]
    parent_nodes = [0 for _ in trees]
    active_rows = [row for row, tree in enumerate(trees) if tree.children.get(0)]

    while active_rows:
        row_index = torch.as_tensor(active_rows, dtype=torch.long, device=tree_logits.device)
        parent_index = torch.as_tensor(
            [parent_nodes[row] for row in active_rows],
            dtype=torch.long,
            device=tree_logits.device,
        )
        parent_logits = tree_logits[row_index, parent_index]
        sampled = sample_from_logits(
            parent_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ).detach().cpu().tolist()

        next_active: list[int] = []
        for row, token in zip(active_rows, sampled):
            tree = trees[row]
            match = next(
                (
                    child
                    for child in tree.children.get(parent_nodes[row], [])
                    if int(tree.tokens[child]) == int(token)
                ),
                None,
            )
            if match is None:
                continue
            accepted_tokens[row].append(int(token))
            accepted_nodes[row].append(int(match))
            parent_nodes[row] = int(match)
            if tree.children.get(int(match)):
                next_active.append(row)
        active_rows = next_active

    if len(accepted_tokens) != batch_size:
        raise RuntimeError("Batched acceptance produced an invalid batch size")
    return accepted_tokens, accepted_nodes, parent_nodes
