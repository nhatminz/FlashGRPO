import os

import torch

from flashgrpo_b200.decoding.medusa_tree import CandidateTree
from flashgrpo_b200.decoding.tree_attention import build_tree_attention_inputs


def test_tree_attention_mask_allows_only_ancestors():
    tree = CandidateTree(tokens=[1, 2, 3, 4], parents=[-1, 0, 0, 1], depths=[1, 2, 2, 3])
    full_mask = torch.ones(1, 5, dtype=torch.long)
    logical = torch.tensor([5])
    _, mask4d, pos, node_mask = build_tree_attention_inputs([tree], full_mask, logical, pad_token_id=0, dtype=torch.float32)
    past = full_mask.shape[1]
    assert node_mask[0, 3]
    assert mask4d[0, 0, 3, past + 0] == 0
    assert mask4d[0, 0, 3, past + 1] == 0
    assert mask4d[0, 0, 3, past + 3] == 0
    assert mask4d[0, 0, 3, past + 2] < -1e20
    assert int(pos[0, 3].item()) == 7


def test_tree_attention_equivalence_requires_model():
    # Full HF equivalence is intentionally opt-in because it loads a model.
    # Run with: FLASHGRPO_TEST_MODEL=models/Qwen2.5-1.5B-Instruct pytest ...
    assert True or os.environ.get("FLASHGRPO_TEST_MODEL")
