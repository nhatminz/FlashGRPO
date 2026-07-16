import torch

from flashgrpo_b200.decoding.acceptance import sample_from_logits
from flashgrpo_b200.decoding.medusa_tree import CandidateTree
from flashgrpo_b200.decoding.acceptance import exact_accept_path


def test_root_only_sampling_matches_target_sampler():
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5]])
    torch.manual_seed(123)
    vanilla = sample_from_logits(logits, do_sample=True, temperature=1.0, top_p=0.95)
    torch.manual_seed(123)
    flash_root = sample_from_logits(logits, do_sample=True, temperature=1.0, top_p=0.95)
    assert int(vanilla.item()) == int(flash_root.item())


def test_exact_accepts_only_target_sampled_child():
    tree = CandidateTree(tokens=[5, 7, 9], parents=[-1, 0, 0], depths=[1, 2, 2])
    logits = torch.full((3, 16), -10.0)
    logits[0, 7] = 10.0
    accepted, nodes, _ = exact_accept_path(tree, logits, do_sample=False, temperature=1.0, top_p=1.0, top_k=None)
    assert accepted == [5, 7]
    assert nodes == [0, 1]
