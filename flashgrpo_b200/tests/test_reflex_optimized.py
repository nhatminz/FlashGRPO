import torch

from flashgrpo_b200.decoding.flash_medusa_decoder import FlashMedusaConfig, FlashMedusaDecoder
from flashgrpo_b200.decoding.medusa_tree import TreePlan, build_batch_trees, build_dense_tree
from flashgrpo_b200.decoding.reflex import (
    LMHeadFeedback,
    PredictionRecord,
    ReflexAuxiliaryRecordBuffer,
    ReflexStateManager,
)
from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.training.flashgrpo_trainer import _merge_reflex_record_batches
from flashgrpo_b200.training.online_medusa_trainer import OnlineMedusaTrainer


def test_batched_tree_builder_matches_reference_tree():
    torch.manual_seed(7)
    batch, vocab = 5, 37
    roots = torch.randint(vocab, (batch,))
    logits = [torch.randn(batch, vocab) for _ in range(2)]
    plan = TreePlan(
        node_budget_per_seq=12,
        active_heads=2,
        topk_by_depth=[3, 2],
        actual_nodes=10,
        mode="fixed",
        layout="dense",
    )
    batched = build_batch_trees(roots, logits, plan)
    reference = [
        build_dense_tree(int(roots[row]), [head[row] for head in logits], plan)
        for row in range(batch)
    ]
    assert [(tree.tokens, tree.parents) for tree in batched] == [
        (tree.tokens, tree.parents) for tree in reference
    ]


def test_sparse_target_feedback_skips_matching_distribution():
    torch.manual_seed(11)
    vocab, hidden = 17, 8
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    logits = torch.randn(vocab)
    log_z = float(torch.logsumexp(logits, dim=-1))
    record = PredictionRecord(
        sequence_id=0,
        anchor_pos=3,
        target_pos=5,
        horizon=2,
        top_ids=torch.arange(vocab, dtype=torch.int32),
        top_logits=logits.to(torch.float16),
        logsumexp=log_z,
    )
    result = LMHeadFeedback(
        lm_head,
        target_topk=vocab,
        union_cap=vocab,
    ).compute_batch([[record]], logits.unsqueeze(0), [0])
    assert not bool(result.has_feedback.item())
    assert float(result.record_tv.item()) < 1e-3
    assert torch.count_nonzero(result.feedback) == 0


def test_hidden_state_update_and_correction_are_rms_bounded():
    torch.manual_seed(13)
    batch, hidden, vocab = 4, 32, 41
    manager = ReflexStateManager(batch, hidden, device=torch.device("cpu"))
    feedback = torch.randn(batch, hidden)
    manager.advance_token(
        list(range(batch)),
        feedback,
        torch.ones(batch, dtype=torch.bool),
        torch.ones(batch),
    )
    assert float(manager.states.square().mean(dim=-1).sqrt().max()) <= 2.0

    heads = MedusaHeads(hidden, vocab, num_heads=2, dtype=torch.float32)
    config = FlashMedusaConfig(
        reflex_enabled=True,
        reflex_state_space="hidden",
        reflex_feedback_enabled=True,
        reflex_proposal_injection_enabled=True,
        reflex_proposal_injection_scale=1.0,
        reflex_relative_rms_delta_base=0.01,
    )
    decoder = FlashMedusaDecoder(object(), heads, object(), config)
    base_hidden = torch.randn(batch, hidden)
    corrected = decoder._apply_reflex_correction(
        base_hidden,
        manager.states,
        torch.full((batch,), 100.0),
        head_idx=0,
        generation_step=0,
    )
    correction_rms = (corrected - base_hidden).float().square().mean(dim=-1).sqrt()
    base_rms = base_hidden.float().square().mean(dim=-1).sqrt()
    assert bool((correction_rms <= 1.011 * 0.01 * base_rms).all())


def test_sparse_refresh_loss_and_cache_teacher_round_trip():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    ids = torch.tensor([[0, 1]], dtype=torch.long)
    values = logits[:, :2]
    log_z = torch.logsumexp(logits, dim=-1)
    _, matching_tv = OnlineMedusaTrainer._sparse_cross_entropy_with_tail(
        logits,
        ids,
        values,
        log_z,
    )
    _, shifted_tv = OnlineMedusaTrainer._sparse_cross_entropy_with_tail(
        logits.flip(-1),
        ids,
        values,
        log_z,
    )
    assert float(matching_tv.item()) < 1e-6
    assert float(shifted_tv.item()) > float(matching_tv.item())

    buffer = ReflexAuxiliaryRecordBuffer(max_records=4)
    buffer.add_anchor_predictions(
        sequence_ids=[0],
        anchor_positions=torch.tensor([10]),
        initial_lengths=torch.tensor([5]),
        hidden_states=torch.randn(1, 8),
        fast_states=None,
        max_horizon=2,
    )
    proposal = PredictionRecord(
        sequence_id=0,
        anchor_pos=10,
        target_pos=12,
        horizon=2,
        top_ids=torch.tensor([0, 1], dtype=torch.int32),
        top_logits=torch.tensor([2.0, 1.0], dtype=torch.float16),
        logsumexp=float(log_z.item()),
    )
    buffer.pop_mature(
        0,
        12,
        generated_tokens=[3, 2, 1, 0, 1, 2, 3],
        true_token=0,
        teacher={
            "target_top_ids": torch.tensor([0, 1], dtype=torch.int32),
            "target_top_logits": torch.tensor([2.0, 1.0], dtype=torch.float16),
            "target_logsumexp": float(log_z.item()),
            "proposal_records": [proposal],
        },
    )
    batch = buffer.to_batch()
    assert bool(batch["has_sparse_teacher"].item())
    assert batch["target_top_ids"].shape == (1, 2)


def test_aux_cache_merge_pads_variable_sparse_supports():
    def records(count: int, prev_width: int, topk: int, offset: int) -> dict:
        return {
            "hidden": torch.full((count, 4), float(offset), dtype=torch.float16),
            "fast_state": torch.empty((count, 0), dtype=torch.float16),
            "labels": torch.arange(offset, offset + count),
            "horizons": torch.full((count,), 2, dtype=torch.long),
            "prev_lens": torch.full((count,), prev_width, dtype=torch.long),
            "prev_tokens": torch.full((count, prev_width), offset, dtype=torch.long),
            "target_top_ids": torch.full((count, topk), offset, dtype=torch.int32),
            "target_top_logits": torch.ones((count, topk), dtype=torch.float16),
            "target_logsumexp": torch.ones(count),
            "old_top_ids": torch.full((count, topk), offset, dtype=torch.int32),
            "old_top_logits": torch.ones((count, topk), dtype=torch.float16),
            "old_logsumexp": torch.ones(count),
            "has_sparse_teacher": torch.ones(count, dtype=torch.bool),
        }

    merged = _merge_reflex_record_batches(
        [records(2, 1, 2, 0), records(3, 2, 4, 10)],
        max_records=4,
    )
    assert merged["hidden"].shape == (4, 4)
    assert merged["prev_tokens"].shape == (4, 2)
    assert merged["target_top_ids"].shape == (4, 4)
    assert merged["old_top_ids"].shape == (4, 4)
    # The cap keeps the newest records after padding/concatenation.
    assert merged["labels"].tolist() == [1, 10, 11, 12]
