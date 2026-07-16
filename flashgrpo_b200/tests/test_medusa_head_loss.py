import torch

from flashgrpo_b200.models.medusa_heads import MedusaHeads


def test_medusa_head_loss_decreases_on_toy_batch():
    torch.manual_seed(0)
    hidden_size = 16
    vocab_size = 32
    heads = MedusaHeads(hidden_size, vocab_size, num_heads=2, tie_lm_head=False)
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-2)
    hidden = torch.randn(2, 12, hidden_size)
    input_ids = torch.randint(0, vocab_size, (2, 12))
    attention_mask = torch.ones(2, 12, dtype=torch.long)
    losses = []
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        loss, _ = heads.compute_loss(hidden, input_ids, attention_mask, chunk_size=4)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
