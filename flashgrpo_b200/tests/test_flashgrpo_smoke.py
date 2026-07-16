from flashgrpo_b200.decoding.medusa_tree import plan_tree


def test_concurrency_tree_shrinks_with_batch():
    small_b = plan_tree(
        active_batch_size=1,
        num_medusa_heads=3,
        tree_mode="concurrency_aware",
        tree_layout="dense",
        cpeak_nodes=32,
        min_tree_nodes_per_seq=1,
        max_tree_nodes_per_seq=16,
        max_tree_depth=4,
        fixed_tree_topk_by_depth=[4, 3, 2],
    )
    large_b = plan_tree(
        active_batch_size=16,
        num_medusa_heads=3,
        tree_mode="concurrency_aware",
        tree_layout="dense",
        cpeak_nodes=32,
        min_tree_nodes_per_seq=1,
        max_tree_nodes_per_seq=16,
        max_tree_depth=4,
        fixed_tree_topk_by_depth=[4, 3, 2],
    )
    assert small_b.actual_nodes >= large_b.actual_nodes
