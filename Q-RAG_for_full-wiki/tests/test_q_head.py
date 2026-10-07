"""Calibration head Q = w·(s·M[a]) + b.

Without it an overfitting run failed: logits s·M ≈ ±20 against targets in
[0,1], and MSE turned the tower anti-correlated with its initialization
(cosine −0.25, own-pool recall 0) before the policy could learn anything.
The tests pin three invariants: line B checkpoints see no new keys; the
search ranking does not depend on the calibration; V(s') is computed on the
same scale as the heads in the loss.
"""

import torch

from rl.q_module import TextQNet, calibrated_soft_value, masked_soft_value


def make_qnet(q_head=None) -> TextQNet:
    return TextQNet(
        state_embed=torch.nn.Identity(),
        action_embed=torch.nn.Identity(),
        q_head=q_head,
    )


def test_uncalibrated_qnet_has_no_head_keys() -> None:
    """Line B checkpoints load strictly and must not see new keys."""
    qnet = make_qnet()
    assert not qnet.calibrated
    assert "q_scale" not in qnet.state_dict()
    assert "q_bias" not in qnet.state_dict()


def test_uncalibrated_head_values_keep_the_legacy_scale() -> None:
    l1 = torch.tensor([0.1, -0.4])
    l2 = torch.tensor([0.3, 0.2])
    h1, h2 = make_qnet().head_values(l1, l2)
    assert torch.equal(h1, 2 * l1)
    assert torch.equal(h2, 2 * l2)


def test_calibrated_head_values_follow_the_affine_formula() -> None:
    qnet = make_qnet(q_head={"scale_init": 0.05, "lr": 1e-3})
    assert qnet.calibrated
    l1 = torch.tensor([10.0, -8.0])
    l2 = torch.tensor([6.0, 2.0])
    h1, h2 = qnet.head_values(l1, l2)
    assert torch.allclose(h1, 0.05 * 2 * l1)
    assert torch.allclose(h2, 0.05 * 2 * l2)


def test_calibration_starts_in_the_reward_scale() -> None:
    """With ‖s‖ ≈ 20 the initial heads must land on the reward scale."""
    qnet = make_qnet(q_head={"scale_init": 1 / 20.8, "lr": 1e-3})
    dots = torch.tensor([18.0, 9.0, -3.0])  # realistic s·M at ‖s‖≈20.8
    h1, _ = qnet.head_values(dots / 2, dots / 2)
    assert h1.abs().max() < 2.0


def test_calibration_preserves_the_search_ranking() -> None:
    """The search ranks by the raw dot product; for w > 0 calibration is monotone."""
    torch.manual_seed(0)
    l1 = torch.randn(64)
    l2 = torch.randn(64)
    qnet = make_qnet(q_head={"scale_init": 0.048, "lr": 1e-3})
    h1, h2 = qnet.head_values(l1, l2)
    raw_order = torch.argsort(l1 + l2, descending=True)
    calibrated_order = torch.argsort(h1 + h2, descending=True)
    assert torch.equal(raw_order, calibrated_order)


def test_calibrated_soft_value_matches_the_head_scale() -> None:
    """V(s') lives on the scale of the loss heads and does not double the estimate."""
    logits_1 = torch.tensor([[10.0, 8.0, -2.0]])
    logits_2 = torch.tensor([[9.0, 7.5, -1.0]])
    available = torch.ones_like(logits_1, dtype=torch.bool)
    scale = torch.tensor(0.05)
    bias = torch.tensor(0.0)

    value = calibrated_soft_value(
        logits_1, logits_2, available, alpha=0.005, top_k_actions=3,
        scale=scale, bias=bias,
    )
    # Soft maximum ≈ maximum of the calibrated head: (0.05·20 + 0.05·18)/2.
    expected = 0.5 * (0.05 * 2 * 10.0 + 0.05 * 2 * 9.0)
    assert torch.allclose(value, torch.tensor([expected]), atol=0.05)


def test_calibrated_soft_value_still_ignores_masked_argmax() -> None:
    """The mask applied before topk must survive calibration."""
    logits_1 = torch.tensor([[100.0, 8.0, -2.0]])
    logits_2 = torch.tensor([[100.0, 7.5, -1.0]])
    available = torch.tensor([[False, True, True]])

    masked = calibrated_soft_value(
        logits_1, logits_2, available, alpha=0.005, top_k_actions=3,
        scale=torch.tensor(0.05), bias=torch.tensor(0.0),
    )
    without_row = calibrated_soft_value(
        logits_1[:, 1:], logits_2[:, 1:], available[:, 1:],
        alpha=0.005, top_k_actions=3,
        scale=torch.tensor(0.05), bias=torch.tensor(0.0),
    )
    assert torch.allclose(masked, without_row)


def test_legacy_soft_value_is_untouched() -> None:
    """Line B keeps using the uncalibrated v1 + v2 path."""
    logits = torch.tensor([[1.0, 0.5, -0.5]])
    available = torch.ones_like(logits, dtype=torch.bool)
    v1, v2 = masked_soft_value(logits, logits, available, 0.005, 3)
    assert torch.allclose(v1, v2)
    assert float(v1[0]) != 0.0
