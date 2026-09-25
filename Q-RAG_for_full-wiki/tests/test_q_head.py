"""Калибровочная голова Q = w·(s·M[a]) + b.

Причина её существования — провал переобучения 2026-08-02: логиты s·M ≈ ±20
при таргетах [0,1], и MSE разворачивал башню в анти-корреляцию со стартом
(косинус −0.25, recall собственного пула 0) раньше, чем политика успевала
чему-то научиться. Тесты фиксируют три инварианта: чекпоинты линии B не
видят новых ключей; ранжирование поиска не зависит от калибровки; V(s')
считается в том же масштабе, что и головы в лоссе.
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
    """Чекпоинты линии B грузятся strict и не должны видеть новых ключей."""
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
    """Стартовые головы при ‖s‖ ≈ 20 обязаны попасть в масштаб награды."""
    qnet = make_qnet(q_head={"scale_init": 1 / 20.8, "lr": 1e-3})
    dots = torch.tensor([18.0, 9.0, -3.0])  # реалистичные s·M при ‖s‖≈20.8
    h1, _ = qnet.head_values(dots / 2, dots / 2)
    assert h1.abs().max() < 2.0


def test_calibration_preserves_the_search_ranking() -> None:
    """Поиск ранжирует сырой dot; при w > 0 калибровка монотонна."""
    torch.manual_seed(0)
    l1 = torch.randn(64)
    l2 = torch.randn(64)
    qnet = make_qnet(q_head={"scale_init": 0.048, "lr": 1e-3})
    h1, h2 = qnet.head_values(l1, l2)
    raw_order = torch.argsort(l1 + l2, descending=True)
    calibrated_order = torch.argsort(h1 + h2, descending=True)
    assert torch.equal(raw_order, calibrated_order)


def test_calibrated_soft_value_matches_the_head_scale() -> None:
    """V(s') живёт там же, где головы лосса, и не удваивает оценку."""
    logits_1 = torch.tensor([[10.0, 8.0, -2.0]])
    logits_2 = torch.tensor([[9.0, 7.5, -1.0]])
    available = torch.ones_like(logits_1, dtype=torch.bool)
    scale = torch.tensor(0.05)
    bias = torch.tensor(0.0)

    value = calibrated_soft_value(
        logits_1, logits_2, available, alpha=0.005, top_k_actions=3,
        scale=scale, bias=bias,
    )
    # Мягкий максимум ≈ максимум калиброванной головы: (0.05·20 + 0.05·18)/2.
    expected = 0.5 * (0.05 * 2 * 10.0 + 0.05 * 2 * 9.0)
    assert torch.allclose(value, torch.tensor([expected]), atol=0.05)


def test_calibrated_soft_value_still_ignores_masked_argmax() -> None:
    """Маска до topk обязана пережить калибровку."""
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
    """Линия B продолжает жить на некалиброванном пути v1 + v2."""
    logits = torch.tensor([[1.0, 0.5, -0.5]])
    available = torch.ones_like(logits, dtype=torch.bool)
    v1, v2 = masked_soft_value(logits, logits, available, 0.005, 3)
    assert torch.allclose(v1, v2)
    assert float(v1[0]) != 0.0
