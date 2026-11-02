import torch

from saseg.e1_train import (
    ValidationAnomalyMonitor,
    known_dice_from_counts,
    per_pixel_loss,
    prediction_scores,
)
from saseg.losses import binary_branch_loss, class_agnostic_loss, saseg_loss


def test_per_pixel_variants_are_finite():
    target = torch.tensor([[[0, 1], [2, 3]]])
    assert torch.isfinite(per_pixel_loss(torch.randn(1, 3, 2, 2), target, 2, False))
    assert torch.isfinite(per_pixel_loss(torch.randn(1, 4, 2, 2), target, 2, True))


def test_per_pixel_scores_have_pixel_shape():
    scores = prediction_scores("perpixel_k1", torch.randn(2, 5, 8, 8), 3)
    assert set(scores) == {"msp", "energy", "maxlogit", "unknown_class"}
    assert all(value.shape == (2, 8, 8) for value in scores.values())


def _saseg_outputs(num_known=2, unknown=2, size=4, batch=2):
    return {
        "logits": torch.randn(batch, num_known + unknown + 2, size, size),
        "known_logits": torch.randn(batch, num_known, size, size),
        "unknown_logits": torch.randn(batch, unknown, size, size),
        "unknown_probs": torch.rand(batch, unknown, size, size),
        "unknown_union": torch.zeros(batch, size, size) if unknown == 0 else torch.rand(batch, size, size),
        "confidence": torch.rand(batch, num_known),
        "confidence_logits": torch.randn(batch, num_known),
        "support_score": torch.randn(batch, num_known),
        "unknown_support_score": torch.randn(batch, unknown),
        "region_embeddings": torch.randn(batch, num_known, 8),
        "unknown_embeddings": torch.randn(batch, unknown, 8),
    }


def test_saseg_scores_include_ensemble_and_respect_query_count():
    scores = prediction_scores("saseg", _saseg_outputs(), 2)
    assert set(scores) == {"msp", "energy", "maxlogit", "unknown_union", "ensemble", "support"}
    assert all(value.shape == (2, 4, 4) for value in scores.values())
    zero = prediction_scores("saseg", _saseg_outputs(unknown=0), 2)
    assert zero["unknown_union"].abs().max() == 0


def test_generic_queries_have_no_unknown_score():
    scores = prediction_scores("saseg_generic", _saseg_outputs(), 2)
    assert "unknown_union" not in scores
    assert "objectness" in scores


def test_binary_branch_and_class_agnostic_scores():
    binary = {"known_logits": torch.randn(2, 3, 4, 4), "unknown_logits": torch.randn(2, 1, 4, 4),
              "unknown_union": torch.rand(2, 4, 4)}
    scores = prediction_scores("binary_branch", binary, 2)
    assert set(scores) == {"unknown_branch", "msp", "energy", "maxlogit"}
    agnostic = {"pixel_logits": torch.randn(2, 4, 4, 4), "known_logits": torch.randn(2, 2, 4, 4),
                "unknown_union": torch.rand(2, 4, 4)}
    scores = prediction_scores("saseg_classagnostic", agnostic, 2)
    assert set(scores) == {"unknown_class", "msp", "energy", "maxlogit"}


def test_new_losses_are_finite():
    target = torch.tensor([[[0, 1], [2, 3]]])
    outputs = _saseg_outputs(size=2, batch=1)
    assert torch.isfinite(saseg_loss(outputs, target, 2)["total"])
    assert torch.isfinite(saseg_loss(outputs, target, 2, generic_queries=True)["total"])
    assert torch.isfinite(saseg_loss(_saseg_outputs(size=2, batch=1, unknown=0), target, 2)["total"])
    binary = {"known_logits": torch.randn(1, 3, 2, 2), "unknown_logits": torch.randn(1, 1, 2, 2)}
    assert torch.isfinite(binary_branch_loss(binary, target, 2)["total"])
    agnostic = {"pixel_logits": torch.randn(1, 4, 2, 2)}
    assert torch.isfinite(class_agnostic_loss(agnostic, target, 2)["total"])


def test_known_dice_uses_only_foreground_classes_present_in_validation():
    result = known_dice_from_counts(
        intersection=torch.tensor([0, 5, 0]).numpy(),
        predicted=torch.tensor([0, 10, 4]).numpy(),
        truth=torch.tensor([0, 10, 0]).numpy(),
        num_known=2,
    )
    assert result["macro_dice"] == 0.5
    assert result["classes_present"] == 1


def test_validation_zero_metrics_warn_then_stop_after_patience():
    monitor = ValidationAnomalyMonitor(patience=3)
    metrics = {
        "known_segmentation": {"validation": {"macro_dice": 0.0}},
        "msp": {"validation": {"f1": 0.0}},
        "energy": {"validation": {"f1": 0.0}},
    }
    assert [event["type"] for event in monitor.update(metrics, 0)] == ["warning", "warning"]
    assert [event["type"] for event in monitor.update(metrics, 1)] == ["warning", "warning"]
    assert [event["type"] for event in monitor.update(metrics, 2)] == ["stop", "stop"]


def test_validation_anomaly_streak_resets_after_recovery():
    monitor = ValidationAnomalyMonitor(patience=3)
    bad = {"known_segmentation": {"validation": {"macro_dice": 0.0}},
           "msp": {"validation": {"f1": 0.0}}}
    good = {"known_segmentation": {"validation": {"macro_dice": 0.4}},
            "msp": {"validation": {"f1": 0.2}}}
    monitor.update(bad, 0)
    monitor.update(good, 1)
    assert [event["type"] for event in monitor.update(bad, 2)] == ["warning", "warning"]
