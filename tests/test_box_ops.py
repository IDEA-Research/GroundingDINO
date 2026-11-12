import torch

from groundingdino.util.box_ops import (
    box_iou,
    box_iou_pairwise,
    generalized_box_iou,
    generalized_box_iou_pairwise,
)


def test_box_iou_pairwise_zero_area_is_finite():
    # A collapsed box has union 0; the batched helper already avoids 0/0 with 1e-6.
    boxes = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    iou, union = box_iou_pairwise(boxes, boxes)
    assert torch.isfinite(iou).all()
    assert torch.isclose(iou[0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(iou[1], torch.tensor(1.0), atol=1e-5)
    assert union[0] == 0


def test_generalized_box_iou_pairwise_zero_area_is_finite():
    boxes = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    giou = generalized_box_iou_pairwise(boxes, boxes)
    assert torch.isfinite(giou).all()
    # Matches the non-pairwise helper, which already divides by (area + 1e-6).
    expected = generalized_box_iou(boxes, boxes).squeeze()
    assert torch.isclose(giou[0], expected, atol=1e-5)
