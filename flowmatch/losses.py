import torch

def epe_loss(list_flow_pred, flow_target, tg_mask):
    """EPE Loss for a batch
    Args:
        list_flow_pred (list(Tensor, shape=Bx2xHxH)): Predicted flows.
        flow_target (Tensor, shape=Bx2xHxH): Target flow.
        tg_mask (Tensor, type=float32, shape=BxHxH): Target image mask.
    Returns:
        losses (list(Tensor, type=float32, shape=(,)): List of losses for each flow_pred.
    """
    tg_mask = tg_mask.unsqueeze(1)  # shape is Bx1xHxH
    mask_size = tg_mask.sum(dim=(1, 2, 3))  # batch-wise number of pixels in mask

    losses = []
    for flow_pred in list_flow_pred:
        # Zero diff for pixels outside mask.
        diff = tg_mask * (flow_pred - flow_target)  # shape (B, 2, H, H)
        norm = torch.norm(diff, p=2, dim=1)  # shape (B, H, H)
        batchwize_loss = norm.sum(dim=(1, 2))  # shape (B,)
        loss = (batchwize_loss / mask_size).mean()
        losses.append(loss)

    return losses
