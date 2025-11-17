import torch

def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: The computed MAE.
    """
    return torch.mean(torch.abs(predictions - targets))


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE) between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: The computed MSE.
    """
    return torch.mean((predictions - targets) ** 2)


def compute_r2(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute R-squared (R²) between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: The computed R² value.
    """
    ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
    ss_residual = torch.sum((targets - predictions) ** 2)
    return 1 - (ss_residual / ss_total)